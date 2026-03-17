"""IndexTTS 2.0 MLX Inference.

This module provides the main inference pipeline for IndexTTS 2.0 using MLX.
Uses PyTorch only for preprocessing (W2V-BERT, SemanticCodec, CAMPPlus).
GPT, S2Mel, and BigVGAN all run on MLX.

Architecture:
- PyTorch: W2V-BERT, SemanticCodec, CAMPPlus (preprocessing only)
- MLX: GPT v2 (autoregressive), S2Mel (CFM), BigVGAN v2 (vocoder)
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# PyTorch imports (for preprocessing only)
import torch
import torchaudio
import librosa

import mlx.core as mx
import mlx.nn as nn

from omegaconf import OmegaConf


# 8 emotion categories in IndexTTS 2.0
EMOTION_CATEGORIES = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
EMOTION_CN_TO_EN = {
    "高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid",
    "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised", "自然": "calm",
}
# Number of vectors per emotion category in emo_matrix
EMO_NUM = [3, 17, 2, 8, 4, 5, 10, 24]  # sum = 73


def parse_emotion(emotion_str: str) -> Dict[str, float]:
    """Parse emotion string into emotion weights dict.

    Supports formats:
    - Single emotion: "happy" -> {"happy": 1.0}
    - Weighted: "happy:0.8,sad:0.2" -> {"happy": 0.8, "sad": 0.2}
    - JSON-like: '{"happy": 0.8, "sad": 0.2}'

    Args:
        emotion_str: Emotion specification string

    Returns:
        Dict mapping emotion names to weights (0.0-1.2)
    """
    import json

    emotion_str = emotion_str.strip()

    # Try JSON format first
    if emotion_str.startswith("{"):
        try:
            return json.loads(emotion_str)
        except json.JSONDecodeError:
            pass

    # Parse comma-separated format
    result = {}
    for part in emotion_str.split(","):
        part = part.strip()
        if ":" in part:
            name, weight = part.split(":", 1)
            name = name.strip().lower()
            weight = float(weight.strip())
        else:
            name = part.lower()
            weight = 1.0

        # Map Chinese to English if needed
        if name in EMOTION_CN_TO_EN:
            name = EMOTION_CN_TO_EN[name]

        if name in EMOTION_CATEGORIES:
            result[name] = max(0.0, min(1.2, weight))
        else:
            print(f"Warning: Unknown emotion '{name}', ignored. Valid: {EMOTION_CATEGORIES}")

    # Default to calm if empty
    if not result:
        result["calm"] = 1.0

    return result


class IndexTTSv2:
    """IndexTTS 2.0 with MLX GPT, S2Mel and BigVGAN.

    Uses PyTorch for preprocessing (W2V-BERT, SemanticCodec, CAMPPlus).
    GPT autoregressive generation, S2Mel CFM, and BigVGAN vocoder all run on MLX.
    """

    def __init__(
        self,
        model_dir: str,
        config_path: Optional[str] = None,
        device: str = "mps",
        mlx_model_dir: Optional[str] = None,
        memory_limit_gb: float = 0,
        quantize_bits: Optional[int] = None,
    ):
        """Initialize IndexTTS 2.0.

        Args:
            model_dir: Path to converted MLX model directory.
            config_path: Path to config.yaml (optional, defaults to model_dir/config.yaml)
            device: PyTorch device for preprocessing (mps, cuda, cpu)
            mlx_model_dir: Alias for model_dir (for backwards compatibility)
            memory_limit_gb: GPU memory limit in GB (0 = no limit)
            quantize_bits: Runtime quantization bits for GPT (4 or 8), None for no quantization
        """
        # Set memory limit if specified
        if memory_limit_gb > 0:
            mx.set_memory_limit(int(memory_limit_gb * 1024 * 1024 * 1024))

        self.model_dir = Path(model_dir)
        self.device = device
        self.quantize_bits = quantize_bits

        # mlx_model_dir is same as model_dir in unified structure
        self.mlx_model_dir = Path(mlx_model_dir) if mlx_model_dir else self.model_dir

        # Find config.yaml (could be in model_dir or mlx_model_dir)
        if config_path:
            self.config_path = config_path
        elif (self.model_dir / "config.yaml").exists():
            self.config_path = str(self.model_dir / "config.yaml")
        elif (self.mlx_model_dir / "config.yaml").exists():
            self.config_path = str(self.mlx_model_dir / "config.yaml")
        else:
            raise FileNotFoundError(f"config.yaml not found in {self.model_dir} or {self.mlx_model_dir}")

        # Determine weight paths based on directory structure
        # Support both old structure (separate dirs) and new unified structure
        if (self.mlx_model_dir / "gpt.safetensors").exists():
            # New unified structure
            self.gpt_weights_path = str(self.mlx_model_dir / "gpt.safetensors")
            self.s2mel_weights_path = str(self.mlx_model_dir / "s2mel.safetensors")
            self.bigvgan_weights_path = str(self.mlx_model_dir / "bigvgan.safetensors")
        else:
            # Legacy structure (separate directories)
            self.gpt_weights_path = "models/gpt_v2/gpt_v2.safetensors"
            self.s2mel_weights_path = "models/s2mel_v2/s2mel.safetensors"
            self.bigvgan_weights_path = "models/bigvgan_v2/bigvgan_v2.safetensors"

        # Load config
        self.cfg = OmegaConf.load(self.config_path)
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        print(f"Loading IndexTTS 2.0...")
        print(f"  Config: {self.config_path}")
        print(f"  MLX weights: {self.mlx_model_dir}")

        # PyTorch preprocessing modules (lazy loaded on first .wav processing)
        # Note: semantic_codec is always needed for vq2emb during generation
        self._preprocessing_initialized = False
        self.semantic_model = None
        self.campplus = None

        # Always load semantic_codec (needed for mel_codes -> embedding)
        self._init_semantic_codec()

        # Always load emotion matrices (small .pt files, needed for --emotion)
        self._load_emotion_matrices()

        # Initialize MLX models (GPT, S2Mel, BigVGAN)
        self._init_mlx_models()

        # Initialize tokenizer
        self._init_tokenizer()

        # Mel spectrogram config for reference audio
        self._init_mel_config()

        # Cache
        self.cache = {}

        print("IndexTTS 2.0 ready!")

    def _init_semantic_codec(self):
        """Initialize semantic codec (always needed for vq2emb during generation)."""
        from mlx_indextts.indextts.utils.maskgct_utils import build_semantic_codec

        print("Loading Semantic Codec...")
        self.semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        try:
            import safetensors.torch
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            safetensors.torch.load_model(self.semantic_codec, ckpt_path)
            print(f"  semantic_codec weights restored from: {ckpt_path}")
        except Exception as e:
            print(f"Warning: Failed to load semantic_codec weights: {e}")
        self.semantic_codec = self.semantic_codec.to(self.device)
        self.semantic_codec.eval()

    def _init_pytorch_modules(self):
        """Initialize PyTorch modules for .wav preprocessing.

        Loads W2V-BERT, CAMPPlus, and emotion matrices.
        Called lazily when processing .wav files (not needed for .npz).
        """
        from mlx_indextts.indextts.utils.maskgct_utils import build_semantic_model
        from mlx_indextts.indextts.s2mel.modules.campplus.DTDNN import CAMPPlus as CAMPPlusModel
        from transformers import AutoFeatureExtractor

        # Find w2v stats file
        w2v_stat_path = self.mlx_model_dir / self.cfg.w2v_stat
        if not w2v_stat_path.exists():
            w2v_stat_path = self.model_dir / self.cfg.w2v_stat
        if not w2v_stat_path.exists():
            raise FileNotFoundError(f"W2V stats file not found: {self.cfg.w2v_stat}")

        # W2V-BERT (Semantic Model)
        print("Loading W2V-BERT...")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=str(w2v_stat_path)
        )
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)
        self.extract_features = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        # CAMPPlus
        print("Loading CAMPPlus...")
        try:
            from huggingface_hub import hf_hub_download
            campplus_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
            self.campplus = CAMPPlusModel(feat_dim=80, embedding_size=192)
            state_dict = torch.load(campplus_path, map_location=self.device)
            self.campplus.load_state_dict(state_dict)
            print(f"  CAMPPlus weights restored from: {campplus_path}")
        except Exception as e:
            print(f"Warning: Failed to load CAMPPlus weights: {e}")
            self.campplus = CAMPPlusModel(feat_dim=80, embedding_size=192)
        self.campplus = self.campplus.to(self.device)
        self.campplus.eval()

    def _load_emotion_matrices(self):
        """Load emotion matrices for emotion control (small .pt files)."""
        emo_matrix_path = None
        spk_matrix_path = None

        for base_dir in [self.mlx_model_dir, self.model_dir]:
            test_emo = base_dir / "feat2.pt"
            test_spk = base_dir / "feat1.pt"
            if not test_emo.exists() and hasattr(self.cfg, 'emo_matrix'):
                test_emo = base_dir / self.cfg.emo_matrix
                test_spk = base_dir / self.cfg.spk_matrix
            if test_emo.exists() and test_spk.exists():
                emo_matrix_path = test_emo
                spk_matrix_path = test_spk
                break

        if emo_matrix_path and spk_matrix_path:
            self.emo_matrix = torch.load(str(emo_matrix_path), map_location=self.device)
            self.spk_matrix = torch.load(str(spk_matrix_path), map_location=self.device)
            self.emo_matrix_split = torch.split(self.emo_matrix, EMO_NUM)
            self.spk_matrix_split = torch.split(self.spk_matrix, EMO_NUM)
        else:
            self.emo_matrix = None
            self.spk_matrix = None
            self.emo_matrix_split = None
            self.spk_matrix_split = None

    def _init_pytorch_modules(self):
        """Initialize PyTorch modules for .wav preprocessing.

        Loads W2V-BERT and CAMPPlus.
        Called lazily when processing .wav files (not needed for .npz).
        """
        from mlx_indextts.indextts.utils.maskgct_utils import build_semantic_model
        from mlx_indextts.indextts.s2mel.modules.campplus.DTDNN import CAMPPlus as CAMPPlusModel
        from transformers import AutoFeatureExtractor

        # Find w2v stats file
        w2v_stat_path = self.mlx_model_dir / self.cfg.w2v_stat
        if not w2v_stat_path.exists():
            w2v_stat_path = self.model_dir / self.cfg.w2v_stat
        if not w2v_stat_path.exists():
            raise FileNotFoundError(f"W2V stats file not found: {self.cfg.w2v_stat}")

        # W2V-BERT (Semantic Model)
        print("Loading W2V-BERT...")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=str(w2v_stat_path)
        )
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)
        self.extract_features = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        # CAMPPlus
        print("Loading CAMPPlus...")
        try:
            from huggingface_hub import hf_hub_download
            campplus_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
            self.campplus = CAMPPlusModel(feat_dim=80, embedding_size=192)
            state_dict = torch.load(campplus_path, map_location=self.device)
            self.campplus.load_state_dict(state_dict)
            print(f"  CAMPPlus weights restored from: {campplus_path}")
        except Exception as e:
            print(f"Warning: Failed to load CAMPPlus weights: {e}")
            self.campplus = CAMPPlusModel(feat_dim=80, embedding_size=192)
        self.campplus = self.campplus.to(self.device)
        self.campplus.eval()

    def _init_mlx_models(self):
        """Initialize MLX models."""
        import mlx.nn as nn
        from mlx_indextts.config import IndexTTSConfig
        from mlx_indextts.models.gpt_v2 import UnifiedVoiceV2
        from mlx_indextts.models.s2mel import S2Mel
        from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config

        # Build config from OmegaConf
        config = IndexTTSConfig.from_omegaconf(self.cfg)

        # Check if model was pre-quantized
        config_json_path = self.mlx_model_dir / "config.json"
        saved_quantize_bits = None
        if config_json_path.exists():
            import json
            with open(config_json_path) as f:
                config_dict = json.load(f)
                saved_quantize_bits = config_dict.get("quantize_bits")

        # Determine effective quantization
        effective_quantize = saved_quantize_bits or self.quantize_bits

        # GPT v2 (MLX)
        print("Loading GPT v2 (MLX)...")
        self.gpt = UnifiedVoiceV2(config)

        # If model was saved with quantization, quantize before loading weights
        if saved_quantize_bits:
            print(f"  Model pre-quantized to {saved_quantize_bits}-bit")
            nn.quantize(self.gpt.gpt, bits=saved_quantize_bits, group_size=64)

        if Path(self.gpt_weights_path).exists():
            self.gpt.load_weights(self.gpt_weights_path)
            print(f"GPT v2 (MLX) loaded from {self.gpt_weights_path}")
        else:
            print(f"Warning: GPT v2 weights not found at {self.gpt_weights_path}")

        # If runtime quantization requested (and model wasn't pre-quantized)
        if self.quantize_bits and not saved_quantize_bits:
            print(f"  Applying runtime {self.quantize_bits}-bit quantization to GPT...")
            nn.quantize(self.gpt.gpt, bits=self.quantize_bits, group_size=64)

        # S2Mel (MLX) - we only use CFM part for inference
        print("Loading S2Mel (MLX)...")
        self.s2mel_mlx = S2Mel()
        if Path(self.s2mel_weights_path).exists():
            self.s2mel_mlx.load_weights(self.s2mel_weights_path)
            print(f"S2Mel (MLX) loaded from {self.s2mel_weights_path}")
        else:
            print(f"Warning: S2Mel weights not found at {self.s2mel_weights_path}")
        self.s2mel_mlx.eval()  # Set to eval mode to disable dropout

        # BigVGAN v2 (MLX)
        print("Loading BigVGAN v2 (MLX)...")
        bigvgan_config = BigVGANV2Config()
        self.bigvgan_mlx = BigVGANV2(bigvgan_config)
        if Path(self.bigvgan_weights_path).exists():
            self.bigvgan_mlx.load_weights(self.bigvgan_weights_path)
            print(f"BigVGAN v2 (MLX) loaded from {self.bigvgan_weights_path}")
        else:
            print(f"Warning: BigVGAN weights not found at {self.bigvgan_weights_path}")

    def _init_tokenizer(self):
        """Initialize text tokenizer."""
        from mlx_indextts.tokenizer import TextTokenizer
        from mlx_indextts.normalize import TextNormalizer

        # Find bpe model
        bpe_path = None
        for base_dir in [self.mlx_model_dir, self.model_dir]:
            test_path = base_dir / self.cfg.dataset.bpe_model
            if not test_path.exists():
                test_path = base_dir / "tokenizer.model"
            if test_path.exists():
                bpe_path = test_path
                break

        if bpe_path is None:
            raise FileNotFoundError(f"BPE model not found: {self.cfg.dataset.bpe_model}")

        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(str(bpe_path), self.normalizer)
        print(f"Tokenizer loaded from {bpe_path}")

    def _init_mel_config(self):
        """Initialize mel spectrogram function using PyTorch (matching index-tts)."""
        from librosa.filters import mel as librosa_mel_fn

        n_fft = self.cfg.s2mel.preprocess_params.spect_params.n_fft
        win_length = self.cfg.s2mel.preprocess_params.spect_params.win_length
        hop_length = self.cfg.s2mel.preprocess_params.spect_params.hop_length
        n_mels = self.cfg.s2mel.preprocess_params.spect_params.n_mels
        sr = self.cfg.s2mel.preprocess_params.sr
        fmin = self.cfg.s2mel.preprocess_params.spect_params.get('fmin', 0)
        fmax = self.cfg.s2mel.preprocess_params.spect_params.get('fmax', None)
        if fmax == "None":
            fmax = None

        # Pre-compute mel basis (will be moved to device on first use)
        mel_basis_np = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self._mel_basis = torch.from_numpy(mel_basis_np).float()
        self._hann_window = torch.hann_window(win_length)
        self._mel_n_fft = n_fft
        self._mel_hop_size = hop_length
        self._mel_win_size = win_length

        def mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
            """Extract mel spectrogram from audio tensor (PyTorch implementation).

            This matches the index-tts implementation exactly, including:
            - Reflect padding before STFT
            - Hann window
            - Log compression with 1e-5 clipping
            """
            device = audio.device

            # Move mel basis and window to device if needed
            if self._mel_basis.device != device:
                self._mel_basis = self._mel_basis.to(device)
                self._hann_window = self._hann_window.to(device)

            # Ensure 2D input (batch, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # Reflect padding (matching index-tts exactly)
            pad_size = int((self._mel_n_fft - self._mel_hop_size) / 2)
            audio = torch.nn.functional.pad(
                audio.unsqueeze(1), (pad_size, pad_size), mode="reflect"
            )
            audio = audio.squeeze(1)

            # STFT
            spec = torch.stft(
                audio,
                self._mel_n_fft,
                hop_length=self._mel_hop_size,
                win_length=self._mel_win_size,
                window=self._hann_window,
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )

            # Magnitude
            spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

            # Mel filterbank
            spec = torch.matmul(self._mel_basis, spec)

            # Log compression
            spec = torch.log(torch.clamp(spec, min=1e-5))

            return spec

        self.mel_fn = mel_spectrogram

    @torch.no_grad()
    def _get_semantic_embedding(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract semantic embedding using W2V-BERT."""
        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _ensure_pytorch_modules(self):
        """Lazy load PyTorch preprocessing modules on first .wav use."""
        if self._preprocessing_initialized:
            return
        self._init_pytorch_modules()
        self._preprocessing_initialized = True

    def _load_speaker(self, npz_path: str) -> dict:
        """Load pre-computed speaker conditioning from .npz file.

        Raises:
            ValueError: If the file is not a v2.0 speaker file
        """
        data = np.load(npz_path)

        # Check version
        if 'version' in data:
            version = float(data['version'][0])
            if version < 2.0:
                raise ValueError(
                    f"Speaker file is v{version:.1f} format, but this is IndexTTS 2.0. "
                    f"Please use the correct model version."
                )
        elif 'conditioning' in data:
            # Old v1.5 format (no version field, has 'conditioning')
            raise ValueError(
                "Speaker file is v1.5 format, but this is IndexTTS 2.0. "
                "Please use the correct model version."
            )

        cache = {
            'audio_path': npz_path,
            'spk_cond_emb': torch.from_numpy(data['spk_cond_emb']).to(self.device),
            'S_ref': torch.from_numpy(data['S_ref']).to(self.device),
            'ref_mel': torch.from_numpy(data['ref_mel']).to(self.device),
            'style': torch.from_numpy(data['style']).to(self.device),
            'prompt_condition': torch.from_numpy(data['prompt_condition']).to(self.device),
        }
        return cache

    def save_speaker(self, audio_path: str, output_path: str) -> None:
        """Pre-compute and save speaker conditioning to .npz file.

        This saves all conditioning data needed for generation, allowing
        faster inference by skipping W2V-BERT, SemanticCodec, and CAMPPlus.

        Args:
            audio_path: Path to reference audio file (.wav)
            output_path: Output path for .npz file
        """
        # Ensure PyTorch modules are loaded
        self._ensure_pytorch_modules()

        # Process reference audio
        ref_data = self._process_reference_audio(audio_path)

        # Save to npz with version
        np.savez(
            output_path,
            version=np.array([2.0]),  # Version identifier
            spk_cond_emb=ref_data['spk_cond_emb'].cpu().numpy(),
            S_ref=ref_data['S_ref'].cpu().numpy(),
            ref_mel=ref_data['ref_mel'].cpu().numpy(),
            style=ref_data['style'].cpu().numpy(),
            prompt_condition=ref_data['prompt_condition'].cpu().numpy(),
        )

    @torch.no_grad()
    def _process_reference_audio(self, audio_path: str):
        """Process reference audio to get all conditioning.

        Supports both .wav files (requires PyTorch preprocessing) and
        .npz files (pre-computed, no PyTorch needed).
        """
        # Check cache
        if self.cache.get('audio_path') == audio_path:
            return self.cache

        # Load from .npz if pre-computed
        if audio_path.endswith('.npz'):
            self.cache = self._load_speaker(audio_path)
            return self.cache

        # Otherwise, need PyTorch modules for preprocessing
        self._ensure_pytorch_modules()

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        audio = torch.tensor(audio).unsqueeze(0)

        # Resample
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        # Semantic embedding (for GPT conditioning)
        spk_cond_emb = self._get_semantic_embedding(audio_16k)

        # Semantic codes (for S2Mel length regulator)
        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)

        # Reference mel (for S2Mel CFM prompt)
        ref_mel = self.mel_fn(audio_22k.to(self.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(self.device)

        # Style embedding (CAMPPlus)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus(feat.unsqueeze(0))

        # Prompt condition via length regulator (MLX)
        S_ref_mx = mx.array(S_ref.cpu().numpy())
        ref_target_lengths_mx = mx.array(ref_target_lengths.cpu().numpy())
        prompt_condition_mx, _, _, _, _ = self.s2mel_mlx.length_regulator(
            S_ref_mx, ylens=ref_target_lengths_mx, n_quantizers=3, f0=None
        )
        mx.eval(prompt_condition_mx)
        prompt_condition = torch.from_numpy(np.array(prompt_condition_mx)).to(self.device)

        # Cache
        self.cache = {
            'audio_path': audio_path,
            'spk_cond_emb': spk_cond_emb,
            'S_ref': S_ref,
            'ref_mel': ref_mel,
            'style': style,
            'prompt_condition': prompt_condition,
        }
        return self.cache

    def _compute_emotion_vector(
        self,
        emotion_weights: Dict[str, float],
        style: torch.Tensor,
        use_random: bool = False,
    ) -> torch.Tensor:
        """Compute emotion vector from emotion weights using emo_matrix.

        Args:
            emotion_weights: Dict mapping emotion names to weights (0.0-1.2)
            style: Style embedding from CAMPPlus (1, 192)
            use_random: If True, randomly select from each emotion category

        Returns:
            Emotion vector (1, 1280) for emovec_layer input
        """
        import torch.nn.functional as F

        if self.emo_matrix is None:
            raise ValueError("Emotion matrices not loaded, cannot use emotion control")

        # Convert emotion_weights to weight vector in category order
        weight_vector = torch.tensor(
            [emotion_weights.get(cat, 0.0) for cat in EMOTION_CATEGORIES],
            device=self.device, dtype=torch.float32
        )

        # For each emotion category, find most similar vector in spk_matrix (or random)
        if use_random:
            import random
            selected_indices = [random.randint(0, n - 1) for n in EMO_NUM]
        else:
            # Use style to find most similar speaker in each category
            selected_indices = []
            for spk_cat in self.spk_matrix_split:
                # Cosine similarity between style and each speaker in category
                similarities = F.cosine_similarity(style.float(), spk_cat.float(), dim=1)
                selected_indices.append(torch.argmax(similarities).item())

        # Gather emotion vectors and compute weighted sum
        emo_vectors = torch.stack([
            self.emo_matrix_split[i][idx]
            for i, idx in enumerate(selected_indices)
        ])  # (8, 1280)

        # Weighted sum: (8,) @ (8, 1280) -> (1280,)
        emovec_mat = (weight_vector.unsqueeze(1) * emo_vectors).sum(dim=0)
        emovec_mat = emovec_mat.unsqueeze(0)  # (1, 1280)

        return emovec_mat

    def generate(
        self,
        text: str,
        reference_audio: str,
        output_path: Optional[str] = None,
        max_mel_tokens: int = 1500,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        diffusion_steps: int = 25,
        cfg_rate: float = 0.7,
        emotion: Optional[Union[str, Dict[str, float]]] = None,
        emo_alpha: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Generate speech from text.

        Args:
            text: Input text to synthesize
            reference_audio: Path to reference audio file
            output_path: Optional path to save output audio
            max_mel_tokens: Maximum mel tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            diffusion_steps: Number of diffusion steps for S2Mel
            cfg_rate: Classifier-free guidance rate
            emotion: Emotion specification. Can be:
                - None: extract from reference audio (default)
                - str: "happy", "happy:0.8,sad:0.2", or JSON
                - dict: {"happy": 0.8, "sad": 0.2}
            emo_alpha: Emotion intensity (0.0=reference audio, 1.0=full specified emotion)
            seed: Random seed for reproducible generation
            verbose: Whether to print progress

        Returns:
            Generated audio waveform as numpy array
        """
        # Set random seed if specified
        if seed is not None:
            mx.random.seed(seed)
            torch.manual_seed(seed)
            if verbose:
                print(f"Using seed: {seed}")

        start_time = time.perf_counter()

        # 1. Process reference audio (PyTorch preprocessing)
        ref_data = self._process_reference_audio(reference_audio)
        spk_cond_emb_pt = ref_data['spk_cond_emb']  # PyTorch tensor
        style_pt = ref_data['style']
        prompt_condition_pt = ref_data['prompt_condition']
        ref_mel_pt = ref_data['ref_mel']

        # Convert to MLX for GPT
        spk_cond_emb = mx.array(spk_cond_emb_pt.cpu().numpy())  # (1, T, 1024)
        # GPT expects NCL format: (batch, 1024, time)
        spk_cond_emb_ncl = spk_cond_emb.transpose(0, 2, 1)

        # 2. Tokenize text
        text_tokens = self.tokenizer.encode(text)
        text_tokens = mx.array([text_tokens], dtype=mx.int32)
        if verbose:
            print(f"Text tokens: {text_tokens.shape[1]}")

        # 3. GPT conditioning (MLX)
        # Speaker conditioning
        cond_lengths = mx.array([spk_cond_emb.shape[1]])
        speech_cond = self.gpt.get_conditioning(spk_cond_emb_ncl, cond_lengths)

        # Emotion conditioning
        # Base emotion vector from reference audio
        base_emo_vec = self.gpt.get_emovec(spk_cond_emb_ncl, cond_lengths)

        if emotion is not None:
            # Parse emotion specification
            if isinstance(emotion, str):
                emotion_weights = parse_emotion(emotion)
            else:
                emotion_weights = emotion

            if verbose:
                print(f"Using specified emotion: {emotion_weights}")

            # Compute target emotion vector from emo_matrix
            target_emo_vec_pt = self._compute_emotion_vector(emotion_weights, style_pt)
            # Pass through emovec_layer and emo_layer (need to do in MLX)
            # Actually the emo_matrix is already in the emovec_layer output space (1280)
            # So we need to apply emo_layer
            target_emo_vec = mx.array(target_emo_vec_pt.cpu().numpy())
            target_emo_vec = self.gpt.emo_layer(target_emo_vec)

            # Blend with alpha: out = base + alpha * (target - base)
            emo_vec = base_emo_vec + emo_alpha * (target_emo_vec - base_emo_vec)
        else:
            emo_vec = base_emo_vec

        # Prepare full conditioning (speaker + emotion + speed)
        conditioning = self.gpt.prepare_conditioning_latents(speech_cond, emo_vec, batch_size=1)

        # 4. GPT autoregressive generation (MLX)
        gpt_start = time.perf_counter()

        # Prepare inputs
        input_emb, _ = self.gpt.prepare_inputs(conditioning, text_tokens)

        # Add start mel token
        mel_start = mx.array([[self.gpt.start_mel_token]], dtype=mx.int32)
        mel_start_emb = self.gpt.mel_embedding(mel_start)
        mel_start_emb = mel_start_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(0)
        input_emb = mx.concatenate([input_emb, mel_start_emb], axis=1)

        # Autoregressive loop
        mel_codes = []
        cache = None

        for i in range(max_mel_tokens):
            if cache is None:
                next_token, _, cache = self.gpt.generate_step(
                    input_emb, cache, temperature, top_k, top_p
                )
            else:
                last_token = mx.array([[mel_codes[-1]]], dtype=mx.int32)
                last_emb = self.gpt.mel_embedding(last_token)
                # Position calculation matches PyTorch: attention_mask.shape[1] - mel_len
                # In PyTorch: mel_len = inputs_embeds.shape[1] (without start_mel_token)
                #             attention_mask starts at mel_len + 1, grows by 1 each step
                # So position for first generated token = (mel_len + 2) - mel_len = 2
                # Position = len(mel_codes) + 1 (since mel_codes[0] is at position 2)
                mel_pos = len(mel_codes) + 1
                last_emb = last_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(mel_pos)
                next_token, _, cache = self.gpt.generate_step(
                    last_emb, cache, temperature, top_k, top_p
                )

            token_id = next_token[0].item()

            if token_id == self.gpt.stop_mel_token:
                break

            mel_codes.append(token_id)
            mx.eval(cache)

            if verbose and (i + 1) % 50 == 0:
                print(f"Generated {i + 1} mel tokens...")

        gpt_gen_time = time.perf_counter() - gpt_start
        if verbose:
            print(f"Generated {len(mel_codes)} mel tokens")

        if len(mel_codes) == 0:
            raise RuntimeError("No mel tokens generated")

        # 5. GPT forward to get latent (MLX)
        s2mel_start = time.perf_counter()

        mel_codes_tensor = mx.array([mel_codes], dtype=mx.int32)
        latent = self.gpt.forward_latent(conditioning, text_tokens, mel_codes_tensor)

        # 6. S2Mel processing

        # gpt_layer projection (MLX)
        latent = self.s2mel_mlx.gpt_layer(latent)

        # We need semantic codes for vq2emb - use PyTorch semantic_codec
        # Convert mel_codes to PyTorch for semantic processing
        codes_pt = torch.tensor([mel_codes], device=self.device)
        S_infer_pt = self.semantic_codec.quantizer.vq2emb(codes_pt.unsqueeze(1))
        S_infer_pt = S_infer_pt.transpose(1, 2)  # (1, T, 1024)

        # Convert to MLX and add latent
        S_infer = mx.array(S_infer_pt.detach().cpu().numpy())
        S_infer = S_infer + latent

        # Length regulator (MLX)
        code_len = len(mel_codes)
        target_lengths = mx.array([int(code_len * 1.72)])
        cond, _, _, _, _ = self.s2mel_mlx.length_regulator(S_infer, target_lengths, n_quantizers=3)

        # Concatenate with prompt condition
        prompt_condition = mx.array(prompt_condition_pt.cpu().numpy())
        cat_condition = mx.concatenate([prompt_condition, cond], axis=1)

        # 7. CFM inference (MLX)
        ref_mel = mx.array(ref_mel_pt.cpu().numpy())
        style = mx.array(style_pt.cpu().numpy())
        x_lens = mx.array([cat_condition.shape[1]])

        mel_out = self.s2mel_mlx.cfm.inference(
            mu=cat_condition,
            x_lens=x_lens,
            prompt=ref_mel,
            style=style,
            f0=None,
            n_timesteps=diffusion_steps,
            temperature=1.0,
            inference_cfg_rate=cfg_rate,
        )
        mx.eval(mel_out)

        s2mel_time = time.perf_counter() - s2mel_start

        # Trim prompt region
        prompt_len = ref_mel.shape[-1]
        mel_out = mel_out[:, :, prompt_len:]

        # 8. BigVGAN vocoder (MLX)
        vocoder_start = time.perf_counter()

        audio_out = self.bigvgan_mlx(mel_out)
        mx.eval(audio_out)

        vocoder_time = time.perf_counter() - vocoder_start

        # Convert to numpy
        audio = np.array(audio_out[0, 0])

        total_time = time.perf_counter() - start_time
        audio_duration = len(audio) / 22050

        if verbose:
            rtf = total_time / audio_duration
            print(f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s (RTF: {rtf:.3f})")
            print(f"  GPT gen: {gpt_gen_time:.2f}s")
            print(f"  S2Mel: {s2mel_time:.2f}s")
            print(f"  BigVGAN: {vocoder_time:.2f}s")

        # Save if output path provided
        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, 22050)

        return audio


