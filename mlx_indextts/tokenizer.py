"""Tokenizer for IndexTTS using SentencePiece BPE."""

from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

from mlx_indextts.normalize import TextNormalizer, tokenize_by_cjk_char


class TextTokenizer:
    """Text tokenizer for IndexTTS using SentencePiece BPE.

    Handles text tokenization with support for CJK characters.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        normalizer: Optional[TextNormalizer] = None,
    ):
        """Initialize tokenizer.

        Args:
            model_path: Path to SentencePiece model file
            normalizer: Optional text normalizer
        """
        self.model_path = Path(model_path)
        self.normalizer = normalizer or TextNormalizer()

        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))

        # Vocabulary info
        self.vocab_size = self.sp.GetPieceSize()

    def normalize(self, text: str) -> str:
        """Normalize text before tokenization."""
        return self.normalizer.normalize(text)

    def tokenize(self, text: str, normalize: bool = True) -> List[str]:
        """Tokenize text into subword tokens.

        Args:
            text: Input text
            normalize: Whether to normalize text first

        Returns:
            List of token strings
        """
        if normalize:
            text = self.normalize(text)

        # Add spaces around CJK characters
        text = tokenize_by_cjk_char(text)

        # Tokenize with SentencePiece
        tokens = self.sp.EncodeAsPieces(text)

        return tokens

    def encode(self, text: str, normalize: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            normalize: Whether to normalize text first

        Returns:
            List of token IDs
        """
        if normalize:
            text = self.normalize(text)

        # Add spaces around CJK characters
        text = tokenize_by_cjk_char(text)

        # Encode with SentencePiece
        ids = self.sp.EncodeAsIds(text)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        return self.sp.DecodeIds(ids)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to IDs.

        Args:
            tokens: List of token strings

        Returns:
            List of token IDs
        """
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to strings.

        Args:
            ids: List of token IDs

        Returns:
            List of token strings
        """
        return [self.sp.IdToPiece(id) for id in ids]

    def split_segments(
        self,
        tokens: List[str],
        max_tokens_per_segment: int = 100,
    ) -> List[List[str]]:
        """Split token list into segments.

        This helps handle long texts by splitting them into
        manageable chunks for the model.

        Args:
            tokens: List of tokens
            max_tokens_per_segment: Maximum tokens per segment

        Returns:
            List of token segments
        """
        if len(tokens) <= max_tokens_per_segment:
            return [tokens]

        segments = []
        current_segment = []

        for token in tokens:
            current_segment.append(token)

            # Check if we should split
            if len(current_segment) >= max_tokens_per_segment:
                # Try to split at a natural boundary
                # Look for punctuation or space tokens
                split_idx = len(current_segment)
                for i in range(len(current_segment) - 1, max(0, len(current_segment) - 20), -1):
                    t = current_segment[i]
                    # Check for sentence-ending punctuation
                    if any(p in t for p in ["。", ".", "!", "?", "！", "？", "；", ";"]):
                        split_idx = i + 1
                        break
                    # Check for comma or other natural breaks
                    elif any(p in t for p in ["，", ",", "、"]):
                        split_idx = i + 1
                        break

                # Split segment
                segments.append(current_segment[:split_idx])
                current_segment = current_segment[split_idx:]

        # Add remaining tokens
        if current_segment:
            segments.append(current_segment)

        return segments
