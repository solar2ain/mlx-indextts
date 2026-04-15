"""Tokenizer for IndexTTS using SentencePiece BPE."""

import warnings
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

from mlx_indextts.normalize import TextNormalizer, tokenize_by_cjk_char


# Punctuation marks that indicate sentence boundaries
PUNCTUATION_MARKS_TOKENS = [
    ".",
    "!",
    "?",
    "▁.",
    # "▁!", # unk in some tokenizers
    "▁?",
    "...",  # ellipsis (without space prefix)
    "▁...",  # ellipsis (with space prefix)
]


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
        max_tokens_per_segment: int = 120,
    ) -> List[List[str]]:
        """Split token list into segments at natural boundaries.

        This handles long texts by splitting them at sentence boundaries
        (punctuation marks), with fallback to comma/hyphen splits.
        Matches the behavior of the original PyTorch IndexTTS implementation.

        Args:
            tokens: List of tokens from tokenize()
            max_tokens_per_segment: Maximum tokens per segment (default: 120)

        Returns:
            List of token segments
        """
        return self._split_segments_by_token(
            tokens,
            split_tokens=PUNCTUATION_MARKS_TOKENS,
            max_tokens_per_segment=max_tokens_per_segment,
        )

    @staticmethod
    def _split_segments_by_token(
        tokenized_str: List[str],
        split_tokens: List[str],
        max_tokens_per_segment: int,
    ) -> List[List[str]]:
        """Split tokenized text by specific tokens.

        Implements recursive splitting logic:
        1. First tries to split at sentence-ending punctuation
        2. Falls back to comma splits if no punctuation found
        3. Falls back to hyphen splits if no comma found
        4. Force splits at max_tokens_per_segment if necessary

        Args:
            tokenized_str: List of tokens
            split_tokens: Tokens to split on (e.g., [".", "!", "?"])
            max_tokens_per_segment: Maximum tokens per segment

        Returns:
            List of token segments
        """
        if len(tokenized_str) == 0:
            return []

        segments: List[List[str]] = []
        current_segment = []
        current_segment_len = 0

        i = 0
        while i < len(tokenized_str):
            token = tokenized_str[i]
            current_segment.append(token)
            current_segment_len += 1

            # Check for recursive splitting opportunities
            if not ("," in split_tokens or "▁," in split_tokens) and \
               ("," in current_segment or "▁," in current_segment):
                # No comma in split_tokens, but current segment has comma -> recurse
                sub_segments = TextTokenizer._split_segments_by_token(
                    current_segment, [",", "▁,"], max_tokens_per_segment
                )
                segments.extend(sub_segments)
                current_segment = []
                current_segment_len = 0
            elif "-" not in split_tokens and "-" in current_segment:
                # No hyphen in split_tokens, but current segment has hyphen -> recurse
                sub_segments = TextTokenizer._split_segments_by_token(
                    current_segment, ["-"], max_tokens_per_segment
                )
                segments.extend(sub_segments)
                current_segment = []
                current_segment_len = 0
            elif current_segment_len <= max_tokens_per_segment:
                # Normal case: check if we hit a split token
                if token in split_tokens and current_segment_len > 2:
                    # Check if next token is a quote (don't split before quote)
                    if i < len(tokenized_str) - 1:
                        next_token = tokenized_str[i + 1]
                        if next_token in ["'", "▁'"]:
                            current_segment.append(next_token)
                            i += 1
                            current_segment_len += 1
                    segments.append(current_segment)
                    current_segment = []
                    current_segment_len = 0
            else:
                # Exceeded max length -> force split
                sub_segments = []
                for j in range(0, len(current_segment), max_tokens_per_segment):
                    end_idx = min(j + max_tokens_per_segment, len(current_segment))
                    sub_segments.append(current_segment[j:end_idx])
                warnings.warn(
                    f"Segment length ({len(current_segment)}) exceeds limit ({max_tokens_per_segment}). "
                    f"Force splitting may cause unexpected behavior.",
                    RuntimeWarning,
                )
                segments.extend(sub_segments)
                current_segment = []
                current_segment_len = 0

            i += 1

        # Add remaining tokens
        if current_segment_len > 0:
            segments.append(current_segment)

        # Merge short adjacent segments
        merged_segments = []
        for segment in segments:
            if len(segment) == 0:
                continue
            if len(merged_segments) == 0:
                merged_segments.append(segment)
            elif len(merged_segments[-1]) + len(segment) <= max_tokens_per_segment // 2:
                # Merge if combined length is less than half max
                merged_segments[-1] = merged_segments[-1] + segment
            elif len(merged_segments[-1]) + len(segment) <= max_tokens_per_segment:
                # Merge if combined length fits within max
                merged_segments[-1] = merged_segments[-1] + segment
            else:
                merged_segments.append(segment)

        return merged_segments
