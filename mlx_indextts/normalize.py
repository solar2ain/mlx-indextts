"""Text normalization for IndexTTS."""

import re
import unicodedata
from typing import List, Optional, Tuple


# CJK Unicode ranges
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0x2CEB0, 0x2EBEF),  # CJK Unified Ideographs Extension F
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
]


def is_cjk_char(char: str) -> bool:
    """Check if a character is a CJK character."""
    code = ord(char)
    for start, end in CJK_RANGES:
        if start <= code <= end:
            return True
    return False


def tokenize_by_cjk_char(text: str, do_upper_case: bool = True) -> str:
    """Add spaces around CJK characters for better tokenization.

    This helps BPE tokenizers handle mixed CJK/Latin text better.
    Note: By default converts text to uppercase to match PyTorch behavior.

    Args:
        text: Input text
        do_upper_case: Whether to convert to uppercase (default True for IndexTTS compatibility)

    Returns:
        Tokenized text with spaces around CJK characters
    """
    output = []
    for char in text:
        if is_cjk_char(char):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)

    # Clean up multiple spaces
    result = "".join(output)
    result = re.sub(r"\s+", " ", result)
    result = result.strip()

    # Convert to uppercase (matches PyTorch IndexTTS behavior)
    if do_upper_case:
        result = result.upper()

    return result


def de_tokenize_by_cjk_char(text: str, do_lower_case: bool = False) -> str:
    """Remove spaces around CJK characters (reverse of tokenize_by_cjk_char)."""
    if do_lower_case:
        text = text.lower()
    # Remove spaces between CJK characters
    result = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return result


def normalize_text(text: str) -> str:
    """Basic text normalization.

    - Unicode normalization (NFKC)
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace various whitespace with regular space
    text = re.sub(r"[\t\n\r\f\v]+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)

    # Strip
    text = text.strip()

    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Handles both English and Chinese punctuation.
    """
    # Sentence-ending punctuation
    sentence_endings = r"[.!?。！？；;]"

    # Split on sentence endings, keeping the delimiter
    parts = re.split(f"({sentence_endings})", text)

    sentences = []
    current = ""

    for part in parts:
        current += part
        if re.match(sentence_endings, part):
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ""

    # Don't forget the last part if it doesn't end with punctuation
    if current.strip():
        sentences.append(current.strip())

    return sentences


class TextNormalizer:
    """Text normalizer for IndexTTS.

    This handles text normalization including:
    - Punctuation mapping (Chinese -> ASCII)
    - English contraction expansion
    - Technical term protection
    - Pinyin tone protection
    - Number/date normalization (via WeTextProcessing if available)
    """

    # Punctuation mapping (Chinese/special -> ASCII)
    CHAR_REP_MAP = {
        "：": ",",
        "；": ",",
        ";": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": " ",
        "·": "-",
        "、": ",",
        "...": "…",
        ",,,": "…",
        "，，，": "…",
        "……": "…",
        """: "'",
        """: "'",
        '"': "'",
        "'": "'",
        "'": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
        ":": ",",
    }

    ZH_CHAR_REP_MAP = {
        "$": ".",
        **CHAR_REP_MAP,
    }

    # Pinyin tone pattern: pinyin + tone number (1-5)
    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"

    # Name pattern: Chinese name with separators
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"

    # Tech term pattern: GPT-5-nano, F5-TTS, etc.
    TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"

    # English contraction pattern
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"

    def __init__(self, enable_glossary: bool = False):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.loaded = False
        self.enable_glossary = enable_glossary
        self.term_glossary = {}

    def load(self):
        """Load normalizer resources."""
        if self.loaded:
            return

        # Try to load WeTextProcessing for number/date normalization
        try:
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
            print(">> Loaded WeTextProcessing normalizers")
        except ImportError:
            # Try tn package (Linux)
            try:
                from tn.chinese.normalizer import Normalizer as NormalizerZh
                from tn.english.normalizer import Normalizer as NormalizerEn
                self.zh_normalizer = NormalizerZh(remove_interjections=False, remove_erhua=False)
                self.en_normalizer = NormalizerEn()
                print(">> Loaded TN normalizers")
            except ImportError:
                print(">> Warning: WeTextProcessing/TN not available, using basic normalization")
                self.zh_normalizer = None
                self.en_normalizer = None

        self.loaded = True

    def _use_chinese(self, text: str) -> bool:
        """Determine if text should use Chinese normalizer."""
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        has_alpha = bool(re.search(r"[a-zA-Z]", text))

        if has_chinese or not has_alpha:
            return True

        # Check for pinyin with tones
        has_pinyin = bool(re.search(self.PINYIN_TONE_PATTERN, text, re.IGNORECASE))
        return has_pinyin

    def _save_tech_terms(self, text: str) -> Tuple[str, Optional[List[str]]]:
        """Protect technical terms by replacing hyphens with placeholders."""
        tech_pattern = re.compile(self.TECH_TERM_PATTERN)
        tech_list = tech_pattern.findall(text)

        if not tech_list:
            return text, None

        # Sort by length (longest first) to avoid partial replacements
        tech_list = sorted(set(tech_list), key=len, reverse=True)

        for term in tech_list:
            protected = term.replace("-", "<H>")
            text = text.replace(term, protected)

        return text, tech_list

    def _restore_tech_terms(self, text: str, tech_list: Optional[List[str]]) -> str:
        """Restore technical terms from placeholders."""
        if not tech_list:
            return text
        return re.sub(r'\s*<H>\s*', '-', text)

    def _save_pinyin_tones(self, text: str) -> Tuple[str, Optional[List[str]]]:
        """Protect pinyin tones by replacing with placeholders."""
        pattern = re.compile(self.PINYIN_TONE_PATTERN, re.IGNORECASE)
        matches = pattern.findall(text)

        if not matches:
            return text, None

        pinyin_list = list(set("".join(m) for m in matches))

        for i, pinyin in enumerate(pinyin_list):
            placeholder = f"<pinyin_{chr(ord('a') + i)}>"
            text = text.replace(pinyin, placeholder)

        return text, pinyin_list

    def _restore_pinyin_tones(self, text: str, pinyin_list: Optional[List[str]]) -> str:
        """Restore pinyin tones from placeholders."""
        if not pinyin_list:
            return text

        for i, pinyin in enumerate(pinyin_list):
            placeholder = f"<pinyin_{chr(ord('a') + i)}>"
            # Correct jqx pinyin (u/ü -> v)
            if pinyin[0].lower() in "jqx":
                pinyin = re.sub(r"([jqx])[uü](n|e|an)*(\d)", r"\g<1>v\g<2>\g<3>", pinyin, flags=re.IGNORECASE)
            text = text.replace(placeholder, pinyin.upper())

        return text

    def _save_names(self, text: str) -> Tuple[str, Optional[List[str]]]:
        """Protect Chinese names with separators."""
        pattern = re.compile(self.NAME_PATTERN)
        names = pattern.findall(text)

        if not names:
            return text, None

        names = list(set(names))

        for i, name in enumerate(names):
            placeholder = f"<n_{chr(ord('a') + i)}>"
            text = text.replace(name, placeholder)

        return text, names

    def _restore_names(self, text: str, names: Optional[List[str]]) -> str:
        """Restore names from placeholders."""
        if not names:
            return text

        for i, name in enumerate(names):
            placeholder = f"<n_{chr(ord('a') + i)}>"
            text = text.replace(placeholder, name)

        return text

    def normalize(self, text: str) -> str:
        """Normalize text for TTS.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return ""

        # Expand English contractions
        text = re.sub(self.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)

        if self._use_chinese(text):
            # Chinese text processing
            # Protect special patterns
            text, tech_list = self._save_tech_terms(text.rstrip())
            text, pinyin_list = self._save_pinyin_tones(text)
            text, names = self._save_names(text)

            # Apply normalizer if available
            if self.zh_normalizer:
                try:
                    text = self.zh_normalizer.normalize(text)
                except Exception:
                    pass

            # Restore protected patterns
            text = self._restore_names(text, names)
            text = self._restore_pinyin_tones(text, pinyin_list)
            text = self._restore_tech_terms(text, tech_list)

            # Apply character replacements
            pattern = re.compile("|".join(re.escape(p) for p in self.ZH_CHAR_REP_MAP.keys()))
            text = pattern.sub(lambda x: self.ZH_CHAR_REP_MAP[x.group()], text)
        else:
            # English text processing
            text, tech_list = self._save_tech_terms(text)

            # Apply normalizer if available
            if self.en_normalizer:
                try:
                    text = self.en_normalizer.normalize(text)
                except Exception:
                    pass

            # Restore protected patterns
            text = self._restore_tech_terms(text, tech_list)

            # Apply character replacements
            pattern = re.compile("|".join(re.escape(p) for p in self.CHAR_REP_MAP.keys()))
            text = pattern.sub(lambda x: self.CHAR_REP_MAP[x.group()], text)

        return text

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def load_glossary(self, glossary_dict: dict):
        """Load external glossary dictionary."""
        if glossary_dict and isinstance(glossary_dict, dict):
            self.term_glossary.update(glossary_dict)

    def load_glossary_from_yaml(self, glossary_path: str) -> bool:
        """Load glossary from YAML file."""
        import os
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary = yaml.safe_load(f)
                if glossary and isinstance(glossary, dict):
                    self.term_glossary = glossary
                    return True
        return False
