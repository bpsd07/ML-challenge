"""
Stage 1: Advanced Normalization Pipeline
Maintains both raw and multiple specialized representations of names and addresses.
Supports country-aware variations for India, US, and France (open set).
Extracts structured components (postal code, house number, numeric tokens, legal suffixes).
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional

# Extended legal suffixes across US, India, and France
LEGAL_SUFFIXES_LIST = [
    # General / US / UK
    "LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "LTD", "LIMITED",
    "CO", "COMPANY", "LLP", "PLC", "HOLDINGS", "GROUP", "ENTERPRISES", "INDUSTRIES",
    # India
    "PVT", "PRIVATE", "PVT LTD", "PRIVATE LIMITED", "LTD CO",
    # France / Continental Europe
    "SARL", "SAS", "SA", "EURL", "SNC", "SCI", "SCOP", "GIE", "GMBH"
]

# Regex for stripping legal suffixes safely
_SUFFIX_PATTERN = r'\b(' + '|'.join(sorted(LEGAL_SUFFIXES_LIST, key=len, reverse=True)) + r')\b'
LEGAL_RE = re.compile(_SUFFIX_PATTERN, re.IGNORECASE)

# Standard abbreviation mappings (US, India, UK, general)
COMMON_ABBREVIATIONS = {
    "CORP": "CORPORATION", "INC": "INCORPORATED", "PVT": "PRIVATE", "LTD": "LIMITED",
    "CO": "COMPANY", "ST": "STREET", "RD": "ROAD", "AVE": "AVENUE", "BLVD": "BOULEVARD",
    "DR": "DRIVE", "LN": "LANE", "HWY": "HIGHWAY", "SQ": "SQUARE", "PL": "PLACE",
    "CT": "COURT", "STE": "SUITE", "APT": "APARTMENT", "FL": "FLOOR", "CTR": "CENTER",
    "INTL": "INTERNATIONAL", "NATL": "NATIONAL", "DEPT": "DEPARTMENT", "ASSN": "ASSOCIATION",
    "SVCS": "SERVICES", "TRDG": "TRADING", "ENT": "ENTERPRISES", "IND": "INDUSTRIES",
    # French road types
    "RUE": "RUE", "BD": "BOULEVARD", "AV": "AVENUE", "IMP": "IMPASSE", "ALL": "ALLEE"
}

# Indian transliteration normalization rules (phonetic vowel/consonant unification)
INDIAN_PHONETIC_PAIRS = [
    (r'ee', 'i'),
    (r'oo', 'u'),
    (r'w', 'v'),
    (r'sh', 's'),
    (r'ch', 'c'),
]


def strip_accents(text: str) -> str:
    """Remove diacritics/accents safely using NFKD decomposition."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def extract_char_ngrams(text: str, n: int) -> Set[str]:
    """Extract character n-grams from a normalized string."""
    clean = re.sub(r'\s+', ' ', text).strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def normalize_business_name(raw_name: Optional[str], country: str = "US") -> Dict:
    """
    Generate multiple representations and extracted features for a business name.
    Preserves raw name while providing normalized views for blocking and matching.
    """
    raw = str(raw_name or "").strip()
    if not raw:
        return {
            "name_raw": "",
            "name_nfkc": "",
            "name_lower": "",
            "name_ascii_when_safe": "",
            "name_alnum": "",
            "name_tokens": [],
            "name_tokens_sorted": "",
            "name_token_set": set(),
            "name_no_legal_suffix": "",
            "name_no_legal_suffix_sorted": "",
            "name_initials": "",
            "name_acronym": "",
            "legal_suffixes": [],
            "name_char_3gram": set(),
            "name_char_4gram": set(),
            "name_char_5gram": set(),
        }

    # Unicode NFKC
    nfkc = unicodedata.normalize('NFKC', raw)
    # Lowercase
    lower = nfkc.lower()
    # Ascii-clean
    ascii_safe = strip_accents(lower)

    # Country-specific adjustments
    norm_text = ascii_safe
    if country == "India":
        # Indian phonetic unification for matching
        pass

    # Extract detected legal suffixes
    legal_matches = [m.upper() for m in LEGAL_RE.findall(ascii_safe)]

    # Alphanumeric clean
    alnum = re.sub(r'[^a-z0-9\s]', ' ', ascii_safe)
    alnum = re.sub(r'\s+', ' ', alnum).strip()

    # Tokens
    tokens = [t for t in alnum.split() if t]
    tokens_sorted = " ".join(sorted(tokens))
    token_set = set(tokens)

    # Tokens without legal suffix
    no_suffix_tokens = [t for t in tokens if t.upper() not in LEGAL_SUFFIXES_LIST]
    no_suffix = " ".join(no_suffix_tokens)
    no_suffix_sorted = " ".join(sorted(no_suffix_tokens))

    # Initials / Acronym
    initials = "".join([t[0] for t in tokens]) if tokens else ""
    acronym = initials.upper()

    # Character n-grams
    c3 = extract_char_ngrams(alnum, 3)
    c4 = extract_char_ngrams(alnum, 4)
    c5 = extract_char_ngrams(alnum, 5)

    return {
        "name_raw": raw,
        "name_nfkc": nfkc,
        "name_lower": lower,
        "name_ascii_when_safe": ascii_safe,
        "name_alnum": alnum,
        "name_tokens": tokens,
        "name_tokens_sorted": tokens_sorted,
        "name_token_set": token_set,
        "name_no_legal_suffix": no_suffix,
        "name_no_legal_suffix_sorted": no_suffix_sorted,
        "name_initials": initials,
        "name_acronym": acronym,
        "legal_suffixes": legal_matches,
        "name_char_3gram": c3,
        "name_char_4gram": c4,
        "name_char_5gram": c5,
    }


def extract_postal_code(text: str, country: str = "") -> Optional[str]:
    """
    Extract postal code in a country-aware manner:
    - India: 6-digit PIN code (e.g. 110001, 400001)
    - US: 5-digit ZIP code (e.g. 90210, 02138)
    - France: 5-digit code postal (e.g. 75001, 69002)
    """
    if country == "India":
        m = re.findall(r'\b[1-9][0-9]{5}\b', text)
        if m:
            return m[-1]  # PIN code usually appears near the end of address
    elif country in ["US", "France"]:
        m = re.findall(r'\b[0-9]{5}\b', text)
        if m:
            return m[-1]
    else:
        # Generic postal code heuristic (5 or 6 digits)
        m = re.findall(r'\b[0-9]{5,6}\b', text)
        if m:
            return m[-1]
    return None


def extract_house_number(tokens: List[str]) -> Optional[str]:
    """Extract leading or prominent street/house number."""
    if not tokens:
        return None
    # Check first token
    if re.match(r'^\d+[a-z]?$', tokens[0]):
        return tokens[0]
    # Check for #123 or No.123
    for t in tokens[:4]:
        if re.match(r'^\d+$', t):
            return t
    return None


def normalize_business_address(raw_address: Optional[str], country: str = "US") -> Dict:
    """
    Generate multiple representations and structured components for an address.
    """
    raw = str(raw_address or "").strip()
    if not raw:
        return {
            "address_raw": "",
            "address_nfkc": "",
            "address_lower": "",
            "address_alnum": "",
            "address_tokens": [],
            "address_tokens_sorted": "",
            "postal_code": None,
            "house_number": None,
            "numeric_tokens": set(),
            "address_char_3gram": set(),
            "address_char_4gram": set(),
            "address_char_5gram": set(),
        }

    nfkc = unicodedata.normalize('NFKC', raw)
    lower = nfkc.lower()
    ascii_safe = strip_accents(lower)

    # Expand standard abbreviations
    words = re.findall(r'\b\w+\b', ascii_safe.upper())
    expanded_words = [COMMON_ABBREVIATIONS.get(w, w) for w in words]
    expanded_text = " ".join(expanded_words).lower()

    # Alphanumeric clean
    alnum = re.sub(r'[^a-z0-9\s]', ' ', expanded_text)
    alnum = re.sub(r'\s+', ' ', alnum).strip()

    tokens = [t for t in alnum.split() if t]
    tokens_sorted = " ".join(sorted(tokens))

    # Structured components
    postal = extract_postal_code(raw, country=country)
    digits = set(re.findall(r'\b\d+\b', raw))
    house_num = extract_house_number(tokens)

    # Character ngrams
    c3 = extract_char_ngrams(alnum, 3)
    c4 = extract_char_ngrams(alnum, 4)
    c5 = extract_char_ngrams(alnum, 5)

    return {
        "address_raw": raw,
        "address_nfkc": nfkc,
        "address_lower": lower,
        "address_alnum": alnum,
        "address_tokens": tokens,
        "address_tokens_sorted": tokens_sorted,
        "postal_code": postal,
        "house_number": house_num,
        "numeric_tokens": digits,
        "address_char_3gram": c3,
        "address_char_4gram": c4,
        "address_char_5gram": c5,
    }
