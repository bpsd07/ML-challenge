"""
Stage 1: Advanced Normalization Pipeline (Enhanced for High-Recall Blocking)
Maintains both raw and multiple specialized representations of names and addresses.
Supports country-aware variations for India, US, and France (open set).
Extracts structured components (postal code, house number, numeric tokens, legal suffixes, phonetic tokens, rare tokens).
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

# Standard abbreviation mappings (US, India, UK, France, general)
COMMON_ABBREVIATIONS = {
    # Corporate
    "CORP": "CORPORATION", "INC": "INCORPORATED", "PVT": "PRIVATE", "LTD": "LIMITED",
    "CO": "COMPANY", "BROS": "BROTHERS", "ASSN": "ASSOCIATION", "MFG": "MANUFACTURING",
    "DIST": "DISTRIBUTION", "INTL": "INTERNATIONAL", "NATL": "NATIONAL", "TECH": "TECHNOLOGY",
    "INFO": "INFORMATION", "SYS": "SYSTEMS", "COMM": "COMMUNICATIONS", "MED": "MEDICAL",
    "PHARM": "PHARMACEUTICAL", "DEV": "DEVELOPMENT", "MKTG": "MARKETING", "CONS": "CONSULTING",
    "ENG": "ENGINEERING", "SVCS": "SERVICES", "TRDG": "TRADING", "ENT": "ENTERPRISES",
    "IND": "INDUSTRIES", "DEPT": "DEPARTMENT",
    # Address / US & UK
    "ST": "STREET", "RD": "ROAD", "AVE": "AVENUE", "BLVD": "BOULEVARD",
    "DR": "DRIVE", "LN": "LANE", "HWY": "HIGHWAY", "SQ": "SQUARE", "PL": "PLACE",
    "CT": "COURT", "STE": "SUITE", "APT": "APARTMENT", "FL": "FLOOR", "CTR": "CENTER",
    "PKWY": "PARKWAY", "CIR": "CIRCLE", "WAY": "WAY", "TER": "TERRACE", "BLDG": "BUILDING",
    # Address / France
    "RUE": "RUE", "BD": "BOULEVARD", "AV": "AVENUE", "IMP": "IMPASSE", "ALL": "ALLEE",
    "CHE": "CHEMIN", "RTE": "ROUTE", "ZA": "ZONE", "ZI": "ZONE", "ZAC": "ZONE",
    "SOC": "SOCIETE", "ETS": "ETABLISSEMENTS"
}

GENERIC_STOPWORDS = {
    "the", "and", "of", "in", "for", "to", "at", "by", "on", "a", "an",
    "co", "company", "services", "service", "enterprises", "enterprise",
    "group", "holdings", "holding", "solutions", "solution", "systems", "system",
    "associates", "consultants", "consulting", "management", "international", "national",
    "india", "us", "usa", "france", "corporation", "limited", "private", "llc", "inc", "pvt", "ltd"
}

# Indian transliteration phonetic mappings
INDIAN_PHONETIC_PAIRS = [
    (r'ee', 'i'),
    (r'oo', 'u'),
    (r'w', 'v'),
    (r'sh', 's'),
    (r'ch', 'c'),
    (r'dh', 'd'),
    (r'th', 't'),
    (r'bh', 'b'),
    (r'ph', 'f'),
    (r'kh', 'k'),
    (r'gh', 'g'),
    (r'jh', 'j')
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


def get_consonant_skeleton(word: str, country: str = "") -> str:
    """
    Compute consonant skeleton with phonetic mapping and duplicate letter collapsing.
    """
    if not word:
        return ""
    w = word.lower().strip()
    if country == "India" or True:
        for pat, rep in INDIAN_PHONETIC_PAIRS:
            w = re.sub(pat, rep, w)
    first = w[0]
    rest = re.sub(r'[aeiouy\s]', '', w[1:])
    combined = first + rest
    # collapse consecutive duplicates
    collapsed = re.sub(r'(.)\1+', r'\1', combined)
    return collapsed


def soundex(word: str) -> str:
    """Standard American Soundex encoding for phonetic blocking."""
    if not word or not word.isalpha():
        return ""
    w = word.upper()
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first = w[0]
    encoded = []
    prev_code = mapping.get(first, '')
    for char in w[1:]:
        code = mapping.get(char, '')
        if code and code != prev_code:
            encoded.append(code)
            prev_code = code
        elif not code and char not in 'HW':
            prev_code = ''
    res = (first + ''.join(encoded) + '000')[:4]
    return res


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
            "distinctive_tokens": [],
            "phonetic_tokens": [],
            "soundex_tokens": [],
            "name_abbrev_expanded": "",
        }

    # Unicode NFKC
    nfkc = unicodedata.normalize('NFKC', raw)
    # Lowercase
    lower = nfkc.lower()
    # Ascii-clean
    ascii_safe = strip_accents(lower)

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

    # Distinctive (non-generic) tokens
    distinctive_tokens = [t for t in no_suffix_tokens if t not in GENERIC_STOPWORDS and len(t) >= 3]

    # Abbreviation expansion
    words_expanded = [COMMON_ABBREVIATIONS.get(t.upper(), t.upper()).lower() for t in tokens]
    name_abbrev_expanded = " ".join(words_expanded)

    # Phonetic tokens & soundex
    phonetic_tokens = [get_consonant_skeleton(t, country=country) for t in distinctive_tokens if len(t) >= 3]
    soundex_tokens = [soundex(t) for t in distinctive_tokens if len(t) >= 3 and soundex(t)]

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
        "distinctive_tokens": distinctive_tokens,
        "phonetic_tokens": phonetic_tokens,
        "soundex_tokens": soundex_tokens,
        "name_abbrev_expanded": name_abbrev_expanded,
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
            return m[-1]
    elif country in ["US", "France"]:
        m = re.findall(r'\b[0-9]{5}\b', text)
        if m:
            return m[-1]
    else:
        m = re.findall(r'\b[0-9]{5,6}\b', text)
        if m:
            return m[-1]
    return None


def extract_house_number(tokens: List[str]) -> Optional[str]:
    """Extract leading or prominent street/house number."""
    if not tokens:
        return None
    if re.match(r'^\d+[a-z]?$', tokens[0]):
        return tokens[0]
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
            "address_distinctive_tokens": [],
            "street_name_tokens": []
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

    # Address distinctive tokens (non-numeric, length >= 4)
    addr_distinctive = [t for t in tokens if not t.isdigit() and len(t) >= 4 and t not in GENERIC_STOPWORDS]

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
        "address_distinctive_tokens": addr_distinctive,
        "street_name_tokens": addr_distinctive[:3]
    }
