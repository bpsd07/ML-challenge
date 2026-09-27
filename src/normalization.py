"""
Stage 1: Advanced Normalization Pipeline (Enhanced for High-Recall Blocking - Stage 2B)
Maintains raw, normalized, and specialized representations of business names and addresses.
Supports country-aware variations for India, US, and France.
Includes:
- Devanagari / Indic script deterministic transliteration
- URL / Domain name stripping and token representation
- Honorific / prefix stripping (M/s, Ms, Mrs, Mr, Smt, Sri, Shri, Dr, The)
- Comprehensive address numeric signature extraction (slash-aware compounds, alphanumeric house tokens, sorted signatures)
"""

import re
import unicodedata
from collections import Counter
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

# Devanagari script transliteration mapping tables
DEV_INDEPENDENT_VOWELS = {
    '\u0905': 'a', '\u0906': 'a', '\u0907': 'i', '\u0908': 'i', '\u0909': 'u',
    '\u090a': 'u', '\u090b': 'ri', '\u090f': 'e', '\u0910': 'ai', '\u0911': 'o',
    '\u0912': 'o', '\u0913': 'o', '\u0914': 'au'
}

DEV_MATRAS = {
    '\u093e': 'a', '\u093f': 'i', '\u0940': 'i', '\u0941': 'u', '\u0942': 'u',
    '\u0943': 'ri', '\u0947': 'e', '\u0948': 'ai', '\u0949': 'o', '\u094a': 'o',
    '\u094b': 'o', '\u094c': 'au'
}

DEV_CONSONANTS = {
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'ch', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092a': 'p', '\u092b': 'f', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'y', '\u0930': 'r', '\u0932': 'l', '\u0933': 'l', '\u0935': 'v',
    '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    '\u0958': 'q', '\u0959': 'kh', '\u095a': 'g', '\u095b': 'z', '\u095c': 'r',
    '\u095d': 'rh', '\u095e': 'f', '\u095f': 'y'
}

DEV_DIGITS = {
    '\u0966': '0', '\u0967': '1', '\u0968': '2', '\u0969': '3', '\u096a': '4',
    '\u096b': '5', '\u096c': '6', '\u096d': '7', '\u096e': '8', '\u096f': '9'
}

# Domain regex
DOMAINS_TLD_REGEX = re.compile(
    r'^(?:https?://)?(?:www\.)?([a-z0-9\-]+)(?:\.com|\.in|\.org|\.net|\.co\.in|\.co|\.us|\.io|\.biz|\.info)(?:/.*)?$',
    re.IGNORECASE
)

# Honorific prefix regex
HONORIFICS_REGEX = re.compile(r'^(?:m/s|m\\s|ms|mrs|mr|smt|sri|shri|dr|the)\b[\s\.\-\/]*', re.IGNORECASE)


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


def transliterate_devanagari(text: str) -> str:
    """
    Deterministic Indic/Devanagari to Latin transliteration.
    Preserves original text, converts Devanagari words to phonetic Latin equivalents.
    Handles inherent 'a' schwa, independent vowels, matras, halant, and special symbols.
    """
    if not text:
        return ""
    text = text.replace('\u0950', ' om ')
    for d_dev, d_lat in DEV_DIGITS.items():
        text = text.replace(d_dev, d_lat)

    res = []
    chars = list(text)
    n = len(chars)
    i = 0
    while i < n:
        c = chars[i]
        if c in DEV_CONSONANTS:
            cons = DEV_CONSONANTS[c]
            if i + 1 < n:
                nxt = chars[i + 1]
                if nxt == '\u094d':  # virama / halant
                    res.append(cons)
                    i += 2
                    continue
                elif nxt in DEV_MATRAS:  # matra replaces inherent 'a'
                    res.append(cons + DEV_MATRAS[nxt])
                    i += 2
                    continue
                elif nxt in DEV_CONSONANTS or nxt in DEV_INDEPENDENT_VOWELS:
                    res.append(cons + 'a')
                    i += 1
                    continue
                elif nxt in ['\u0902', '\u0901']:  # Anusvara / Chandrabindu
                    res.append(cons + 'an')
                    i += 2
                    continue
                elif nxt in ['\u0903']:  # Visarga
                    res.append(cons + 'ah')
                    i += 2
                    continue
                else:
                    # End of word or separator: schwa deletion
                    res.append(cons)
                    i += 1
                    continue
            else:
                res.append(cons)
                i += 1
        elif c in DEV_INDEPENDENT_VOWELS:
            res.append(DEV_INDEPENDENT_VOWELS[c])
            i += 1
        elif c in ['\u0902', '\u0901']:
            res.append('n')
            i += 1
        elif c in ['\u0903']:
            res.append('h')
            i += 1
        elif c == '\u094d':
            i += 1
        else:
            res.append(c)
            i += 1

    out = ''.join(res)
    return re.sub(r'\s+', ' ', out).strip()


INDIC_BASES = {
    'Bengali': 0x0980,
    'Gurmukhi': 0x0A00,
    'Gujarati': 0x0A80,
    'Odia': 0x0B00,
    'Tamil': 0x0B80,
    'Telugu': 0x0C00,
    'Kannada': 0x0C80,
    'Malayalam': 0x0D00
}


def detect_script(text: str) -> str:
    """Detect dominant script of text."""
    if not text:
        return "Latin"
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["Bengali"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["Gurmukhi"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["Gujarati"] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            counts["Odia"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["Tamil"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["Telugu"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["Kannada"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["Malayalam"] += 1
        elif ch.isalpha():
            counts["Latin"] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return "Latin"


def transliterate_indic(text: str) -> str:
    """
    Universal Indic-to-Latin transliterator covering:
    Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam.
    Maps non-Devanagari Brahmic characters to Devanagari ISCII block, then executes
    phonetic transliteration to Latin.
    """
    if not text:
        return ""
    dev_chars = []
    for ch in text:
        cp = ord(ch)
        mapped = False
        for name, base in INDIC_BASES.items():
            if base <= cp <= base + 0x7F:
                dev_chars.append(chr(0x0900 + (cp - base)))
                mapped = True
                break
        if not mapped:
            dev_chars.append(ch)
    dev_text = ''.join(dev_chars)
    return transliterate_devanagari(dev_text)


def strip_domain_tld(text: str) -> Optional[str]:
    """
    Extract domain-stripped name representation for URL-like business names.
    Handles www.example.com, example.com, example.in, wilsonjanennacpa.com - 7713691326.
    """
    if not text:
        return None
    clean = text.lower().strip()
    # Strip trailing phone numbers / dashes
    clean = re.sub(r'[\s\-]+[0-9]{7,12}\b', '', clean).strip()
    # Strip leading honorific if any
    clean = re.sub(r'^(?:m/s|ms|mrs|mr|smt|sri|shri|dr|the)\b[\s\.]*', '', clean).strip()

    m = DOMAINS_TLD_REGEX.search(clean)
    if m:
        domain_body = m.group(1).replace('-', '')
        return domain_body if len(domain_body) >= 3 else None

    m2 = re.search(r'\b([a-z0-9\-]{3,})\.(?:com|in|org|net|co\.in|co|us|biz)\b', clean)
    if m2:
        domain_body = m2.group(1).replace('-', '')
        return domain_body if len(domain_body) >= 3 else None

    return None


def strip_honorifics(text: str) -> Optional[str]:
    """
    Deterministic removal of leading honorifics / prefixes.
    Preserves original text while returning stripped version when present.
    """
    if not text:
        return None
    clean = text.strip()
    stripped = HONORIFICS_REGEX.sub('', clean).strip()
    return stripped if stripped and stripped.lower() != clean.lower() else None


def extract_address_signatures(raw_address: str) -> Dict:
    """
    Extract normalized numeric components, slash-aware compounds, alphanumeric house tokens,
    and sorted numeric signatures.
    Handles formats: 80/28, 38/2, Pno-S-513, A-25, H.No G-755, Door No 825, #A-303, Noa4/524/2, Plot No.1098/100, S. No. 192/5.
    """
    if not raw_address:
        return {
            "numeric_atoms": set(),
            "slash_compounds": set(),
            "house_tokens": set(),
            "sorted_numeric_sig": None,
            "house_signature": None
        }

    clean = raw_address.lower().strip()

    # 1. All numeric atoms (isolated digit sequences) with zero-stripping
    numeric_atoms = set()
    for d in re.findall(r'\b\d+\b', clean):
        numeric_atoms.add(d)
        norm_d = d.lstrip('0')
        if norm_d:
            numeric_atoms.add(norm_d)

    # Prefix-digit extractions (e.g. C-323 -> 323, A-95 -> 95, D/903 -> 903, #570 -> 570)
    for pfx, num in re.findall(r'([a-z/#\-]+)(\d+)', clean):
        numeric_atoms.add(num)
        norm_num = num.lstrip('0')
        if norm_num:
            numeric_atoms.add(norm_num)

    # 2. Slash-aware numeric components (e.g. 80/28, 38/2, 1098/100, 4/524/2, 8-0/28, 192/5)
    slash_compounds = set()
    slash_patterns = re.findall(r'(?:\b|[a-z])(\d+(?:[/-]\d+)+)\b', clean)
    for sp in slash_patterns:
        parts = [p for p in re.split(r'[/-]', sp) if p.isdigit()]
        if len(parts) >= 2:
            slash_compounds.add('_'.join(parts))
            slash_compounds.add(''.join(parts))
            for p in parts:
                numeric_atoms.add(p)
                norm_p = p.lstrip('0')
                if norm_p:
                    numeric_atoms.add(norm_p)

    # 3. Alphanumeric house tokens (e.g., Pno-S-513, A-25, H.No G-755, #A-303, Door No 825, Noa4/524/2)
    house_tokens = set()

    # Standalone letter+digits like #A-303, A-25, S-513, G-755
    for m in re.finditer(r'#?([a-z]{1,2})[\s\-]?(\d{1,5})\b', clean):
        pfx, num = m.groups()
        if pfx not in ['in', 'to', 'at', 'on', 'th', 'st', 'nd', 'rd', 'fl']:
            house_tokens.add(f"{pfx}{num}")
            numeric_atoms.add(num)

    # Prefix-driven patterns (door no 825, shop no-46, flat no 303, pno-s-513, plot no 284, h.no g-755)
    kw_pattern = r'\b(door|shop|flat|pno|plot|h[\.\s]?no|house[\.\s]?no|unit|room|block|blk|noa)[\s\.\-\#:]*(?:no[\s\.\-:]*)?([a-z0-9\-\/]+)'
    for m in re.finditer(kw_pattern, clean):
        kw, val = m.groups()
        val_clean = re.sub(r'[^a-z0-9]', '', val)
        if val_clean and len(val_clean) <= 10:
            house_tokens.add(f"{kw}_{val_clean}")
            house_tokens.add(val_clean)
            for d in re.findall(r'\d+', val):
                numeric_atoms.add(d)

    # 4. Canonical house signature
    house_sig = None
    if slash_compounds:
        house_sig = sorted(list(slash_compounds))[0]
    elif house_tokens:
        house_sig = sorted(list(house_tokens))[0]
    elif numeric_atoms:
        house_sig = sorted(list(numeric_atoms), key=lambda x: (len(x), x))[0]

    # 5. Sorted numeric signature (sorted unique numeric atoms joined by underscore)
    sorted_unique = sorted(list(numeric_atoms), key=lambda x: (len(x), int(x) if x.isdigit() else 0))
    sorted_num_sig = '_'.join(sorted_unique[:4]) if sorted_unique else None

    return {
        "numeric_atoms": numeric_atoms,
        "slash_compounds": slash_compounds,
        "house_tokens": house_tokens,
        "sorted_numeric_sig": sorted_num_sig,
        "house_signature": house_sig
    }


def normalize_business_name(raw_name: Optional[str], country: str = "US") -> Dict:
    """
    Generate multiple representations and extracted features for a business name.
    Preserves raw name while providing normalized views for blocking and matching.
    Includes transliterated views for Devanagari, domain cleaning, and honorific stripping.
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
            "domain_clean": None,
            "name_no_honorific": None,
            "no_honorific_tokens": [],
            "no_honorific_distinctive": [],
            "name_transliterated": "",
            "transliterated_tokens": [],
            "transliterated_distinctive": [],
            "name_concat_distinctive": ""
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
    distinctive_tokens = [t for t in no_suffix_tokens if t not in GENERIC_STOPWORDS and len(t) >= 2]

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

    # --- Targeted Additions ---
    # Domain / URL cleaning
    domain_clean = strip_domain_tld(raw)

    # Honorific stripping
    honorific_clean = strip_honorifics(raw)
    honorific_tokens = []
    honorific_distinctive = []
    if honorific_clean:
        h_ascii = strip_accents(unicodedata.normalize('NFKC', honorific_clean).lower())
        h_alnum = re.sub(r'[^a-z0-9\s]', ' ', h_ascii).strip()
        honorific_tokens = [t for t in h_alnum.split() if t and t.upper() not in LEGAL_SUFFIXES_LIST]
        honorific_distinctive = [t for t in honorific_tokens if t not in GENERIC_STOPWORDS and len(t) >= 2]

    # Multi-script Indic transliteration & detection
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in raw)
    detected_script = detect_script(raw)
    translit_raw = transliterate_indic(raw) if has_indic else ""
    translit_tokens = []
    translit_distinctive = []
    translit_phonetic = []
    if translit_raw:
        t_clean = re.sub(r'[^a-z0-9\s]', ' ', translit_raw.lower())
        t_tokens = [t for t in t_clean.split() if t]
        translit_tokens = [t for t in t_tokens if t.upper() not in LEGAL_SUFFIXES_LIST]
        translit_distinctive = [t for t in translit_tokens if t not in GENERIC_STOPWORDS and len(t) >= 2]
        translit_phonetic = [get_consonant_skeleton(t, country=country) for t in translit_distinctive]

    # Concatenated distinctive tokens (for matching against domain-stripped candidate names)
    concat_distinctive = "".join(distinctive_tokens) if len(distinctive_tokens) >= 2 else ""

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
        "domain_clean": domain_clean,
        "detected_script": detected_script,
        "name_no_honorific": honorific_clean,
        "no_honorific_tokens": honorific_tokens,
        "no_honorific_distinctive": honorific_distinctive,
        "name_transliterated": translit_raw,
        "transliterated_tokens": translit_tokens,
        "transliterated_distinctive": translit_distinctive,
        "transliterated_phonetic": translit_phonetic,
        "name_concat_distinctive": concat_distinctive,
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
    Includes slash-aware numeric components, alphanumeric house tokens, and sorted signatures.
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
            "numeric_atoms": set(),
            "slash_compounds": set(),
            "house_tokens": set(),
            "sorted_numeric_sig": None,
            "house_signature": None,
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
    house_num = extract_house_number(tokens)

    # Address numeric signatures & house token extraction
    addr_sigs = extract_address_signatures(raw)

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
        "numeric_tokens": addr_sigs["numeric_atoms"],
        "numeric_atoms": addr_sigs["numeric_atoms"],
        "slash_compounds": addr_sigs["slash_compounds"],
        "house_tokens": addr_sigs["house_tokens"],
        "sorted_numeric_sig": addr_sigs["sorted_numeric_sig"],
        "house_signature": addr_sigs["house_signature"],
        "address_char_3gram": c3,
        "address_char_4gram": c4,
        "address_char_5gram": c5,
        "address_distinctive_tokens": addr_distinctive,
        "street_name_tokens": addr_distinctive[:3]
    }
