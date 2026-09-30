import re
import sys
import unicodedata
from typing import List, Tuple
import pandas as pd

# Ensure Unicode prints cleanly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Character replacements: German ligature, French ligatures, ampersand
LIGATURE_MAP = str.maketrans({
    "ß": "ss",
    "œ": "oe",
    "æ": "ae",
    "Œ": "oe",
    "Æ": "ae",
    "&": " and ",
})

# Punctuation and whitespace regular expressions
PUNCT_RE = re.compile(r"[\W_]+")

# Address extraction regular expressions
POSTAL_RE = re.compile(r"\b\d{5,6}\b")
HOUSE_NO_RE = re.compile(r"\b\d+[a-z]?\b")
LANDMARK_RE = re.compile(
    r"\b(?:near|opp|opposite|behind|beside|next to)\b\s+(.*)",
    flags=re.IGNORECASE,
)

# Acronym dot regex (e.g. S.A.S. -> SAS, L.L.C. -> LLC)
DOT_ACRONYM_RE = re.compile(r"(?<=\b[a-zA-Z])\.")

# Universal abbreviation dictionary (US, India, France merged)
# Normalization only needs to be consistent across sources.
ABBREVIATIONS = {
    # Legal / corporate
    "private": "pvt",
    "limited": "ltd",
    "corporation": "corp",
    "company": "co",
    "incorporated": "inc",
    # Street / address (US / India / France)
    "street": "st",
    "road": "rd",
    "drive": "dr",
    "avenue": "ave",
    "av": "ave",
    "boulevard": "blvd",
    "bd": "blvd",
    "bld": "blvd",
    "suite": "ste",
    "apartment": "apt",
    "floor": "fl",
    "flr": "fl",
    "number": "no",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "route": "rte",
    "center": "ctr",
    "centre": "ctr",
    "plaza": "plz",
    "place": "pl",
    "building": "bldg",
    "block": "blk",
}

# Legal suffixes and stopwords for name_core
LEGAL_STOPWORDS = {
    "pvt", "ltd", "llc", "inc", "corp", "co",
    "sarl", "sas", "sa", "eurl",
    "the", "and", "of",
}


def _clean_and_abbreviate(text: str) -> str:
    """Strip accents with NFKD, lowercase, replace punctuation with space,
    collapse whitespace, and canonicalize abbreviations token-wise."""
    if not text:
        return ""
    # 1. Ligatures and '&'
    t = text.translate(LIGATURE_MAP)
    # 2. Decompose accents via NFKD and drop combining marks
    nfkd = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in nfkd if not unicodedata.combining(c))
    # 3. Collapse dots in acronyms (e.g. S.A.S. -> SAS, L.L.C. -> LLC)
    t = DOT_ACRONYM_RE.sub("", t)
    # 4. Lowercase
    t = t.lower()
    # 5. Punctuation -> space
    t = PUNCT_RE.sub(" ", t).strip()
    # 6. Token-wise abbreviation canonicalization
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in t.split()]
    return " ".join(tokens)


def _make_name_core(name_full: str) -> str:
    """Drop legal suffixes and stopwords from name_full. Fallback to name_full if empty."""
    tokens = name_full.split()
    core_tokens = [tok for tok in tokens if tok not in LEGAL_STOPWORDS]
    return " ".join(core_tokens) if core_tokens else name_full


def _extract_postal(addr: str) -> str:
    """Extract last match of \b\d{5,6}\b in the address ('' if none)."""
    matches = POSTAL_RE.findall(addr)
    return matches[-1] if matches else ""


def _extract_house_no(addr: str) -> str:
    """Extract first numeric token in address ('' if none)."""
    match = HOUSE_NO_RE.search(addr)
    return match.group(0).lower() if match else ""


def _extract_landmark(addr: str) -> str:
    """Extract text after landmark keywords (near/opp/opposite/behind/beside/next to)."""
    match = LANDMARK_RE.search(addr)
    if not match:
        return ""
    raw_tail = match.group(1).strip()
    return _clean_and_abbreviate(raw_tail)


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input records DataFrame.
    
    Input DataFrame must have columns: id, source, name, address, country.
    
    Returns DataFrame with EXACTLY these 9 columns (contract):
    id, source, country, name_full, name_core, addr_norm, postal, house_no, landmark
    """
    # 1. Country: lowercase and strip
    country = df["country"].astype(str).str.strip().str.lower().astype("category")
    
    # 2. Source: categorical
    source = df["source"].astype("category")
    
    # 3. Process names fast via list comprehension
    raw_names = df["name"].fillna("").astype(str).values
    name_full_list = [_clean_and_abbreviate(n) for n in raw_names]
    name_core_list = [_make_name_core(nf) for nf in name_full_list]
    
    # 4. Process addresses fast
    raw_addrs = df["address"].fillna("").astype(str).values
    addr_norm_list = [_clean_and_abbreviate(a) for a in raw_addrs]
    
    # 5. Extract structured fields
    # Extract postal from raw address (or normalized address, both contain the digits)
    postal_list = [_extract_postal(a) for a in raw_addrs]
    # Extract house_no from addr_norm where tokens are separated cleanly
    house_no_list = [_extract_house_no(an) for an in addr_norm_list]
    # Extract landmark from raw address and normalize the captured landmark
    landmark_list = [_extract_landmark(a) for a in raw_addrs]
    
    # 6. Assemble DataFrame with exact contract columns
    out_df = pd.DataFrame({
        "id": df["id"].values,
        "source": source,
        "country": country,
        "name_full": name_full_list,
        "name_core": name_core_list,
        "addr_norm": addr_norm_list,
        "postal": postal_list,
        "house_no": house_no_list,
        "landmark": landmark_list,
    })
    
    return out_df


def run_self_test():
    """Self-test covering ~10 hand-written examples across US, India, and France."""
    test_cases: List[Tuple[str, str, str, str]] = [
        # (id, country, raw_name, raw_address)
        ("T1", "US", "Google & Sons, Inc.", "1600 Amphitheatre Parkway, Suite 100, Mountain View, CA 94043"),
        ("T2", "US", "Standard Environmental Corporation", "1347 Truitt Road, Sylacauga, AL"),
        ("T3", "US", "Pacheco Dynamics L.L.C.", "4004 Gloria Drive, Acton, TX 76049"),
        ("T4", "India", "Reliance Industries Private Limited", "Near Reliance Mall, Plot 42, MG Road, Mumbai 400001"),
        ("T5", "India", "Vcan Real LLP", "Opposite City Hospital, No. 68/3-1, Bangalore 560001"),
        ("T6", "India", "Shree Ganesh Traders Pvt Ltd", "Behind SBI Bank, Shop No. 12, Station Road, Surat"),
        ("T7", "India", "Bari Trading Care", "Beside Post Office, 115 MG Road, Trivandrum 695001"),
        ("T8", "France", "Société Générale S.A.S.", "29 Boulevard Haussmann, 75009 Paris"),
        ("T9", "France", "Cœur de France S.A.R.L.", "15 Rue de l'Étoile, 69002 Lyon"),
        ("T10", "France", "Café de la Paix E.U.R.L.", "5 Avenue de l'Opéra, 75001 Paris"),
        ("T11", "France", "Boulangerie Saint-Honoré", "Next to Métro Bastille, 12 Rue de la Roquette, 75011 Paris"),
    ]
    
    df_raw = pd.DataFrame({
        "id": [c[0] for c in test_cases],
        "source": ["S1"] * len(test_cases),
        "country": [c[1] for c in test_cases],
        "name": [c[2] for c in test_cases],
        "address": [c[3] for c in test_cases],
    })
    
    df_norm = normalize_records(df_raw)
    
    print("=" * 80)
    print("NORMALIZATION SELF-TEST (11 Hand-Written Examples across US, India, France)")
    print("=" * 80)
    for i, row in df_norm.iterrows():
        raw_name = test_cases[i][2]
        raw_addr = test_cases[i][3]
        print(f"[{row['id']}] Country: {row['country']} | Source: {row['source']}")
        print(f"  Raw Name  : {raw_name}")
        print(f"  Name Full : {row['name_full']}")
        print(f"  Name Core : {row['name_core']}")
        print(f"  Raw Addr  : {raw_addr}")
        print(f"  Addr Norm : {row['addr_norm']}")
        print(f"  Postal    : '{row['postal']}' | House No: '{row['house_no']}' | Landmark: '{row['landmark']}'")
        print("-" * 80)
        
    # Assertions for contract compliance
    expected_cols = ["id", "source", "country", "name_full", "name_core", "addr_norm", "postal", "house_no", "landmark"]
    assert list(df_norm.columns) == expected_cols, f"Columns mismatch: {list(df_norm.columns)} vs {expected_cols}"
    
    # Specific assertions
    # T1: Inc removed from core, & -> and, postal 94043, house_no 1600
    assert "inc" not in df_norm.loc[0, "name_core"].split()
    assert df_norm.loc[0, "postal"] == "94043"
    assert df_norm.loc[0, "house_no"] == "1600"
    
    # T4: landmark extracted, pvt/ltd removed from core
    assert "pvt" not in df_norm.loc[3, "name_core"].split()
    assert df_norm.loc[3, "landmark"].startswith("reliance mall")
    assert df_norm.loc[3, "postal"] == "400001"
    
    # T8: French accents é -> e, SAS removed, Boulevard -> blvd
    assert df_norm.loc[7, "name_full"] == "societe generale sas"
    assert df_norm.loc[7, "name_core"] == "societe generale"
    assert "blvd" in df_norm.loc[7, "addr_norm"]
    assert df_norm.loc[7, "postal"] == "75009"
    
    # T9: French ligature œ -> oe
    assert "coeur" in df_norm.loc[8, "name_full"]
    assert "sarl" not in df_norm.loc[8, "name_core"].split()
    
    # T10: e-acute in Café -> cafe, eurl removed, avenue -> ave
    assert df_norm.loc[9, "name_full"].startswith("cafe de la paix")
    assert "eurl" not in df_norm.loc[9, "name_core"].split()
    assert "ave" in df_norm.loc[9, "addr_norm"]
    
    print("\nALL SELF-TEST ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_self_test()
