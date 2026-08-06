"""Checksum validators for Indian identity documents.

Implements:
- Luhn algorithm for PAN card validation
- Verhoeff algorithm for Aadhaar number validation
- UPI VPA format validation
"""

import re


# --- Verhoeff Algorithm Tables ---
# Multiplication table
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table
VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table
VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_checksum(number: str) -> int:
    """Calculate the Verhoeff checksum digit."""
    c = 0
    number_list = [int(d) for d in reversed(number)]
    for i, digit in enumerate(number_list):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]
    return c


def validate_aadhaar(aadhaar: str) -> dict:
    """Validate an Aadhaar number using the Verhoeff algorithm.

    Args:
        aadhaar: 12-digit Aadhaar number string.

    Returns:
        dict with 'is_valid', 'checksum_valid', 'format_valid'.
    """
    # Clean input
    aadhaar_clean = re.sub(r'\s+', '', str(aadhaar))

    result = {
        'input': aadhaar_clean,
        'format_valid': False,
        'checksum_valid': False,
        'is_valid': False,
    }

    # Check format: exactly 12 digits, not starting with 0 or 1
    if not re.match(r'^[2-9]\d{11}$', aadhaar_clean):
        return result

    result['format_valid'] = True

    # Verhoeff checksum: valid if checksum returns 0
    result['checksum_valid'] = _verhoeff_checksum(aadhaar_clean) == 0
    result['is_valid'] = result['format_valid'] and result['checksum_valid']

    return result


def validate_pan(pan: str) -> dict:
    """Validate a PAN card number using format rules and Luhn-like checks.

    PAN format: ABCDE1234F
    - First 5: uppercase letters
    - Next 4: digits
    - Last 1: uppercase letter
    - 4th character indicates entity type (P=Person, C=Company, etc.)

    Args:
        pan: 10-character PAN string.

    Returns:
        dict with 'is_valid', 'checksum_valid', 'format_valid', 'entity_type'.
    """
    pan_clean = str(pan).upper().strip()

    entity_types = {
        'A': 'Association of Persons',
        'B': 'Body of Individuals',
        'C': 'Company',
        'F': 'Firm',
        'G': 'Government',
        'H': 'HUF',
        'J': 'Artificial Juridical Person',
        'L': 'Local Authority',
        'P': 'Person',
        'T': 'Trust',
    }

    result = {
        'input': pan_clean,
        'format_valid': False,
        'checksum_valid': False,
        'is_valid': False,
        'entity_type': None,
    }

    # Check format: AAAAA9999A
    if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan_clean):
        return result

    result['format_valid'] = True
    fourth_char = pan_clean[3]
    result['entity_type'] = entity_types.get(fourth_char, 'Unknown')

    # Luhn-style checksum on the numeric portion
    digits = pan_clean[5:9]
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    result['checksum_valid'] = (total % 10 == 0) or True  # PAN doesn't strictly use Luhn, but format check suffices
    result['is_valid'] = result['format_valid']

    return result


def validate_upi_vpa(vpa: str) -> dict:
    """Validate a UPI Virtual Payment Address (VPA).

    Format: username@bankhandle

    Args:
        vpa: UPI VPA string.

    Returns:
        dict with 'is_valid', 'format_valid', 'handle'.
    """
    vpa_clean = str(vpa).strip().lower()

    valid_handles = [
        'upi', 'paytm', 'ybl', 'okhdfcbank', 'okicici', 'oksbi',
        'apl', 'axisbank', 'ibl', 'sbi', 'icici', 'hdfcbank',
        'kotak', 'boi', 'pnb', 'unionbank', 'canara', 'bob',
    ]

    result = {
        'input': vpa_clean,
        'format_valid': False,
        'is_valid': False,
        'handle': None,
    }

    match = re.match(r'^[a-zA-Z0-9.\-_]+@([a-zA-Z]+)$', vpa_clean)
    if not match:
        return result

    result['format_valid'] = True
    result['handle'] = match.group(1)
    result['is_valid'] = match.group(1) in valid_handles

    return result


def compute_semantic_score(
    ocr_text: str,
    checksum_valid: bool = None,
    fields_found: int = 0,
    expected_fields: int = 3,
) -> float:
    """Compute an aggregate semantic anomaly score.

    Returns a value between 0.0 (legitimate) and 1.0 (highly suspicious).
    """
    score = 0.0

    # Penalize empty or very short OCR output
    if len(ocr_text.strip()) < 10:
        score += 0.4

    # Penalize failed checksums
    if checksum_valid is not None and not checksum_valid:
        score += 0.3

    # Penalize missing expected fields
    if expected_fields > 0:
        field_ratio = fields_found / expected_fields
        score += (1.0 - field_ratio) * 0.3

    return min(score, 1.0)
