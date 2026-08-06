"""QR code and barcode decoder for UPI QR documents."""

import re
from typing import Optional


def decode_upi_qr(raw_data: str) -> dict:
    """Parse UPI QR code data string.

    UPI QR format: upi://pay?pa=vpa@bank&pn=Name&am=100&cu=INR

    Args:
        raw_data: Raw QR code string.

    Returns:
        Parsed UPI fields dictionary.
    """
    result = {
        "is_upi": False,
        "vpa": None,
        "payee_name": None,
        "amount": None,
        "currency": None,
        "merchant_code": None,
        "transaction_note": None,
    }

    if not raw_data or not raw_data.strip().lower().startswith("upi://"):
        return result

    result["is_upi"] = True

    # Parse query parameters
    params_str = raw_data.split("?", 1)[-1] if "?" in raw_data else ""
    params = {}
    for param in params_str.split("&"):
        if "=" in param:
            key, value = param.split("=", 1)
            params[key.lower().strip()] = value.strip()

    result["vpa"] = params.get("pa")
    result["payee_name"] = params.get("pn")
    result["amount"] = params.get("am")
    result["currency"] = params.get("cu", "INR")
    result["merchant_code"] = params.get("mc")
    result["transaction_note"] = params.get("tn")

    return result


def try_decode_image(image_path: str) -> Optional[str]:
    """Attempt to decode a QR code from an image file.

    Returns the raw QR data string, or None if no QR code found.
    """
    try:
        import zxingcpp
        from PIL import Image
        img = Image.open(image_path)
        results = zxingcpp.read_barcodes(img)
        if results:
            return results[0].text
        return None
    except ImportError:
        return None
    except Exception:
        return None
