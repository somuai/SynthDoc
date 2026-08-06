import re
from PIL import Image
import numpy as np


class EnsembleOCR:
    """Ensemble OCR engine combining Tesseract and PaddleOCR for robust text extraction
    from Indian identity documents."""

    def __init__(self, use_paddle: bool = False):
        self.use_paddle = use_paddle
        self._tesseract_available = False
        self._paddle_engine = None

        # Check Tesseract availability
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            self._tesseract_available = False

        # Initialize PaddleOCR if requested
        if use_paddle:
            try:
                from paddleocr import PaddleOCR
                self._paddle_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except ImportError:
                self._paddle_engine = None

    def _run_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract OCR."""
        if not self._tesseract_available:
            return ""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
            return text.strip()
        except Exception:
            return ""

    def _run_paddle(self, image: Image.Image) -> str:
        """Extract text using PaddleOCR."""
        if self._paddle_engine is None:
            return ""
        try:
            img_np = np.array(image)
            results = self._paddle_engine.ocr(img_np, cls=True)
            if results and results[0]:
                lines = [line[1][0] for line in results[0] if line[1]]
                return "\n".join(lines)
            return ""
        except Exception:
            return ""

    def _merge_texts(self, tesseract_text: str, paddle_text: str) -> str:
        """Merge OCR outputs, preferring the longer/more complete result."""
        if not tesseract_text:
            return paddle_text
        if not paddle_text:
            return tesseract_text

        # Use the longer text as primary, but also try to recover any unique
        # fields from the shorter one
        if len(paddle_text) > len(tesseract_text):
            primary, secondary = paddle_text, tesseract_text
        else:
            primary, secondary = tesseract_text, paddle_text

        return primary

    def _extract_fields(self, text: str) -> dict:
        """Extract structured fields from raw OCR text."""
        fields = {}

        # PAN pattern: ABCDE1234F
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text.upper())
        if pan_match:
            fields['pan_number'] = pan_match.group()

        # Aadhaar pattern: 1234 5678 9012
        aadhaar_match = re.search(r'\d{4}\s*\d{4}\s*\d{4}', text)
        if aadhaar_match:
            fields['aadhaar_number'] = aadhaar_match.group().replace(' ', '')

        # Name patterns
        name_match = re.search(r'(?:Name|name)\s*[:\-]?\s*([A-Z][a-zA-Z\s]+)', text)
        if name_match:
            fields['name'] = name_match.group(1).strip()

        # DOB patterns
        dob_match = re.search(r'\d{2}[/\-]\d{2}[/\-]\d{4}', text)
        if dob_match:
            fields['dob'] = dob_match.group()

        # Father's name
        father_match = re.search(r"(?:Father|FATHER)(?:'s)?\s*(?:Name)?\s*[:\-]?\s*([A-Z][a-zA-Z\s]+)", text)
        if father_match:
            fields['father_name'] = father_match.group(1).strip()

        # UPI VPA pattern
        vpa_match = re.search(r'[a-zA-Z0-9.]+@[a-zA-Z]+', text)
        if vpa_match:
            fields['upi_vpa'] = vpa_match.group()

        return fields

    def extract(self, image: Image.Image) -> dict:
        """Run ensemble OCR and return structured results.

        Returns:
            dict with keys: 'tesseract', 'paddle', 'merged', 'fields'
        """
        tesseract_text = self._run_tesseract(image)
        paddle_text = self._run_paddle(image) if self.use_paddle else ""
        merged = self._merge_texts(tesseract_text, paddle_text)
        fields = self._extract_fields(merged)

        return {
            'tesseract': tesseract_text,
            'paddle': paddle_text,
            'merged': merged,
            'fields': fields,
        }
