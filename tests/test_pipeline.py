"""Integration tests for the SynthDoc pipeline."""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock


class TestDocumentClassification:
    def test_pan_by_filename(self):
        from api.pipeline import _classify_document_type
        assert _classify_document_type("pan_card_001.jpg") == "PAN_CARD"

    def test_aadhaar_by_filename(self):
        from api.pipeline import _classify_document_type
        assert _classify_document_type("aadhaar_front.png") == "AADHAAR"

    def test_passport_by_filename(self):
        from api.pipeline import _classify_document_type
        assert _classify_document_type("passport_scan.jpg") == "PASSPORT"

    def test_voter_by_filename(self):
        from api.pipeline import _classify_document_type
        assert _classify_document_type("voter_id.jpg") == "VOTER_ID"

    def test_driving_license_by_filename(self):
        from api.pipeline import _classify_document_type
        assert _classify_document_type("driving_license.jpg") == "DRIVING_LICENSE"

    def test_default_to_pan(self):
        from api.pipeline import _classify_document_type
        assert _classify_document_type("unknown_doc.jpg") == "PAN_CARD"


class TestDeviceSelection:
    def test_returns_valid_device(self):
        from api.pipeline import _get_device
        device = _get_device()
        assert device.type in ["cpu", "cuda"]
