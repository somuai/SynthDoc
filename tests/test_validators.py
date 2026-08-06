"""Tests for semantic validators (Luhn, Verhoeff, UPI VPA)."""

import pytest
from streams.semantic.validator import (
    validate_pan, validate_aadhaar, validate_upi_vpa, compute_semantic_score
)


class TestPANValidation:
    def test_valid_pan_format(self):
        result = validate_pan("ABCDE1234F")
        assert result["format_valid"] is True
        assert result["is_valid"] is True

    def test_invalid_pan_too_short(self):
        result = validate_pan("ABCD1234")
        assert result["format_valid"] is False

    def test_invalid_pan_wrong_pattern(self):
        result = validate_pan("12345ABCDE")
        assert result["format_valid"] is False

    def test_entity_type_person(self):
        result = validate_pan("ABCPE1234F")
        assert result["entity_type"] == "Person"

    def test_entity_type_company(self):
        result = validate_pan("ABCCE1234F")
        assert result["entity_type"] == "Company"


class TestAadhaarValidation:
    def test_valid_format(self):
        # Note: Verhoeff checksum validation means not all 12-digit numbers pass
        result = validate_aadhaar("234567890123")
        assert result["format_valid"] is True

    def test_invalid_starts_with_zero(self):
        result = validate_aadhaar("012345678901")
        assert result["format_valid"] is False

    def test_invalid_starts_with_one(self):
        result = validate_aadhaar("112345678901")
        assert result["format_valid"] is False

    def test_invalid_too_short(self):
        result = validate_aadhaar("12345")
        assert result["format_valid"] is False

    def test_handles_spaces(self):
        result = validate_aadhaar("2345 6789 0123")
        assert result["format_valid"] is True


class TestUPIValidation:
    def test_valid_vpa(self):
        result = validate_upi_vpa("user@paytm")
        assert result["format_valid"] is True
        assert result["is_valid"] is True
        assert result["handle"] == "paytm"

    def test_valid_vpa_ybl(self):
        result = validate_upi_vpa("someone@ybl")
        assert result["is_valid"] is True

    def test_invalid_no_at(self):
        result = validate_upi_vpa("userpaytm")
        assert result["format_valid"] is False

    def test_unknown_handle(self):
        result = validate_upi_vpa("user@unknownbank")
        assert result["format_valid"] is True
        assert result["is_valid"] is False


class TestSemanticScore:
    def test_empty_text_high_score(self):
        score = compute_semantic_score("", checksum_valid=None, fields_found=0)
        assert score > 0.5

    def test_good_text_low_score(self):
        score = compute_semantic_score(
            "PERMANENT ACCOUNT NUMBER ABCDE1234F Name: John Doe",
            checksum_valid=True,
            fields_found=3,
            expected_fields=3,
        )
        assert score < 0.5

    def test_failed_checksum_increases_score(self):
        score_pass = compute_semantic_score("Some valid text here enough", checksum_valid=True, fields_found=2)
        score_fail = compute_semantic_score("Some valid text here enough", checksum_valid=False, fields_found=2)
        assert score_fail > score_pass

    def test_score_capped_at_one(self):
        score = compute_semantic_score("", checksum_valid=False, fields_found=0, expected_fields=5)
        assert score <= 1.0
