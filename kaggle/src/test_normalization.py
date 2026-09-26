"""
Unit tests for Stage 1 normalization module.
"""

import unittest
from normalization import (
    normalize_business_name,
    normalize_business_address,
    extract_postal_code,
    extract_house_number
)


class TestNormalization(unittest.TestCase):

    def test_name_normalization_us(self):
        res = normalize_business_name("Apple Inc.", country="US")
        self.assertEqual(res["name_raw"], "Apple Inc.")
        self.assertEqual(res["name_no_legal_suffix"], "apple")
        self.assertIn("INC", res["legal_suffixes"])
        self.assertEqual(res["name_tokens"], ["apple", "inc"])
        self.assertEqual(res["name_tokens_sorted"], "apple inc")

    def test_name_normalization_india(self):
        res = normalize_business_name("Reliance Industries Pvt. Ltd.", country="India")
        self.assertIn("PVT", res["legal_suffixes"])
        self.assertIn("LTD", res["legal_suffixes"])
        self.assertEqual(res["name_no_legal_suffix"], "reliance")
        self.assertEqual(res["name_initials"], "ripl")

    def test_name_normalization_france(self):
        res = normalize_business_name("Boulangerie Saint-Honoré SARL", country="France")
        self.assertEqual(res["name_ascii_when_safe"], "boulangerie saint-honore sarl")
        self.assertIn("SARL", res["legal_suffixes"])
        self.assertEqual(res["name_no_legal_suffix"], "boulangerie saint honore")

    def test_reordered_tokens(self):
        res1 = normalize_business_name("Apex Dental Clinic LLC")
        res2 = normalize_business_name("Clinic Dental Apex")
        self.assertEqual(res1["name_no_legal_suffix_sorted"], res2["name_no_legal_suffix_sorted"])

    def test_postal_code_extraction(self):
        # US ZIP
        self.assertEqual(extract_postal_code("123 Main St, Springfield, IL 62701", country="US"), "62701")
        # India PIN
        self.assertEqual(extract_postal_code("Plot 42, Andheri East, Mumbai 400069", country="India"), "400069")
        # France Code Postal
        self.assertEqual(extract_postal_code("15 Rue de Rivoli, 75001 Paris", country="France"), "75001")

    def test_address_normalization_and_abbreviations(self):
        res = normalize_business_address("100 N. Main St., Ste. 400", country="US")
        self.assertEqual(res["house_number"], "100")
        self.assertIn("street", res["address_tokens"])
        self.assertIn("suite", res["address_tokens"])
        self.assertIn("100", res["numeric_tokens"])


if __name__ == "__main__":
    unittest.main()
