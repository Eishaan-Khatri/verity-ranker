import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verity_ranker.candidates import parse_candidate
from verity_ranker.jd import parse_jd
from verity_ranker.verification import build_verification_index, verify_candidate_claims


class ClaimVerificationTest(unittest.TestCase):
    def setUp(self):
        self.role = parse_jd((ROOT / "data" / "sample" / "jd.txt").read_text(encoding="utf-8"))

    def test_negated_claim_is_unsupported(self):
        candidate = parse_candidate(ROOT / "data" / "sample" / "candidates" / "C002.txt")
        index = build_verification_index(verify_candidate_claims(candidate, self.role))
        self.assertEqual(index[candidate.candidate_id]["machine learning"].status, "unsupported")
        self.assertEqual(index[candidate.candidate_id]["machine learning"].proof_strength, 0.0)

    def test_skills_line_only_claim_is_weak(self):
        candidate = parse_candidate(ROOT / "data" / "sample" / "candidates" / "C005.txt")
        index = build_verification_index(verify_candidate_claims(candidate, self.role))
        self.assertEqual(index[candidate.candidate_id]["fastapi"].status, "weakly_supported")
        self.assertEqual(index[candidate.candidate_id]["fastapi"].proof_strength, 0.15)

    def test_action_evidence_is_verified(self):
        candidate = parse_candidate(ROOT / "data" / "sample" / "candidates" / "C001.txt")
        index = build_verification_index(verify_candidate_claims(candidate, self.role))
        self.assertEqual(index[candidate.candidate_id]["fastapi"].status, "verified")
        self.assertEqual(index[candidate.candidate_id]["fastapi"].proof_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
