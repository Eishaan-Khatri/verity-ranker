import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verity_ranker.pipeline import run_pipeline


class V1PipelineTest(unittest.TestCase):
    def test_sample_pipeline_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ranked_output.csv"
            report = Path(tmp) / "audit_report.json"
            result = run_pipeline(
                jd_path=ROOT / "data" / "sample" / "jd.txt",
                candidates_dir=ROOT / "data" / "sample" / "candidates",
                output_path=output,
                report_path=report,
                weights_path=ROOT / "configs" / "scoring_weights.json",
                evidence_path=Path(tmp) / "evidence_ledger.json",
                claim_report_path=Path(tmp) / "claim_verification_report.json",
            )
            self.assertEqual(result.candidate_count, 5)
            self.assertTrue(output.exists())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], "v2_verified_ranker")
            self.assertEqual(payload["ranked_candidates"][0]["candidate_id"], "C001")


if __name__ == "__main__":
    unittest.main()
