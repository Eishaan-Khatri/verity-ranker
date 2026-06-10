import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app


class StreamlitAppTest(unittest.TestCase):
    def test_sample_data_runner(self):
        _output_path, _report_path, _evidence_path, _claims_path, rows, report, evidence, claims = app.run_with_sample_data()
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["candidate_id"], "C001")
        self.assertEqual(report["version"], "v2_verified_ranker")
        self.assertGreater(len(evidence), 0)
        self.assertGreater(len(claims), 0)


if __name__ == "__main__":
    unittest.main()
