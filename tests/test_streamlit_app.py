import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app


class StreamlitAppTest(unittest.TestCase):
    def test_sample_data_runner(self):
        _output_path, _report_path, rows, report = app.run_with_sample_data()
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["candidate_id"], "C001")
        self.assertEqual(report["version"], "v1_basic_ranker")


if __name__ == "__main__":
    unittest.main()

