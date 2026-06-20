import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class StreamlitAppTest(unittest.TestCase):
    @patch("streamlit.set_page_config")
    @patch("streamlit.markdown")
    def test_v2_pipeline_helper(self, _md, _cfg):
        import app

        jd_text, candidate_texts = app._load_sample_inputs()
        self.assertTrue(len(jd_text) > 50)
        self.assertEqual(len(candidate_texts), 4)

        result = app.run_v2_pipeline(
            jd_text=jd_text,
            candidate_texts=candidate_texts,
            force_fallback=True,
            k=25,
            stability_runs=3,
        )
        self.assertIn("ranked_output", result)
        self.assertEqual(len(result["ranked_output"]), 4)
        self.assertIsNotNone(result.get("recruiter_report"))
        self.assertIsNotNone(result.get("evidence_ledger"))


if __name__ == "__main__":
    unittest.main()

