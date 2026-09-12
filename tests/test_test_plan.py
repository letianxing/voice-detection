import json
import tempfile
import unittest
from pathlib import Path

from voice_detection.test_plan import default_asr_scenarios, write_asr_template


class AsrTemplateTest(unittest.TestCase):
    def test_template_contains_required_acceptance_scenarios(self):
        scenario_ids = {row["scenario_id"] for row in default_asr_scenarios()}

        self.assertIn("quiet_natural_sentence", scenario_ids)
        self.assertIn("short_word_stop", scenario_ids)
        self.assertIn("numbers_dates_units", scenario_ids)
        self.assertIn("robot_speaking_barge_in", scenario_ids)
        self.assertIn("two_speakers_overlap", scenario_ids)
        self.assertIn("long_running_restart", scenario_ids)
        self.assertIn("eating_while_speaking", scenario_ids)
        self.assertIn("doorway_occlusion", scenario_ids)
        self.assertIn("multi_direction_overlap", scenario_ids)

    def test_write_jsonl_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asr.jsonl"
            write_asr_template(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertGreater(len(rows), 20)
        self.assertIn("environment", rows[0])
        self.assertIn("timing", rows[0])
        self.assertNotIn("intent", rows[0])
        self.assertNotIn("permission", rows[0])


if __name__ == "__main__":
    unittest.main()
