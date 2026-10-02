from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp031_focus_mask_group_ablation as generator


class Exp031GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_contains_grouped_arms(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp031_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                sources = []
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        source = "".join(item["source"])
                        compile(source, item.get("id", "cell"), "exec")
                        sources.append(source)
                all_source = "\n".join(sources)
                self.assertIn("PURE_MASK_FEATURES", all_source)
                self.assertIn("logistic_mask_geometry", all_source)
                self.assertIn("logistic_mask_quality", all_source)
                self.assertIn("logistic_mask_source", all_source)
                self.assertIn("paired_video_bootstrap.csv", all_source)
                self.assertIn("TARGET_VIDEO_COUNT = 24", all_source)
        finally:
            generator.OUT = original

    def test_notebook_is_diagnostic_only(self) -> None:
        source = generator.RANKER + generator.OUTPUTS
        self.assertNotIn("submission.csv", source)
        self.assertIn("'official_cv': None", source)
        self.assertIn("'submission_generated': False", source)


if __name__ == "__main__":
    unittest.main()
