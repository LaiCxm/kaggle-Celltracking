from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp030_rich_candidate_ranker as generator


class Exp030GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_contains_rich_arms(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp030_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                sources = []
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        source = "".join(item["source"])
                        compile(source, item.get("id", "cell"), "exec")
                        sources.append(source)
                all_source = "\n".join(sources)
                self.assertIn("enumerate_divisions", all_source)
                self.assertIn("parent_union_overlap", all_source)
                self.assertIn("logistic_rich_mask_permuted", all_source)
                self.assertIn("forest_rich", all_source)
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
