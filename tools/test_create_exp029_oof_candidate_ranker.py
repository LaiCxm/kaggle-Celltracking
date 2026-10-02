from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp029_oof_candidate_ranker as generator


class Exp029GeneratorTests(unittest.TestCase):
    def test_notebook_compiles(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp029_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        compile("".join(item["source"]), item.get("id", "cell"), "exec")
                sources = "\n".join(
                    "".join(item["source"])
                    for item in notebook["cells"] if item["cell_type"] == "code"
                )
                self.assertIn("TARGET_VIDEO_COUNT = 24", sources)
                self.assertIn("leave_one_group_out_splits", sources)
                self.assertIn("oof_logistic_3feat", sources)
                self.assertIn("'official_cv': None", sources)
                self.assertIn("'submission_generated': False", sources)
        finally:
            generator.OUT = original

    def test_no_submission_path_in_notebook(self) -> None:
        source = generator.RANKER + generator.OUTPUTS
        self.assertNotIn("submission.csv", source)


if __name__ == "__main__":
    unittest.main()
