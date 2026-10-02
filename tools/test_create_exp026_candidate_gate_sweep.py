"""Static and self-contained import tests for the EXP026 notebook generator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import create_exp026_candidate_gate_sweep as generator


class Exp026NotebookGeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_embedded_package_imports(self) -> None:
        original_out = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp026_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                for cell in code_cells:
                    compile("".join(cell["source"]), cell.get("id", "cell"), "exec")
                embedded_source = "".join(code_cells[0]["source"])
                self.assertIn("gate_sweep.py", embedded_source)
                local_source = embedded_source.replace("/kaggle/working", Path(directory).as_posix())
                script = Path(directory) / "execute_embedded_cell.py"
                script.write_text(local_source, encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(script)], cwd=directory,
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            generator.OUT = original_out

    def test_no_submission_or_official_cv_claim(self) -> None:
        source = generator.CANDIDATE_SWEEP + generator.OUTPUTS
        self.assertNotIn("submission.csv", source)
        self.assertIn("'official_cv': None", source)
        self.assertIn("'submission_generated': False", source)


if __name__ == "__main__":
    unittest.main()
