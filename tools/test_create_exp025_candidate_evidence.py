"""Tests for the self-contained EXP025 notebook generator."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import create_exp025_candidate_evidence as generator


class Exp025NotebookGeneratorTests(unittest.TestCase):
    def test_embedded_manifest_covers_package_initialiser_imports(self) -> None:
        generator.validate_embedded_modules()
        package_dir = generator.ROOT / "tools" / "focus_first"
        tree = ast.parse((package_dir / "__init__.py").read_text(encoding="utf-8"))
        required = {
            f"{node.module.split('.', 1)[0]}.py"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
        }
        self.assertTrue(required.issubset(set(generator.EMBEDDED_FOCUS_MODULES)))
        self.assertIn("runtime.py", generator.EMBEDDED_FOCUS_MODULES)

    def test_generated_notebook_is_valid_and_all_code_compiles(self) -> None:
        original_out = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp025_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                self.assertTrue(code_cells)
                for cell in code_cells:
                    compile("".join(cell["source"]), cell.get("id", "cell"), "exec")
                embedded_source = "".join(code_cells[0]["source"])
                self.assertIn("runtime.py", embedded_source)
                local_source = embedded_source.replace(
                    "/kaggle/working", Path(directory).as_posix()
                )
                script = Path(directory) / "execute_embedded_cell.py"
                script.write_text(local_source, encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            generator.OUT = original_out


if __name__ == "__main__":
    unittest.main()
