import importlib.util
from pathlib import Path


TUTORIAL = Path(__file__).resolve().parents[3] / "tutorials" / "03-matrix-multiplication.py"


def test_tn_pipeline_load_modes():
    spec = importlib.util.spec_from_file_location("metax_matmul_tutorial", TUTORIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_tn_pipeline_load_modes()


if __name__ == "__main__":
    test_tn_pipeline_load_modes()
