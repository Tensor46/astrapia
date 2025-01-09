__all__ = ["MODELS", "PATH_TO_ASSETS"]

import pathlib
from typing import Any

import yaml


PATH_TO_ASSETS = pathlib.Path(__file__).resolve().parent
MODELS = {
    "coreml-long": PATH_TO_ASSETS / "long.mlpackage",
    "coreml-short": PATH_TO_ASSETS / "short.mlpackage",
    "onnxruntime-long": PATH_TO_ASSETS / "long.onnx",
    "onnxruntime-short": PATH_TO_ASSETS / "short.onnx",
}


def load_yaml(path_to_yaml: pathlib.Path) -> dict[str, Any]:
    with open(path_to_yaml) as f:
        data = yaml.safe_load(f)
    return data
