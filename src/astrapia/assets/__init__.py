__all__ = ["MODELS", "PATH_TO_ASSETS"]

import pathlib
from typing import Any

import yaml


PATH_TO_ASSETS = pathlib.Path(__file__).resolve().parent
MODELS = {
    "coreml-long": PATH_TO_ASSETS / "long.mlpackage",
    "coreml-short": PATH_TO_ASSETS / "short.mlpackage",
    "coreml-mesh": PATH_TO_ASSETS / "mesh.mlpackage",
    "onnxruntime-long": PATH_TO_ASSETS / "long.onnx",
    "onnxruntime-short": PATH_TO_ASSETS / "short.onnx",
    "onnxruntime-mesh": PATH_TO_ASSETS / "mesh.onnx",
}


def load_yaml(path_to_yaml: pathlib.Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path_to_yaml (pathlib.Path): Path to the YAML file to read.

    Returns:
        dict[str, Any]: Parsed YAML data.

    Raises:
        FileNotFoundError: If `path_to_yaml` does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(path_to_yaml) as f:
        data = yaml.safe_load(f)
    return data
