
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



MACOS = platform.system() == "Darwin"  # macOS environment



def export_formats():
    r"""
    Returns a DataFrame of supported YOLOv5 model export formats and their properties.

    Returns:
        pandas.DataFrame: A DataFrame containing supported export formats and their properties. The DataFrame
        includes columns for format name, CLI argument suffix, file extension or directory name, and boolean flags
        indicating if the export format supports training and detection.

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        The DataFrame contains the following columns:
        - Format: The name of the model format (e.g., PyTorch, TorchScript, ONNX, etc.).
        - Include Argument: The argument to use with the export script to include this format.
        - File Suffix: File extension or directory name associated with the format.
        - Supports Training: Whether the format supports training.
        - Supports Detection: Whether the format supports detection.
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


