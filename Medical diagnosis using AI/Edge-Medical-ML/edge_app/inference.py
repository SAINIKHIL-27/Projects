from pathlib import Path

import numpy as np
import onnxruntime as ort

import preprocess

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "diabetes": "diabetes.onnx",
    "heart": "heart.onnx",
    "parkinsons": "parkinsons.onnx",
    "lungs": "lungs.onnx",
    "thyroid": "thyroid.onnx",
}

PREPROCESSORS = {
    "diabetes": preprocess.preprocess_diabetes,
    "heart": preprocess.preprocess_heart,
    "parkinsons": preprocess.preprocess_parkinsons,
    "lungs": preprocess.preprocess_lungs,
    "thyroid": preprocess.preprocess_thyroid,
}


class EdgeInferenceEngine:
    def __init__(self, disease: str):
        if disease not in MODEL_FILES:
            raise ValueError(f"Unknown disease: {disease}")

        model_path = MODELS_DIR / MODEL_FILES[disease]
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. Run convert_models.py first."
            )

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.disease = disease

    def predict(self, values) -> int:
        features = PREPROCESSORS[self.disease](values).reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: features})
        return _extract_label(outputs)


def _extract_label(outputs) -> int:
    if not outputs:
        raise ValueError("No outputs returned from the model.")

    label_output = outputs[0]
    if isinstance(label_output, list):
        label_output = np.array(label_output)

    if isinstance(label_output, np.ndarray):
        label_value = label_output.ravel()[0]
    else:
        label_value = label_output

    return int(label_value)
