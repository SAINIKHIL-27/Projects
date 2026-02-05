from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parent
SOURCE_MODELS_DIR = BASE_DIR.parent / "Models"
TARGET_MODELS_DIR = BASE_DIR / "models"

MODEL_CONFIG = {
    "diabetes": {
        "source": "diabetes_model.sav",
        "target": "diabetes.onnx",
        "features": 8,
    },
    "heart": {
        "source": "heart_disease_model.sav",
        "target": "heart.onnx",
        "features": 13,
    },
    "parkinsons": {
        "source": "parkinsons_model.sav",
        "target": "parkinsons.onnx",
        "features": 22,
    },
    "lungs": {
        "source": "lungs_disease_model.sav",
        "target": "lungs.onnx",
        "features": 15,
    },
    "thyroid": {
        "source": "Thyroid_model.sav",
        "target": "thyroid.onnx",
        "features": 7,
    },
}


def convert_models() -> None:
    TARGET_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, config in MODEL_CONFIG.items():
        source_path = SOURCE_MODELS_DIR / config["source"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing model file: {source_path}")

        model = joblib.load(source_path)
        initial_type = [("float_input", FloatTensorType([None, config["features"]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        target_path = TARGET_MODELS_DIR / config["target"]
        target_path.write_bytes(onnx_model.SerializeToString())
        print(f"Converted {name} -> {target_path}")


if __name__ == "__main__":
    convert_models()
