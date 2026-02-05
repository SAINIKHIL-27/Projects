# Edge Medical ML (Offline)

This project is a beginner-friendly **Edge ML** demo that runs **fully offline** on a CPU-only device (like a Raspberry Pi or a laptop). It converts existing scikit-learn `.sav` models into **ONNX** format and runs **lightweight CLI inference** without any web UI.

## What is Edge ML?
Edge ML means running machine learning **directly on local devices** instead of sending data to a cloud server. This keeps inference **offline**, **private**, and **low-latency**.

## Streamlit ML vs Edge ML (Simple Comparison)
| Streamlit ML (Original) | Edge ML (This Project) |
| --- | --- |
| Web UI in browser | Terminal/CLI interface |
| Requires Streamlit server | Runs offline, no server |
| Pickle model files | ONNX model files |
| Cloud-like workflow | Edge device workflow |

## Project Structure
```
Edge-Medical-ML/
│
├── models/
│   ├── diabetes.onnx
│   ├── heart.onnx
│   ├── parkinsons.onnx
│   ├── lungs.onnx
│   └── thyroid.onnx
│
├── edge_app/
│   ├── inference.py
│   ├── preprocess.py
│   └── main.py
│
├── convert_models.py
├── requirements.txt
└── README.md
```

## How It Works (Offline Inference Flow)
```
User Input (CLI)
      ↓
Preprocess Features
      ↓
ONNX Runtime (CPU)
      ↓
Prediction (0 or 1)
      ↓
Low Risk / High Risk
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Convert the `.sav` models to ONNX (run once):
   ```bash
   python convert_models.py
   ```

3. Run the Edge app:
   ```bash
   cd edge_app
   python main.py
   ```

## Notes for Edge Devices
- Works on Raspberry Pi or offline laptops.
- CPU only (no GPU needed).
- No internet required for inference.

## Medical Disclaimer
This project is for **educational/demo purposes only** and is **not medical advice**. Predictions are based on pre-trained models and should not be used for real medical diagnosis or treatment decisions.
