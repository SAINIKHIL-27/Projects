import numpy as np


def preprocess_diabetes(values):
    return np.array([
        values["pregnancies"],
        values["glucose"],
        values["blood_pressure"],
        values["skin_thickness"],
        values["insulin"],
        values["bmi"],
        values["diabetes_pedigree"],
        values["age"],
    ], dtype=np.float32)


def preprocess_heart(values):
    return np.array([
        values["age"],
        values["sex"],
        values["cp"],
        values["trestbps"],
        values["chol"],
        values["fbs"],
        values["restecg"],
        values["thalach"],
        values["exang"],
        values["oldpeak"],
        values["slope"],
        values["ca"],
        values["thal"],
    ], dtype=np.float32)


def preprocess_parkinsons(values):
    return np.array([
        values["fo"],
        values["fhi"],
        values["flo"],
        values["jitter_percent"],
        values["jitter_abs"],
        values["rap"],
        values["ppq"],
        values["ddp"],
        values["shimmer"],
        values["shimmer_db"],
        values["apq3"],
        values["apq5"],
        values["apq"],
        values["dda"],
        values["nhr"],
        values["hnr"],
        values["rpde"],
        values["dfa"],
        values["spread1"],
        values["spread2"],
        values["d2"],
        values["ppe"],
    ], dtype=np.float32)


def preprocess_lungs(values):
    return np.array([
        values["gender"],
        values["age"],
        values["smoking"],
        values["yellow_fingers"],
        values["anxiety"],
        values["peer_pressure"],
        values["chronic_disease"],
        values["fatigue"],
        values["allergy"],
        values["wheezing"],
        values["alcohol_consuming"],
        values["coughing"],
        values["shortness_of_breath"],
        values["swallowing_difficulty"],
        values["chest_pain"],
    ], dtype=np.float32)


def preprocess_thyroid(values):
    return np.array([
        values["age"],
        values["sex"],
        values["on_thyroxine"],
        values["tsh"],
        values["t3_measured"],
        values["t3"],
        values["tt4"],
    ], dtype=np.float32)
