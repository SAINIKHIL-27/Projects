from inference import EdgeInferenceEngine


def ask_float(prompt: str) -> float:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def collect_diabetes():
    return {
        "pregnancies": ask_float("Number of pregnancies"),
        "glucose": ask_float("Glucose level"),
        "blood_pressure": ask_float("Blood pressure"),
        "skin_thickness": ask_float("Skin thickness"),
        "insulin": ask_float("Insulin level"),
        "bmi": ask_float("BMI"),
        "diabetes_pedigree": ask_float("Diabetes pedigree function"),
        "age": ask_float("Age"),
    }


def collect_heart():
    return {
        "age": ask_float("Age"),
        "sex": ask_float("Sex (1 = male, 0 = female)"),
        "cp": ask_float("Chest pain type (0-3)"),
        "trestbps": ask_float("Resting blood pressure"),
        "chol": ask_float("Serum cholesterol (mg/dl)"),
        "fbs": ask_float("Fasting blood sugar > 120 mg/dl (1/0)"),
        "restecg": ask_float("Resting ECG results (0-2)"),
        "thalach": ask_float("Max heart rate achieved"),
        "exang": ask_float("Exercise induced angina (1/0)"),
        "oldpeak": ask_float("ST depression induced by exercise"),
        "slope": ask_float("Slope of peak exercise ST segment (0-2)"),
        "ca": ask_float("Major vessels colored by fluoroscopy (0-3)"),
        "thal": ask_float("Thal (0 = normal, 1 = fixed defect, 2 = reversible defect)"),
    }


def collect_parkinsons():
    return {
        "fo": ask_float("MDVP:Fo(Hz)"),
        "fhi": ask_float("MDVP:Fhi(Hz)"),
        "flo": ask_float("MDVP:Flo(Hz)"),
        "jitter_percent": ask_float("MDVP:Jitter(%)"),
        "jitter_abs": ask_float("MDVP:Jitter(Abs)"),
        "rap": ask_float("MDVP:RAP"),
        "ppq": ask_float("MDVP:PPQ"),
        "ddp": ask_float("Jitter:DDP"),
        "shimmer": ask_float("MDVP:Shimmer"),
        "shimmer_db": ask_float("MDVP:Shimmer(dB)"),
        "apq3": ask_float("Shimmer:APQ3"),
        "apq5": ask_float("Shimmer:APQ5"),
        "apq": ask_float("MDVP:APQ"),
        "dda": ask_float("Shimmer:DDA"),
        "nhr": ask_float("NHR"),
        "hnr": ask_float("HNR"),
        "rpde": ask_float("RPDE"),
        "dfa": ask_float("DFA"),
        "spread1": ask_float("Spread1"),
        "spread2": ask_float("Spread2"),
        "d2": ask_float("D2"),
        "ppe": ask_float("PPE"),
    }


def collect_lungs():
    return {
        "gender": ask_float("Gender (1 = male, 0 = female)"),
        "age": ask_float("Age"),
        "smoking": ask_float("Smoking (1 = yes, 0 = no)"),
        "yellow_fingers": ask_float("Yellow fingers (1 = yes, 0 = no)"),
        "anxiety": ask_float("Anxiety (1 = yes, 0 = no)"),
        "peer_pressure": ask_float("Peer pressure (1 = yes, 0 = no)"),
        "chronic_disease": ask_float("Chronic disease (1 = yes, 0 = no)"),
        "fatigue": ask_float("Fatigue (1 = yes, 0 = no)"),
        "allergy": ask_float("Allergy (1 = yes, 0 = no)"),
        "wheezing": ask_float("Wheezing (1 = yes, 0 = no)"),
        "alcohol_consuming": ask_float("Alcohol consuming (1 = yes, 0 = no)"),
        "coughing": ask_float("Coughing (1 = yes, 0 = no)"),
        "shortness_of_breath": ask_float("Shortness of breath (1 = yes, 0 = no)"),
        "swallowing_difficulty": ask_float("Swallowing difficulty (1 = yes, 0 = no)"),
        "chest_pain": ask_float("Chest pain (1 = yes, 0 = no)"),
    }


def collect_thyroid():
    return {
        "age": ask_float("Age"),
        "sex": ask_float("Sex (1 = male, 0 = female)"),
        "on_thyroxine": ask_float("On thyroxine (1 = yes, 0 = no)"),
        "tsh": ask_float("TSH level"),
        "t3_measured": ask_float("T3 measured (1 = yes, 0 = no)"),
        "t3": ask_float("T3 level"),
        "tt4": ask_float("TT4 level"),
    }


def main():
    options = {
        "1": ("Diabetes", "diabetes", collect_diabetes),
        "2": ("Heart Disease", "heart", collect_heart),
        "3": ("Parkinson's", "parkinsons", collect_parkinsons),
        "4": ("Lung Cancer", "lungs", collect_lungs),
        "5": ("Hypo-Thyroid", "thyroid", collect_thyroid),
    }

    print("\nEdge Medical ML - Offline Inference")
    for key, (label, _, _) in options.items():
        print(f"{key}. {label}")

    choice = input("Select a model (1-5): ").strip()
    if choice not in options:
        print("Invalid option. Exiting.")
        return

    label, disease_key, collector = options[choice]
    print(f"\nEnter inputs for {label} prediction:\n")
    values = collector()

    engine = EdgeInferenceEngine(disease_key)
    result = engine.predict(values)

    if result == 1:
        print("\nResult: High Risk")
    else:
        print("\nResult: Low Risk")


if __name__ == "__main__":
    main()
