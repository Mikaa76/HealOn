import joblib
import pandas as pd
import os

# Path to trained models
base_dir = os.path.dirname(os.path.dirname(__file__))  
di_model = os.path.join(base_dir, "models", "diabetes_model.pkl")
hrt_model = os.path.join(base_dir, "models", "heartrisk_model.pkl")

# Load pre-trained model pipelines 
diab_model = joblib.load(di_model)
heart_model = joblib.load(hrt_model)

def predict_hs(info):
    """
    Function for health risk prediction

    Args:
        It takes dict of data submitted via form.

    Returns:
        returns risk score for diabetes and heart disease.
    """

    # converting dict to array dataframe for model
    data = pd.DataFrame([info])
    
    # Model inference
    diab_risk = diab_model.predict(data)   # diabetes risk score
    heart_risk = heart_model.predict(data) # heart risk score
    
    # Both models return arrays → extract first element + round for readability
    diab_risk = round(diab_risk[0], 2)
    heart_risk = round(heart_risk[0], 2)

    return diab_risk, heart_risk
