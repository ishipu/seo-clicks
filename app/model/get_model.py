from tensorflow.keras.models import load_model
import os.path
from build_model.build_model import build

def get_model(default_model_name = 'clicks_predictor_model.h5'):
    if os.path.exists(default_model_name):
        print("Modle Found")
        savedModel=load_model(default_model_name)
        return savedModel

    else:
        print("No model found, running build_model...")
        build()
        return load_model("model/clicks_predictor_model.h5")
