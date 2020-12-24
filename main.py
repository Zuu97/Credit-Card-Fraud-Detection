from variables import*
from util import*

from inference import InferenceModel 
from model import FraudDetection

def run_inference(TFmodel, keras_model):
    if not os.path.exists(model_converter):
        TFmodel.TFconverter(keras_model)
    TFmodel.TFinterpreter()

if __name__ == "__main__":
    model = FraudDetection()
    model.run()

    X, Xtest, Y, Ytest = load_data()

    TFmodel = InferenceModel()
    keras_model = model.model
    run_inference(TFmodel, keras_model)
    print(TFmodel.InferenceOutput(Xtest[10]))