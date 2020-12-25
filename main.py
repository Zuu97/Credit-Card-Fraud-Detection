from variables import*
from util import*

from inference import InferenceModel 
from model import FraudDetection

TFmodel = InferenceModel()

def run_inference(TFmodel):
    keras_model = FraudDetection()
    keras_model.run()

    if not os.path.exists(model_converter):
        TFmodel.TFconverter(keras_model.model)
    TFmodel.TFinterpreter()

def obtain_output(transaction_data):
    output = TFmodel.InferenceOutput(transaction_data)
    output = output.squeeze()  > 0.5
    return int(output)


if __name__ == "__main__":
    run_inference(TFmodel)
    
    transaction_data = [-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919]
    transaction_data = np.array(transaction_data)
    output = obtain_output(transaction_data)
    output_str = class_dict[output]

    print("transaction data : {}".format(transaction_data))
    print("transaction is {}".format(output_str))
