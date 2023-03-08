from flask import Flask, request
import base64
from flask_cors import CORS

import torch
from ModelClass import MultilayerRNN_MNIST, transform_image
import pickle


batch_size = 64
input_size = 28
hidden_size = 100      # neurons
layer_size = 2         # layers
output_size = 3

model = MultilayerRNN_MNIST(input_size, hidden_size, layer_size, output_size, relu=False)
model.load_state_dict(torch.load("model.pt"))
model.eval()

labels = pickle.loads(open("torchlables.pickle","rb").read())

def getLabels(index):
    result = ""
    for name, i in labels.items():
        if i == index:
            result = name
    
    return result

app = Flask(__name__)
CORS(app)

@app.route("/")
def test():
    return "halo"


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    path = f"uploads/{image.filename}"
    image.save(path)

    image_tensor = transform_image(path)
    output = model.forward(image_tensor)
    outputValue, pred = output.max(1)

    outputsSum = 0
    outputValues = output.tolist()[0]

    for i in outputValues:
        if i > 0:
            outputsSum += i
    
    confidents = []
    for idx,val in enumerate(outputValues):
        confident = 0 if val < 0 else (val / outputsSum * 100) 
        confidents.append({
            "lable": getLabels(idx),
            "confident": float("{:.2f}".format(confident))
        })

    result = getLabels(pred.item())
    confidentValue = 0
    return {
        "prediction": {
            "lable": result,
            "confidents": confidents
        }
    }

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)