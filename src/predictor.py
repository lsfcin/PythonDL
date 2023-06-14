import torch
import dnn_module
import device_module

def predict(model_path, test_data) :
  model = dnn_module.NeuralNetwork().to(device_module.device)
  model.load_state_dict(torch.load(model_path))

  classes = [
      "T-shirt/top",
      "Trouser",
      "Pullover",
      "Dress",
      "Coat",
      "Sandal",
      "Shirt",
      "Sneaker",
      "Bag",
      "Ankle boot",
  ]

  model.eval()
  x, y = test_data[0][0], test_data[0][1]
  with torch.no_grad():
      x = x.to(device_module.device)
      pred = model(x)
      predicted, actual = classes[pred[0].argmax(0)], classes[y]
      print(f'Predicted: "{predicted}", Actual: "{actual}"')