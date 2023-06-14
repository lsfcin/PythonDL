import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import device_module
import predictor
import trainer

available = torch.cuda.is_available()
device_count = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0)
current_device = torch.cuda.current_device()

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_data_loader = DataLoader(training_data, batch_size=batch_size)
test_data_loader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_data_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

print(f"Using {device_module.device} device")

FLAG_TRAIN = False

if(FLAG_TRAIN):
  model = trainer.train_epochs(5, train_data_loader, test_data_loader)
  print("Done!")

  torch.save(model.state_dict(), "models/model.pth")
  print("Saved PyTorch Model State to model.pth")

predictor.predict("models/model.pth", test_data)