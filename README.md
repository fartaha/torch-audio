# torch-audio
Urban sound classification project
* multicalss classification problem
* UrbanSound8k dataset
* 10 sound clasess

++ Audio features: 

++ Spectograms

++ Mel Spectograms

## Step1
installing torch, torch audio, torchvision
```console
pip install torch torchaudio torchvision
```

## Step2
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 

# ToTensor:
# It takes image in and reshapes a new tensor where each value
# is normalized between zero and one

# MNIST():
# This is a concrete implementation of a dataset class that comes
# with PyTorch

# DataLoader:
# is a class that we can use to wrap a dataset in our case the train_data
# and it will allow us to fetch data to load data in batches
# dataloader is an issuable object and so when we iterate through it we load
# data, we load data in batches (memory efficiency)

# Build models:
# 1- the models we create are in classes which inherit from the module class (nn.Module) that
# comes directly from PyTorch 
# 2- we need a couple of methods:
#       * all the constructor
#           # Sequential comes with PyTorch that allows us to pack multiple layers together
#           the data will flow sequentially from one layer to the next layer
#       * forward


# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__() # invoke the constructor of the base class (nn.Module)
        # defining the layers as attributes
        self.flatten = nn.Flatten() # nn.Flatten() is object from PyTorch convert MNIST imgaes to 1-D array
        self.dense_layers = nn.Sequential( 
            nn.Linear(28*28, 256), # dense layer
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1) # kind of a normalization to get the probability

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor() # transform the data to ToTensor object                 
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor() # transform the data to ToTensor object                 
    )
    return train_data, validation_data

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss at each batch
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step() # update the weights
    # printing the loss for the last batch we have
    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-------------------------------")
    print("Training is done.")



# Let's see this code works!?
if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets() # _: since we are not interested in validation_set here
    print("MNIST dataset downloaded")

    # create data loader for the train set
    trian_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device ="cuda"
    else: 
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device) # first define the FeedForwardNet class and then assign it to a device

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train the model
    train(feed_forward_net, trian_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")
```
## Step3
```python
MNIST dataset downloaded
Using cpu device
Epoch 1
Loss: 1.512393832206726
-------------------------------
Epoch 2
Loss: 1.4952977895736694
-------------------------------
Epoch 3
Loss: 1.491615891456604
-------------------------------
Epoch 4
Loss: 1.4797844886779785
-------------------------------
Epoch 5
Loss: 1.4772511720657349
-------------------------------
Epoch 6
Loss: 1.4759060144424438
-------------------------------
Epoch 7
Loss: 1.474435806274414
-------------------------------
Epoch 8
Loss: 1.4737602472305298
-------------------------------
Epoch 9
Loss: 1.4729491472244263
-------------------------------
Epoch 10
Loss: 1.4729899168014526
-------------------------------
Training is done.
Model trained and stored at feedforwardnet.pth
<<<<<<< HEAD

# Step3:
Inference:
```python
import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def predict(model, input, target, class_mapping):
    model.eval() # switch for evaluation 
    # model.train()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [[0.1, 0.01, ..., 0.6]]
        predicted_index =predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

# script
if __name__=="__main__":
    # load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a sample from validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    # make inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
```
```python
Predicted: '7', expected: '7'
```
=======
```
>>>>>>> 59f47171ea7ca05f80c73d82921f7cfb4a3c4957
