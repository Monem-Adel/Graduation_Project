from matplotlib import pyplot as plt
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
from PIL import Image
torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

class_names = ['Final_State' , 'Start_State' , 'State', 'Transition' ]
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the images to the desired size
    transforms.ToTensor(),
])

# Load the dataset using ImageFolder 
# don't forget to put the path of the dataset  folder!!!!!!!!!!!!!!!!! 
dataset = ImageFolder(root=r"D:\4th\4TH First Sem\Graduation Project\TheDatatset", transform=data_transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) 

# neural network 
class Net(nn.Module):
 def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(8 * 8 * 8, 32)
    self.fc2 = nn.Linear(32, 4)
 def forward(self, x):
    out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
    out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
    out = out.view(-1, 8 * 8 * 8)
    out = torch.tanh(self.fc1(out))
    out = self.fc2(out)
    return out
 
 import datetime 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           shuffle=True) 
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                         shuffle=False)
def training_loop (n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range (1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            torch.tensor(labels, dtype=torch.int8) #tensor(1, dtype=torch.int8)
            #got an error msg the labels must be tensor not int
            #labels_tensor = torch.tensor(labels, dtype=torch.int8)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train+= loss.item()
        if epoch==1 or epoch % 10 ==0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train/len(train_dataset)))


model = Net()
optimizer = optim.SGD(model.parameters(), lr= 1e-2)
loss_fn = nn.CrossEntropyLoss()
"""training_loop(
    n_epochs= 100,
    optimizer = optimizer,
    model= model,
    loss_fn = loss_fn,
    train_loader= train_loader,
)"""


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           shuffle=False)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                         shuffle=False)

def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <2>
                total += labels.shape[0]  # <3>
                correct += int((predicted == labels).sum())  # <4>

        print("Accuracy {}: {:.2f}".format(name , correct / total))

#validate(model, train_loader, val_loader)
training_loop(
    n_epochs= 50,
    optimizer = optimizer,
    model= model,
    loss_fn = loss_fn,
    train_loader= train_loader,
)

validate(model, train_loader, val_loader)

def classify_image(image_path, model, class_names):
    image = Image.open(image_path)
    image = data_transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    return class_names[class_index]

# Example usage:
image_path = "D:\\4th\\4TH First Sem\\Graduation Project\\Test\\4\\TestImage.jpg"
predicted_class = classify_image(image_path, model, class_names)
print("Predicted class:", predicted_class)
