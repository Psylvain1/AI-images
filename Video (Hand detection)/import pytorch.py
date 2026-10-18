import torch
import torchvision
import torchvision.transforms as transforms


import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((35, 35)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify() # Verify that the file is an image
        return True
    except (IOError, SyntaxError) as e:
        print(f"Bad image file: {path}, error: {e}")
        return False

# Custom loader that checks for valid images
def custom_loader(path):
    if is_valid_image(path):
        return torchvision.datasets.folder.default_loader(path)
    else:
        # Return a dummy tensor for invalid images to avoid crashing the loader
        # This dummy tensor should have the same shape as a valid image tensor after transform
        # Assuming the expected shape is (3, 32, 32) based on common image sizes and your network architecture
        print(f"Skipping invalid image: {path}")
        return torch.zeros(3, 32, 32)

""""
trainset = torchvision.datasets.ImageFolder(root='/home/ssppa/image learning/vf06bnjefr9bav63fepmz6/images.cv_vf06bnjefr9bav63fepmz6/data/train', transform=transform, loader=custom_loader)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='/home/ssppa/image learning/vf06bnjefr9bav63fepmz6/images.cv_vf06bnjefr9bav63fepmz6/data/test', transform=transform, loader=custom_loader)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
"""


# 📂 Load the full dataset
full_dataset = torchvision.datasets.ImageFolder(root='/home/ssppa/image_learning/archive (1)/dataset', transform=transform)

# 🔀 Define split sizes
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# ✂️ Split the dataset with generator for fix batch
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)

# 📦 Create DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

import os

folder_path = '/home/ssppa/image_learning/archive (1)/dataset'

# List only directories
classes = [name for name in os.listdir(folder_path)
           if os.path.isdir(os.path.join(folder_path, name))]
print(len(classes))


import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 516)
        self.fc2 = nn.Linear(516, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

net.load_state_dict(torch.load('/home/ssppa/image_learning/dog_recog.pth'))


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


PATH = '/home/ssppa/image_learning/dog_recog.pth'
torch.save(net.state_dict(), PATH)


dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
