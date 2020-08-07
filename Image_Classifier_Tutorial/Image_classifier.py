import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

##### DATA MANAGEMENT #####

### Define hyperparameters

batch_size = 25#4
num_epochs = 20#200#2
learning_rate = 0.001#0.001
momentum = 0.9#0.9
initial_net_width = 512#6  # Precision increases drastically with this
two_thirds_of_inw = int(initial_net_width/3) * 2

### Define numworkers for ubuntu and windows
os_system = 'windows'#'ubuntu'

if os_system=='windows':
    numworkers = 0
else:
    numworkers = 2

### Transforming/normalizing torchvision output datasets

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=numworkers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=numworkers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


### Showing images for fun

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

### DEFINE A CNN

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, initial_net_width, 5)   # kernel size 5 <=> filter 5x5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(initial_net_width, two_thirds_of_inw, 5)  # net_width (#nodes) == 2/3 prev layer
        self.fc1 = nn.Linear(two_thirds_of_inw * 5 * 5, 120)  # Number of input features is 16*5*5 due to con and pool
                                               # applied to images of 3*32*32 (RGB of 32x32 pixels)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, two_thirds_of_inw * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

### Initiliaze Net
net = Net()

### Load state dict of model
try:
    net.load_state_dict(torch.load('state_dict'))
    net.eval()
except:
    print('No model state dict found')

### SEND NET TO GPU IF AVAILABLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

### DEFINE LOSS FUNCTION AND OPTIMIZER

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

### TRAIN THE NETWORK

print('Beginning trainning...')

for epoch in tqdm(range(num_epochs), total=num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' %
            #      (epoch + 1, i + 1, running_loss / 2000))

            # Store mean running loss of 2000 mini-batches
            final_mean_running_loss = running_loss / 2000

            running_loss = 0.0

# Print final statistics
print('Final Mean Running Loss = ', final_mean_running_loss)

print('Finished Training')

### TEST THE NETWORK

dataiter = iter(testloader)
dataiter_next = dataiter.next()
images, labels = dataiter_next[0].to(device), dataiter_next[1].to(device)

# print images
imshow(torchvision.utils.make_grid(images).cpu())
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

outputs = net(images).to(device)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(batch_size)))

### PERFORMANCE ON WHOLE DATASET

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

### Save state dict of model
torch.save(net.state_dict(), 'state_dict')
