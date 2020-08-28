import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

##### DATA MANAGEMENT #####

### Define hyperparameters

batch_size = 25#4
num_epochs = 4#20#2  # Aumentar numero de nodos (con 50 epochs hizo lo mismo)
learning_rate = 0.001#0.001
momentum = 0.9#0.9
num_filters_conv1 = 1024#512#6  # Precision increases drastically with this
num_filters_conv2 = int(num_filters_conv1/3) * 2

#two_thirds_of_input_number_nodes = int(input_number_nodes/3) * 2  # (#nodes) == 2/3 prev layer

### Define numworkers for ubuntu and windows
os_system = 'windows'#'ubuntu'

if os_system=='windows':
    numworkers = 0
else:
    numworkers = 2

### Transforming/normalizing torchvision output datasets

# Original transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Data augmentation  (20 epochs => 68%)
'''
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomRotation(degrees=45),
    #transforms.RandomCrop(24,24),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
'''

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
    fig = plt.figure(1)
    fig.clf()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.show()


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
        # conv2d(input_channels, output_channels, kernel_size)
        # input_channels in input layer is 3 (RGB) != (#nodes)
        # apparently (#nodes in hidden layers) == (#filters/kernels in hidden layers)
        self.conv1 = nn.Conv2d(3, num_filters_conv1, 5)   # kernel size 5 <=> filter 5x5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters_conv1, num_filters_conv2, 5)
        # linear(input_features, output_features)
        self.fc1 = nn.Linear(num_filters_conv2 * 5 * 5, 120)  # Number of input features
                                                              # is num_filters_conv2*5*5 due to con and pool
                                                              # applied to images of 3*32*32 (RGB of 32x32 pixels)
                                                              # 32*32 -- kernel 5*5 -> 28*28 -- maxpool 2*2 -> 14*14
                                                              # -- kernel 5x5 -> 10*10 -- maxpool 2*2 -> 5*5 pixel image
                                                              # and there are num_filters_conv2 number of 5*5
                                                              # tensors
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, num_filters_conv2 * 5 * 5)
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

### PRINT NUMBER OF TOTAL PARAMETERS AND TOTAL LEARNABLE PARAMETERS
pytorch_total_params = sum(p.numel() for p in net.parameters())
pytorch_learn_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("Number of Learnable parameters: ", pytorch_learn_params,
      "Number of Total parameters: ", pytorch_total_params)

### DEFINE LOSS FUNCTION AND OPTIMIZER

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

### TRAIN THE NETWORK

print('Beginning trainning...')

# For plotting loss vs epoch
fig_mean_loss = plt.figure(2)
loss4plot = []

for ind_epoch, epoch in tqdm(enumerate(range(num_epochs)), total=num_epochs):  # loop over the dataset multiple times

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
        #if i % 2000 == 1999:    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' %
            #      (epoch + 1, i + 1, running_loss / 2000))

            # Store mean running loss of 2000 mini-batches
        #    mean_running_loss = running_loss / 2000

        #    running_loss = 0.0

    # Store mean running loss of # mini-batches

    mean_running_loss = running_loss / len(trainloader)

    loss4plot.append(mean_running_loss)

    ### Save state dict of model
    torch.save(net.state_dict(), 'state_dict')

    fig_mean_loss.clf()
    plt.plot(range(num_epochs)[0:ind_epoch+1], loss4plot, 'g-')
    plt.title("Mean running loss vs epoch")
    plt.xlabel("Epoch (units)")
    plt.ylabel("Running loss")
    fig_mean_loss.savefig("loss_vs_epoch.jpg")

final_mean_running_loss = mean_running_loss

# Print final statistics
try:
    print('Final Mean Running Loss = ', final_mean_running_loss)
except:
    print('Too low number of mini-batches to show Final Mean Running Loss')

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
