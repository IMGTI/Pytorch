from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# GPU or CPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_filters_conv1 = 1024
num_filters_conv2 = int(num_filters_conv1/3) * 2

# Classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters_conv1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters_conv1, num_filters_conv2, 5)
        self.fc1 = nn.Linear(num_filters_conv2 * 5 * 5, 120)
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


def test_one_image(img, net):
    '''
    img - Image tensor
    '''

    # convert image to torch tensor and add batch dim
    batch = (img).clone().detach().unsqueeze(0)  # Not sure to use img or img/255

    # We don't need gradients for test, so wrap in
    # no_grad to save memory
    with torch.no_grad():
        batch = batch.to(device)

        # forward propagation
        output = net(batch)

        # get prediction
        _, output = torch.max(output, 1)

    return classes[output]

net = Net().to(device)

### Load state dict of model
try:
    net.load_state_dict(torch.load('state_dict'))
    net.eval()
    #print('State dict: ', net.state_dict())
except:
    print('No model state dict found')

### Test image
num_tests = 100
image_name = 'prueba_perro2.jpg'
image_resize = (32,32)
img = Image.open(image_name).convert('RGB')
img.show()
# Convert PIL image to numpy array
data = np.asarray(img)
# Obtain mean and std for each RGB channel, and convert them to
# range 0-1 (as same as .ToTensor does)
mean = data.mean(axis=(0,1))/255  # Mean in each RGB-channel
std = data.std(axis=(0,1))/255  # Std in each RGB-channel

Loader = transforms.Compose([transforms.Resize(image_resize, interpolation=Image.NEAREST),
                             transforms.ToTensor()])
# Normalize with image mean and std
#Normalize = transforms.Compose([transforms.Normalize(mean=mean,
#                                                     std=std)])

# Normalize with model mean and std
Normalize = transforms.Compose([transforms.Normalize(mean=(0.5,0.5,0.5),
                                                     std=(0.5,0.5,0.5))])

img = Normalize(Loader(img)).to(device)

show_img_tensor = transforms.Compose([transforms.ToPILImage()])(img.cpu()).show()
probs_dict = {}

try:
    for i in tqdm(range(num_tests), total=num_tests):
        result = test_one_image(img, net)
        if not result in probs_dict.keys():
            probs_dict[result] = 1
        else:
            probs_dict[result] += 1
    most_prob = list(probs_dict.keys())[np.argmax(probs_dict.values())]

    print('Resultado: ', most_prob, '\nProbability: ',
          probs_dict[most_prob] * 100 / num_tests)
except:
    print('ERROR')
