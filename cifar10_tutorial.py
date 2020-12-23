#1. torchvision 을 사용하여 CIFAR10의 학습용 / 시험용 데이터셋을 불러오고, 정규화(nomarlizing)합니다.
#2. 합성곱 신경망(Convolution Neural Network)을 정의합니다.
#3. 손실 함수를 정의합니다.
#4. 학습용 데이터를 사용하여 신경망을 학습합니다.
#5. 시험용 데이터를 사용하여 신경망을 검사합니다.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CUDA 기기가 존재한다면, 아래 코드가 CUDA 장치를 출력합니다:
print(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지를 보여주기 위한 함수
def imshow(img, labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    row, _, __, ___ = img.size()
    for i in range(len(npimg)):
        plt.subplot(1,row,i+1)
        plt.imshow(np.transpose(npimg[i], (1, 2, 0)))
        plt.title(classes[labels[i]])  # 제목 추가
        plt.axis("off")  # 축 제거
    plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
print(labels.size())

# 이미지 보여주기
imshow(images, labels)

# # 정답(label) 출력
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = Net()
net = Net().to(device)

n_epoch = 100
model_name = 'cifar10_classifier'
path = './'+model_name+'_'+'model.pt'

# set up the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)

for epoch in range(1, n_epoch + 1):
    print("epoch: [%d/%d]" % (epoch, n_epoch))

    net.train()

    for data, label in tqdm(trainloader):
        x = data.to(device)
        gt = label.to(device)

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(x)
        loss = criterion(outputs, gt)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), path)

    correct = 0
    total = 0
    count = 0

    net.eval()
    for data, label in tqdm(trainloader):
        x = data.to(device)
        gt = label.to(device)

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(x)
        loss = criterion(outputs, gt)

        predicted = torch.argmax(outputs, dim=1)
        total += label.size(0)
        correct += (predicted == gt).sum().item()

    print("LOSS: {}".format(loss.item()))
    print("Validating Accuracy of the model {} %".format(100 * correct / total))

    scheduler.step()
print('Finished Training')