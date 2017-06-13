import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(1, 1)
        self.conv3_64 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv64_64 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv64_128 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv128_128 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv128_256 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv256_256 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv256_512 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv512_512 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fcf = nn.Linear(512, 10)
        self.drop = nn.Dropout(p=0.5)
        self.re = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3_64(x)
        x = self.bn_64(x)
        x = self.re(x)
        x = self.conv64_64(x)
        x = self.bn_64(x)
        x = self.re(x)

        x = self.pool1(x)
        
        x = self.conv64_128(x)
        x = self.bn_128(x)
        x = self.re(x)
        x = self.conv128_128(x)
        x = self.bn_128(x)
        x = self.re(x)

        x = self.pool1(x)

        x = self.conv128_256(x)
        x = self.bn_256(x)
        x = self.re(x)
        x = self.conv256_256(x)
        x = self.bn_256(x)
        x = self.re(x)

        x = self.pool1(x)

        x = self.conv256_512(x)
        x = self.bn_512(x)
        x = self.re(x)
        x = self.conv512_512(x)
        x = self.bn_512(x)
        x = self.re(x)

        x = self.pool1(x)

        x = self.conv512_512(x)
        x = self.bn_512(x)
        x = self.re(x)
        x = self.conv512_512(x)
        x = self.bn_512(x)
        x = self.re(x)

        x = self.pool1(x)

        x = x.view(x.size(0), -1)
        
        x = self.drop(x)
        x = self.fc1(x)
        x = self.re(x)

        x = self.drop(x)
        x = self.fc1(x)
        x = self.re(x)
        
        x = self.fcf(x)
        return x

load = True
if load:
    checkpoint = torch.load('./net_300_epoch.data')
    net = checkpoint['net']
else:
    net = Net()
    net.cuda()

    criterion = nn.CrossEntropyLoss()

    MAX_EP = 300
    start_lr = 0.05

    for epoch in range(MAX_EP):  # loop over the dataset multiple times
    
        optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9, weight_decay=5e-4)
        if (epoch+1) % 30 == 0:
            start_Lr = start_lr/2

        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.cuda().size(0)
            correct += predicted.eq(labels.cuda().data).cpu().sum()

        print('%d Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch+1, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if (epoch+1)%10 == 0:
            state = {
                'net': net,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+str(epoch+1))

    print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))





