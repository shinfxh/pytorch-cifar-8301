'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
# from utils import progress_bar

import logging

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18(a=False)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

log_filename = "./logs/resnet-18-vanilla-seed0.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    logging.info(f'Epoch {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_acc = 100.*correct/total

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), train_acc, correct, total))
    print(f'Train Loss: {train_loss/(batch_idx + 1)}, Train accuracy: {train_acc}')
    logging.info(f'Train Loss: {train_loss/(batch_idx + 1)}, Train accuracy: {train_acc}')
    
    return train_loss, train_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    print(f'Test Loss: {test_loss/(batch_idx + 1)}, Test accuracy: {acc}')
    logging.info(f'Test Loss: {test_loss/(batch_idx + 1)}, Test accuracy: {acc}')

    return test_loss, acc

total_epochs = 300
train_loss_arr = []
train_acc_arr = []
test_loss_arr = []
test_acc_arr = []
a_vals = [[] for i in range(4)]
stages = [2, 2, 2, 2]
for i, stage in enumerate(stages):
    a_vals[i] = [[1] for j in range(stage)]

for epoch in range(start_epoch, start_epoch+total_epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    train_loss_arr.append(round(train_loss, 2))
    train_acc_arr.append(train_acc)
    test_loss_arr.append(round(test_loss, 2))
    test_acc_arr.append(test_acc)
    scheduler.step()
    
    def layer_operation(layer_num):
        layer = getattr(net.module, f'layer{layer_num + 1}')
        for i in range(stages[layer_num]):
            block = layer[i]
            a_val = block.a.cpu().detach().numpy().item()
            print(f'Stage {layer_num + 1} Block {i + 1}:', a_val)
            a_vals[layer_num][i].append(a_val)
    
    for j in range(4):
        layer_operation(j)


print(train_loss_arr)
print(train_acc_arr)
print(test_loss_arr)
print(test_acc_arr)
print(a_vals)

logging.info("Training Loss Array: " + str(train_loss_arr))
logging.info("Training Accuracy Array: " + str(train_acc_arr))
logging.info("Testing Loss Array: " + str(test_loss_arr))
logging.info("Testing Accuracy Array: " + str(test_acc_arr))
final_a = [[x[-1] for x in stage] for stage in a_vals]
logging.info("Final vals for a: " + str(final_a))
