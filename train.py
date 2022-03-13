from torch.utils import model_zoo
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
import time

from resnet.model import ResNet, BasicBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model


res = resnet34(False)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                            torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform,
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print(f"测试数据集的长度为：{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)


res = res.cuda()

loss = nn.CrossEntropyLoss()
loss = loss.cuda()

lr = 0.01
optimizer = torch.optim.SGD(params=res.parameters(), lr=lr)

total_train_step = 0
total_test_step = 0
epoch = 10;

writer = SummaryWriter("../logs_train")
start_time = time.time()

for i in range(epoch):
    print(f'----------第{i + 1}轮训练开始----------')
    res.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = res(imgs)
        los = loss(outputs, targets)

        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f'训练次数:{total_train_step}, loss:{los.item()}')
            writer.add_scalar("train_loss", los.item(), total_train_step)
    res.eval()
    total_ac = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, target = data
            imgs = imgs.cuda()
            target = target.cuda()

            outputs = res(imgs)
            los = loss(outputs, target)
            total_test_loss += los.item()
            ac = (outputs.argmax(1) == target).sum()
            total_ac += ac

    print(f'整体测试集上的Loss：{total_test_loss}')
    print(f'全体测试集上的正确率{total_ac / test_data_size}')
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_ac / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(res, f'res_{i}.pth')
    print("模型已保存")
writer.close()
