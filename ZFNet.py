from torchvision import datasets
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import time
import matplotlib.pyplot as plt

batch_size = 96
num_workers = 0
n_epoches = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using {} device.".format(device))

train_transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
validate_transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './FIRE-SMOKE-DATASET/FIRE-SMOKE-DATASET'
train_dir = data_dir + '/Train'
validate_dir = data_dir + '/Validation'

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
validate_data = datasets.ImageFolder(train_dir, transform=validate_transform)
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class ZFNet(nn.Module):

    def __init__(self, num_classes=2):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = ZFNet(num_classes = 2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

train_accuracy_list = []
train_loss_list = []

def train(epochs, train_loader, model, optimizer, criterion, save_path):
    best_acc = 0
    time_start = time.time()
    for epoch in range(epochs+1):
        train_loss = 0.0
        train_acc = 0.0
        validate_loss = 0.0
        validate_acc = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_acc = train_acc + torch.sum(preds == target.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                loss = criterion(output, target)
                validate_acc = validate_acc + torch.sum(preds == target.data)
                validate_loss = validate_loss + loss.item()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        validate_loss = validate_loss / len(validate_loader.dataset)
        validate_acc = validate_acc / len(validate_loader.dataset)
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        print('Epoch: {} \tTraining Acc: {:6f} \tTraining Loss: {:6f}'.format(epoch, train_acc, train_loss))
        if validate_acc > best_acc:
            best_acc = validate_acc
            torch.save(model.state_dict(), save_path)
    time_end = time.time()
    train_time = (time_end - time_start)/epochs
    print('Training Time for Each Epoch is ', train_time)
    print('finished training')

train(n_epoches, train_loader, model, optimizer, criterion,'./trained-model-ZFNet.pth')

plt.style.use("ggplot")
plt.figure()
plt.plot(train_accuracy_list, label="train_acc")
plt.title("ZFNet Train Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
#Test
test_dir = data_dir + '/Test'
test_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=96, shuffle=True)
num_classes = 2
save_path = './trained-model-ZFNet.pth'
net = ZFNet(num_classes=num_classes)
net = net.to(device)
net.load_state_dict(torch.load(save_path))
test_acc = 0
net.eval()
start_time = time.time()
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        _, prediction = torch.max(output.data, 1)
        test_acc = test_acc + torch.sum(prediction == labels)
test_acc = test_acc / len(test_dataloader.dataset)
end_time = time.time()
print('test time', end_time - start_time)
print('Test Accuracy is ', test_acc)