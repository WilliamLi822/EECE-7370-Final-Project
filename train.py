import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.optim as optim
from model import DNCNN
from torchvision import datasets, transforms

# hyperparameter
num_epoches = 100
num_classes = 2
learning_rate = 0.01
momentum = 0.9
batch_size = 96
learning_rate_decay = 0.01
#use gpu or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using {} device.".format(device))
class_name = ['Fire', 'Neutral', 'Smoke']
net = DNCNN(num_classes=num_classes)
net = net.to(device)
# Loss Function, Optimizer and learning rate decay
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=learning_rate_decay)
# load dataset
data_dir = './FIRE-SMOKE-DATASET/FIRE-SMOKE-DATASET'
train_dir = data_dir + '/Train'

validate_dir = data_dir + '/Validation'
train_transforms = transforms.Compose([
    transforms.Resize(size=(56, 56)),
    transforms.RandomResizedCrop(48),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
validate_transforms = transforms.Compose([
    transforms.Resize(size=(56, 56)),
    transforms.RandomResizedCrop(48),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
validate_dataset = datasets.ImageFolder(validate_dir,transform=validate_transforms)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

train_acc_list = []
train_loss_list = []
validate_acc_list = []
validate_loss_list = []
save_path = './trained-model-DNCNN.pth'
def train(num_epoches, train_dataloader, model, optimizer, loss_function, save_path):
    best_acc = 0
    time_start = time.time()
    #train
    for epoch in range(1, (num_epoches + 1)):
        train_loss = 0
        train_acc = 0
        validate_loss = 0
        validate_acc = 0
        model.train()
        for j, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            _, prediction = torch.max(output, 1)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
            train_acc = train_acc + torch.sum(prediction == labels)
        scheduler.step()
        #validation
        model.eval()
        with torch.no_grad():
            for j, (images, labels) in enumerate(validate_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                _, prediction = torch.max(output, 1)
                loss = loss_function(output, labels)
                validate_loss = validate_loss + loss.item()
                validate_acc = validate_acc + torch.sum(prediction == labels)
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_acc / len(train_dataloader.dataset)
        validate_loss = validate_loss / len(validate_dataloader.dataset)
        validate_acc = validate_acc/ len(validate_dataloader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        validate_acc_list.append(validate_acc)
        validate_loss_list.append(validate_loss)
        print('Epoch: {} \tTraining Acc: {:6f} \tTraining Loss: {:6f}'.format(epoch, train_acc, train_loss))
        if validate_acc > best_acc:
            best_acc = validate_acc
            torch.save(model.state_dict(), save_path)
    time_end = time.time()
    train_time = (time_end - time_start)/num_epoches
    print('Training Time for Each Epoch is ', train_time)
    print('finished training')


train(num_epoches, train_dataloader, net, optimizer, loss_function,save_path)
#training accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(train_acc_list, label="train_accuracy")
plt.title("Train Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.show()





