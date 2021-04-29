import torch
from model import DNCNN
from torchvision import datasets, transforms
import time
from thop import profile

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using {} device.".format(device))
# Test
test_dir = './FIRE-SMOKE-DATASET/FIRE-SMOKE-DATASET/Test'
test_transforms = transforms.Compose([
    transforms.Resize(size=(48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=96, shuffle=True)
#DNCNN
num_classes = 2
save_path = './trained-model-DNCNN.pth'
net = DNCNN(num_classes=num_classes)
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




