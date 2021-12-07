import os
import sys
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from emil import EMIL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 32
b = 256
num_epochs = 10
fmap_dims = (16, 16)
patch_size = 1
patch_stride = 1
output_type = 'multiclass'
model_save_path = 'models/mnist_emil.pt'

net = EMIL(
    output_type = output_type,
    num_inp_channels = 1,
    num_fmap_channels = 128,
    att_dim = 128,
    num_classes = 10,
    patch_size = patch_size,
    patch_stride = patch_stride,
    K_min = 100
).to(device)

# data prep
img_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = img_transform,
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = img_transform
)
train_loader = DataLoader(
    train_data,
    batch_size=b, 
    shuffle=True, 
    num_workers=1
)
test_loader = DataLoader(
    test_data,
    batch_size=b, 
    shuffle=False, 
    num_workers=1
)

loss_fn = nn.NLLLoss() if output_type == 'multiclass' else nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters())

# begin training
for epoch in range(num_epochs):
    net.train()
    for data in train_loader:
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()        
        pred = net(images, output_heatmaps=False)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
    net.eval()
    correct, total = 0, 0
    # evaluate
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            pred = net(images, output_heatmaps=False)
            pred_class = torch.max(pred, 1)[1] 
            correct += (pred_class == labels).sum()
            total += labels.shape[0]
        print("Epoch: {}, Accuracy: {}".format(epoch+1, correct / total))
    
# save model
torch.save({
    'state_dict': net.state_dict(),
    'optimizer' : optimizer.state_dict(),
}, model_save_path)