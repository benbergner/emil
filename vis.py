import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from emil import EMIL
from vis_utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 32
output_type = 'multiclass'
model_load_path = 'models/mnist_emil.pt'
patch_pred_map_path = 'imgs/patch_pred_map.png'
patch_weight_map_path = 'imgs/patch_weight_map.png'

fmap_dims = (16, 16)
patch_size = 1
patch_stride = 1
num_classes = 10
num_images = 64
num_patches = int((fmap_dims[0] - patch_size) / patch_stride + 1) ** 2

net = EMIL(
    output_type = output_type,
    num_inp_channels = 1,
    num_fmap_channels = 128,
    att_dim = 128,
    num_classes = num_classes,
    patch_size = patch_size,
    patch_stride = patch_stride,
    K_min = 100
).to(device)

#data prep
img_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = img_transform
)
test_loader = DataLoader(
    test_data,
    batch_size=num_images, 
    shuffle=True, 
    num_workers=1
)

checkpoint = torch.load(model_load_path)
net.load_state_dict(checkpoint['state_dict'], strict=False)
net.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader):
        images, labels = data[0].to(device), data[1].to(device)
        # load random images
        if i == 0:
            break
    
    pred, x_local, x_weight = net(images, output_heatmaps=True)
    pred_class = torch.max(pred, 1)[1]

    idx = torch.repeat_interleave(pred_class, num_patches)    
    x_local_pred_class = x_local.view(-1, num_classes)[torch.arange(num_images * num_patches), idx].view(num_images, num_patches)

heatmap_local = get_heatmap(x_local_pred_class, fmap_dims, patch_size, patch_stride, img_size)
heatmap_weight = get_heatmap(x_weight[:, :, 0], fmap_dims, patch_size, patch_stride, img_size)

vis_multiple_imgs(images, heatmap_local, fpath=patch_pred_map_path)
vis_multiple_imgs(images, heatmap_weight, fpath=patch_weight_map_path)