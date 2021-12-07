import torch
from torch import nn
from resnet import resnet18

class EMIL(nn.Module):
    def __init__(self, output_type, num_inp_channels, num_fmap_channels, att_dim, num_classes, patch_size, patch_stride, K_min):
        super().__init__()
        self.num_classes = num_classes
        self.K_min = K_min

        self.backbone = resnet18(num_channels=num_inp_channels, num_classes=num_classes)
        self.patch_extractor = nn.AvgPool2d(patch_size, patch_stride)
        
        shared_output_layer = nn.Linear(num_fmap_channels, num_classes)
        shared_output_layer_att = nn.Softmax(dim=-1) if output_type == 'multiclass' else nn.Sigmoid()
        self.shared_output_layer = nn.Sequential(    
            shared_output_layer,
            shared_output_layer_att
        )

        self.att_tanh = nn.Sequential(
            nn.Linear(num_fmap_channels, att_dim),
            nn.Tanh()
        )
        self.att_sigm = nn.Sequential(
            nn.Linear(num_fmap_channels, att_dim),
            nn.Sigmoid()
        )
        self.att_outer = nn.Sequential(
            nn.Linear(att_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, output_heatmaps = False):
        x = self.backbone(img)
        x = self.patch_extractor(x)
        
        b, c, _, _ = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        b, k, c = x.shape
        x = x.reshape(-1, c)

        x_local = self.shared_output_layer(x).view(b, k, self.num_classes)
        x_weight = self.att_outer(self.att_tanh(x) * self.att_sigm(x)).view(b, k, 1)
        pred = torch.sum(x_local * x_weight, dim = 1) / torch.clamp(torch.sum(x_weight, dim = 1), min = self.K_min)

        if output_heatmaps:
            return pred, x_local, x_weight
        else:
            return pred
