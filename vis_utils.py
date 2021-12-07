import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

def get_heatmap(patch_scores, fmap_dims, patch_size, patch_stride, img_size):
    """ 
        Returns a heatmap of *img_size*
        *patch_scores* is either patch pred probabilities or attention weights
    """

    device = patch_scores.device
    num_patches_y = int((fmap_dims[1] - patch_size) / patch_stride + 1)

    b, k = patch_scores.shape
    heatmap = torch.zeros((b, 1, fmap_dims[0], fmap_dims[1])).to(device)
    heatmap_norm  = torch.zeros_like(heatmap)
    for i in range(b):
        for j in range(k):
            patch_score = patch_scores[i,j].squeeze()
            row_id = int(j // num_patches_y)
            col_id = int(j % num_patches_y)
            heatmap[i, 0, row_id : row_id + patch_size, col_id : col_id + patch_size] += patch_score
            heatmap_norm[i, 0, row_id : row_id + patch_size, col_id : col_id + patch_size] += torch.ones_like(patch_score)
    heatmap /= heatmap_norm
    heatmap = nn.functional.interpolate(heatmap, (img_size, img_size), mode = 'bilinear')
    return heatmap

def vis_multiple_imgs(images, heatmaps, fpath):
    """ 
        Saves multiple *images* with *heatmaps* overlays in *fpath*
    """

    b = images.shape[0]
    num_rows = int(b**0.5)
    num_cols = num_rows
    
    fig = plt.figure(figsize=(8,8), dpi=300)
    for i, (img, hmap) in enumerate(zip(images, heatmaps)):
        img = img.cpu().numpy()
        img = (img - np.min(img)) / np.ptp(img)
        img = np.transpose(img, (1, 2, 0))

        hmap = hmap.cpu().numpy()
        hmap = np.transpose(hmap, (1, 2, 0))

        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        im_img = ax.imshow(img, alpha = 1., interpolation = 'bilinear', cmap = 'gray')
        im_hmap = ax.imshow(hmap, alpha = 0.5, interpolation = 'bilinear', cmap = 'viridis', vmin = 0., vmax = 1.)            
    
        ax.tick_params(
            axis = 'both',       
            which = 'both',      
            bottom = False,      
            top = False,         
            left = False,
            labelbottom = False,
            labelleft = False
        ) 
        
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.savefig(fpath, bbox_inches = 'tight', pad_inches = 0, format = 'png', transparent = False)