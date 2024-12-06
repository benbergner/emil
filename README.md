<img src="images/emil.png" width="800" />

# EMIL

Implementation of the EMIL architecture. Illustrated with MNIST and a simplified ResNet backbone.

## Usage
```python
import torch
from emil import EMIL

net = EMIL(
  output_type = 'multiclass',
  num_inp_channels = 1,
  num_fmap_channels = 128,
  att_dim = 128,
  num_classes = 10,
  patch_size = 1,
  patch_stride = 1,
  k_min = 100
)

img = torch.randn(1, 1, 32, 32)

pred, pred_local, pred_weight = net(img, output_heatmaps=True) # (1, 10), (1, k, 10), (1, k, 1)
```

`pred_local` holds local patch predictions, `pred_weight` holds attention weights.

## Parameters
`output_type`: string.<br />
Either `multiclass` or `binary`

`num_inp_channels`: int.<br />
Number of input channels

`num_fmap_channels`: int.<br />
Number of channels of the last conv layer

`att_dim`: int.<br />
Number of hidden dimensions in gated attention

`num_classes`: int.<br />
Number of classes

`patch_size`: int.<br />
Patch size in embedding space

`patch_stride`: int.<br />
Patch stride in embedding space

`k_min`: int.<br />
Minimum number of patches to achieve full class score. Default value is based on a maximum number of 16x16=256 patches.
For classical Multiple Instance Learning problems, such as the detection of dental caries, `k_min = 1` is recommended.

## Visualization
Both `pred_local` and `pred_weight` can be visualized as heatmaps. We show an example for MNIST and k_min=100.

<img src="images/mnist_pred_weight.png" width="350" />

## Citation
```bibtex
@InProceedings{pmlr-v172-bergner22a,
  title = 	 {Interpretable and Interactive Deep Multiple Instance Learning for Dental Caries Classification in Bitewing X-rays},
  author =       {Bergner, Benjamin and Rohrer, Csaba and Taleb, Aiham and Duchrau, Martha and De Leon, Guilherme and Rodrigues, Jonas and Schwendicke, Falk and Krois, Joachim and Lippert, Christoph},
  booktitle = 	 {Proceedings of The 5th International Conference on Medical Imaging with Deep Learning},
  pages = 	 {130--149},
  year = 	 {2022},
  editor = 	 {Konukoglu, Ender and Menze, Bjoern and Venkataraman, Archana and Baumgartner, Christian and Dou, Qi and Albarqouni, Shadi},
  volume = 	 {172},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--08 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v172/bergner22a/bergner22a.pdf},
  url = 	 {https://proceedings.mlr.press/v172/bergner22a.html},
  abstract = 	 {We propose a simple and efficient image classification architecture based on deep multiple instance learning, and apply it to the challenging task of caries detection in dental radiographs. Technically, our approach contributes in two ways: First, it outputs a heatmap of local patch classification probabilities despite being trained with weak image-level labels. Second, it is amenable to learning from segmentation labels to guide training. In contrast to existing methods, the human user can faithfully interpret predictions and interact with the model to decide which regions to attend to. Experiments are conducted on a large clinical dataset of 38k bitewings (316k teeth), where we achieve competitive performance compared to various baselines. When guided by an external caries segmentation model, a significant improvement in classification and localization performance is observed.}
}
```
