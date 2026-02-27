# Physical Regularization Loss (PRL) for Image Segmentation

PyTorch implementation of the **Physical Regularization Loss (PRL)**, as described in the paper: *"Physical Regularization Loss: Integrating Physical Knowledge to Image Segmentation"*.

This loss function integrates physical constraints via anisotropic diffusion equations (Perona-Malik) into the deep learning optimization process to enhance edge preservation and model generalization.

The detailed paper can see: <u>https://doi.org/10.1007/s11263-026-02776-5</u>

---

## 📖 Overview

Standard data-driven segmentation models (like U-Net or DeepLab) often struggle with blurry boundaries or limited training data. **PRL** addresses this by:
- **Incorporating Physical Laws**: Modeling the segmentation process as a heat diffusion evolution.
- **Anisotropic Smoothing**: Applying high diffusion in homogeneous regions while stopping at object boundaries using a **Structure Tensor**.
- **Edge-Preserving Regularization**: Penalizing the model when its prediction's physical divergence deviates from the ground truth's geometric structure.

---

## 🛠️ If you want to cite this paper:

You can use this bib file:
---
@article{ding2026physical,

  title={Physical Regularization Loss: Integrating Physical Knowledge to Image Segmentation},
  
  author={Ding, Yan and Li, Shuang and Li, Huafeng and Qi, Guanqiu and Cong, Baisen and Gong, Yunpeng and Zhu, Zhiqin},
  
  journal={International Journal of Computer Vision},

  volume={134},
  
  number={3},
  
  pages={137},
  
  year={2026},
  
  publisher={Springer}
  
}
