# Gaussian Random Fields as an Abstract Representation of Patient Metadata for Multimodal Medical Image Segmentation

This repository contains source code for the following paper (please cite if using code or methods from this repo):

```BibTex
@article{cassidy2025grf,
  author  = {Bill Cassidy and Connah Kendrick and Neil D. Reeves and Joseph M. Pappachan and Shaghayegh Raad and Moi Hoon Yap},
  year    = {2025},
  month   = {05},
  pages   = {},
  title   = {Gaussian random fields as an abstract representation of patient metadata for multimodal medical image segmentation},
  journal = {Scientific Reports},
  doi     = {10.1038/s41598-025-03393-x}
}
```

Before training the model, create the conda environment as follows:

    conda env create -f environment.yml

The dataset should be organised using the following directory structure:

    dataset
    └─ grf
    ├─ train
    |   └─ images
    |   └─ masks
    └─ test
        └─ images
        └─ masks

Generate the GRF images (based on your metadata) using:

    python generate_grf_images.py

You can then train the model using:

    python train.py --rect --augmentation

After the model has been trained, the trained weights will be saved to the ``weights/exp`` directory.

Test the model using:

    python test.py --rect --tta vh

Prediction masks are saved to the ``pred_mask`` directory.

Optionally, if you are training several different models with different i values and want to merge prediction masks, use:

    python merge_avg_masks.py

Convert the prediction masks to 8-bit:

    python convert_masks.py

Test metrics can then be generated using:

    python get_metrics.py

Test metrics are saved to the ``metrics`` directory.

For more information on the base segmentation model (HarDNet-CWS), please refer to the following [repository](https://github.com/mmu-dermatology-research/hardnet-cws).
