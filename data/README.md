# Data

This directory gives detailed instructions on how to download and work with the SHDocs dataset as well as to access it's metadata data using `mlcroissant`

## Downloading SHDocs

1. SHDocs can be downloaded from the following links:
    - SHDocs raw data: [Microsoft OneDrive](https://hometeamsnt-my.sharepoint.com/:u:/g/personal/jovin_leong_hometeamsnt_onmicrosoft_com/EcJDKkwv5_BKk4T7E2-A0qsB06zOpMj-wywW7bz1v3wF-w?e=2u4cHQ) or [Google Drive](https://drive.google.com/file/d/1J5OekUKSCrcqfXLzCMkgae7nPLd-5WCN/view?usp=sharing)
    - SHDocs processed data: [Microsoft OneDrive](https://hometeamsnt-my.sharepoint.com/:u:/g/personal/jovin_leong_hometeamsnt_onmicrosoft_com/EY_m6N_VApVKhBOFZe-4rNYB_r0W1FExQT6WHDO77Rti0A?e=lLxmYM) or [Google Drive](https://drive.google.com/file/d/1zotvR5kAfYz5564C0QvuExIksmNtuPst/view?usp=sharing)
2. Once downloaded, move both the raw data and processed data into this directory and unzip them.
3. With that done, you can use the data directly or use the `croissant.ipynb` notebook to access the dataset [Croissant](https://mlcommons.org/working-groups/data/croissant/) metadata. Further instructions on working with Croissant are provided in the notebook and [here](https://github.com/mlcommons/croissant/tree/main/)

## Working with SHDocs

### Raw data

The raw data can be used directly as it consists solely of image captures in various reconstructions detailed in our paper. Of note are the following:

- `deglared`: `deglared` images are reconstructions made by using all four polarization angles captured by the polarization sensor without glare.
- `s0_norm`: `s0_norm` images are reconstructions made by normalizing the aggregate of `i0` and `i90` which is equivalent to a "normal" greyscale image captured by a camera. The `s0_norm` image and `deglared` images are perfectly aligned and can be thought of as counterfactuals which can be used in specular highlight removal and general image enhancement tasks.

### Processed data

The method of processing the data is described in our paper and the [pipeline](https://github.com/JovinLeong/SHDocs?tab=readme-ov-file#method). Essentially, we scale the image captures to align them to the FUNSD text annotations so as to enable OCR evaluation of the enhanced images.

Typically, models can be trained using the raw data and then the processed testing data can be passed through these trained models and evaluated.

Due to resource limitations, we currently only have the processed data for the testing set. This enables evaluation but limits the ability to perform text-aware model training; however, we will provide processed testing data as soon as possible.

## Known issues

- Due to resource limitations, we currently only have the processed data for the testing set. However, we will provide processed testing data as soon as possible.
- The following images in the dataset are known to have been corrupted:

    ```{bash}
    raw_captures/training_data/images/0060000813_a/i0.png
    ```
