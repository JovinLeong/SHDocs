# Model

This directory contains the details to retrain [MIMO-UNetPlus](https://github.com/chosj95/MIMO-UNet) as specified in the paper and use the MIMO-UNetPlus model to enhance images. Base code are attributed to the original authors of MIMO-UNetPlus.

Our model, a MIMO-UNetPlus retrained on SHDocs, that we reference in our paper can be downloaded in the following links

- [Microsoft OneDrive](https://hometeamsnt-my.sharepoint.com/:u:/g/personal/jovin_leong_hometeamsnt_onmicrosoft_com/EQNbX5o3r_tBg19zlIO2mlYB4iLxUTHKJmB2sm4s52_gMQ)
- [Google Drive](https://drive.google.com/file/d/1KYpwIinRQhET2BDsx5F18pIBzGwnv0yJ/view?usp=sharing)

Below are the results of our generalizability study; more details and analysis can be found in our paper.

| Model         | Dataset model was trained on |   Size  | SHIQ PSNR | SHIQ SSIM | PSD PSNR | PSD SSIM |
|---------------|------------------------------|:-------:|:---------:|:---------:|:--------:|:--------:|
| TSHRNet       | SSHR, SHIQ, PSD              |  468 MB |    25.6   |   0.933   |   22.8   |   0.903  |
| Hu et al.     | SHIQ                         |  412 MB |    33.9   |   0.980   |   27.5   |   0.970  |
| MIMO-UNetPlus | SHDocs                       | 64.6 MB |    22.4   |   0.915   |   28.0   |   0.956  |
| MIMO-UNetPlus | GoPro                        | 64.6 MB |    23.1   |   0.903   |   18.1   |   0.705  |

## Prerequisites

Next, install all the prerequisites and dependencies as specified in the main README dependencies.

## Train

As described in the paper, the model we used was [MIMO-UNetPlus](https://github.com/chosj95/MIMO-UNet). Instructions, initial weights, and the code to retrain MIMO-UNetPlus on SHDocs can be found [here](https://github.com/chosj95/MIMO-UNet?tab=readme-ov-file#train).

The weights of MIMO-UNetPlus were trained on SHDocs with randomly initialized weights, batch size of 4, a learning rate of 0.0001, and a gamma of 0.5 for 3000 epochs with early stopping based on the validation PSNR. [The weights used in the paper can be found here](https://hometeamsnt-my.sharepoint.com/:u:/g/personal/jovin_leong_hometeamsnt_onmicrosoft_com/EQNbX5o3r_tBg19zlIO2mlYB4iLxUTHKJmB2sm4s52_gMQ?e=9cxj4h).

The training samples used for the validation set are listed in [```validation_ids.txt```](https://github.com/JovinLeong/SHDocs/blob/main/model/validation_ids.txt)

Our model that is referenced in our paper, a MIMO-UNetPlus model retrained on SHDocs, can be downloaded through the following links:

- [Microsoft OneDrive](https://hometeamsnt-my.sharepoint.com/:u:/g/personal/jovin_leong_hometeamsnt_onmicrosoft_com/EQNbX5o3r_tBg19zlIO2mlYB4iLxUTHKJmB2sm4s52_gMQ)
- [Google Drive](https://drive.google.com/file/d/1KYpwIinRQhET2BDsx5F18pIBzGwnv0yJ/view?usp=sharing)


## Enhance

To enhance images using the MIMO-UNet model retrained on SHDocs, you can use [MIMO-UNet's testing function](https://github.com/chosj95/MIMO-UNet?tab=readme-ov-file#test) with ```--save_image=True```.

Alternatively, we provide a script ```enhance.py``` that extends some of the existing code from [MIMO-UNet](https://github.com/chosj95/MIMO-UNet) to enhance images without requiring the ground truth images.

To run the script, execute:

```{bash}
python main.py --model_name "MIMO-UNetPlus" --data_dir <path to data to be enhanced> --test_model <path to weights>
```

e.g.

```{bash}
python main.py --model_name "MIMO-UNetPlus" --data_dir "../data/shdocs_dataset/restructured/images" --test_model 'weights/SHDocsMIMO-UNetPlus.pkl'
```
