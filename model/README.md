# Model

This directory contains the details to retrain [MIMO-UNetPlus](https://github.com/chosj95/MIMO-UNet) as specified in the paper and use the MIMO-UNetPlus model to enhance images. Base code are attributed to the original authors of MIMO-UNetPlus.

## Prerequisites

Next, install the all the prerequisites and dependenices as specified in the main README dependencies.

## Train

As described in the paper, the model we used was [MIMO-UNetPlus](https://github.com/chosj95/MIMO-UNet). Instructions, initial weights, and the code to retrain MIMO-UNetPlus on SHDocs can be found [here](https://github.com/chosj95/MIMO-UNet?tab=readme-ov-file#train).

The weights of MIMO-UNetPlus were trained on SHDocs with a randomly initialized weights, batch size of 4, a learning rate of 0.0001, and a gamma of 0.5 for 3000 epochs with early stopping based on the validation PSNR. [The weights used in the paper can be found here](https://hometeamsnt-my.sharepoint.com/:u:/g/personal/jovin_leong_hometeamsnt_onmicrosoft_com/EQNbX5o3r_tBg19zlIO2mlYB4iLxUTHKJmB2sm4s52_gMQ?e=9cxj4h).

The training samples used for the validation set are listed in [```validation_ids.txt```](https://github.com/JovinLeong/SHDocs/blob/main/model/validation_ids.txt)

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

Prior to image enhancement, the directory structure can be made easier to work with through the use of the ```restructure_captures``` function in [```utils/image_utils.py```](https://github.com/JovinLeong/SHDocs/blob/main/utils/image_utils.py).
