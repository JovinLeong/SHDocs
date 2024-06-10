# Based heavily off of https://github.com/chosj95/MIMO-UNet
import os
import sys
import time
import torch
import argparse
from PIL import Image as Image
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from models.MIMOUNet import build_net
from utils.common import get_timenow

# Import a module from one directory up
sys.path.append('../')

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count

class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider

def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

def enhance_dataloader(path, batch_size=1, num_workers=0):
    return DataLoader(EnhanceDataset(path),
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers,
                      pin_memory=True)

class EnhanceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self._check_image(self.image_list)
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, self.image_list[idx]))
        
        # Pad image to multiple of 8
        width, height = image.size
        pad_width = 8 - width % 8
        pad_height = 8 - height % 8
        padded_image = Image.new(image.mode, (width + pad_width, height + pad_height))
        padded_image.paste(image, (0, 0))
        
        # Express as 3-channel tensor
        padded_image = F.to_tensor(padded_image)
        if padded_image.shape[0] == 1:
            padded_image = padded_image.expand(3, -1, -1)
        return padded_image, name, (pad_width, pad_height)

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

def _enhance(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.test_model, map_location=device)
    model.load_state_dict(state_dict['model'])
    dataloader = enhance_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, _, pad_dims = data
            input_img = input_img.to(device)
            tm = time.time()
            _ = model(input_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Enhance
        for iter_idx, data in enumerate(dataloader):
            input_img, name, pad_dims = data

            input_img = input_img.to(device)

            tm = time.time()

            pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            # Unpad
            pad_width, pad_height = pad_dims
            pred = pred[:, :, :-pad_height, :-pad_width]
            pred_clip = torch.clamp(pred, 0, 1)

            save_name = os.path.join(args.result_dir, name[0])
            pred_clip += 0.5 / 255
            
            pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
            pred.save(save_name)
            
def main(args):

    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)

    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)

    if torch.cuda.is_available():
        model.cuda()
    _enhance(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNet', choices=['MIMO-UNet', 'MIMO-UNetPlus'], type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/shdocs')
    args = parser.parse_args()
    
    timenow = get_timenow()
    args.model_save_dir = os.path.join(f'results/{timenow}', args.model_name, 'weights/')
    args.result_dir = os.path.join(f'results/{timenow}', args.model_name, 'result_image/')
    main(args)
