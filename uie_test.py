import os
import argparse
import numpy as np
from PIL import Image
from uie_codes.quality_no_refer import calculate_path, calculate_path_NRIQA
from uie_codes.quality_refer import calc_psnr, calc_mse, calc_ssim

parser = argparse.ArgumentParser(description='Testing U45 dataset')

parser.add_argument('--dataset', type=str, default='T90', 
                    help='options:[T90, C60, U45]')

hparams = parser.parse_args()


test_path = "./data/UIEDP/" + hparams.dataset           # the pred images

if hparams.dataset == "T90":
    gt_path = "./data/reference/T90/"   # GT path
else:
    gt_path = None

PSNR_list = []
SSIM_list = []
MSE_list = []

if gt_path is not None:
    for filename in os.listdir(test_path):
        file_path = os.path.join(test_path, filename)
        file_path_gt = os.path.join(gt_path, filename)
        pred_img = Image.open(file_path)
        gt_img = Image.open(file_path_gt)
        gt_img = gt_img.resize((pred_img.size))
        pred_img = np.array(pred_img) / 255.
        gt_img = np.array(gt_img) / 255.
        PSNR_list.append(calc_psnr(pred_img, gt_img, is_for_torch=False))
        SSIM_list.append(calc_ssim(pred_img, gt_img, is_for_torch=False))
        # SSIM_list.append(SSIMLoss(torch.tensor(pred_img).permute(2,0,1).unsqueeze(0).float(), torch.tensor(gt_img).permute(2,0,1).unsqueeze(0).float()).item())
        MSE_list.append(calc_mse(pred_img, gt_img, is_for_torch=False).item())

    print("PSNR: ", np.mean(PSNR_list))
    print("SSIM: ", np.mean(SSIM_list))
    print("MSE: ", np.mean(MSE_list))

calculate_path_NRIQA(test_path)
