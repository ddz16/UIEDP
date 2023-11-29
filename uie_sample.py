import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import pyiqa
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from uie_codes.UIEB import UIEBDataset
from uie_codes.Loss import MyMAELoss, MyMSELoss, SSIM, MSSSIM, PerceptualLoss, ColorLoss, IlluminationLoss

def save_images(image, save_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # image is a numpy array of shape NHWC
    for i in range(image.shape[0]):
        save_single_image(image[i], save_name[i], save_dir)

def save_single_image(image, save_name, save_dir):
    # image is a numpy array of shape HWC
    # image = normalize_and_quantize(image)
    Image.fromarray(image).save(os.path.join(save_dir, save_name))

def normalize_img(img):
    if th.max(img) > 1 or th.min(img) < 0:
        # img: b x c x h x w
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h*w)
        im_max = th.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = th.min(temp_img, dim=2)[0].view(b, c, 1)

        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-7)
        
        img = temp_img.view(b, c, h, w)
    
    return img

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading UIE dataset...")

    assert args.image_size == 256
    assert args.uie_dataset in ['T90', 'C60', 'U45']
    raw_path = './data/raw/' + args.uie_dataset
    label_path = './data/pseudo_label/' + args.uie_dataset
    output_path = './data/UIEDP/' + args.uie_dataset
    UIEDataset = UIEBDataset(raw_path=raw_path, label_path=label_path, image_size=args.image_size)
    UIEDataloader = DataLoader(UIEDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.log("creat UIE loss...")
    MAELoss = MyMAELoss()
    SSIMLoss = MSSSIM(val_range=2.0)
    PLoss = PerceptualLoss()
    IQALoss = pyiqa.create_metric('musiq-koniq', as_loss=True, device=dist_util.dev()) # input -1~1
    URankerLoss = pyiqa.create_metric('uranker', as_loss=True, device=dist_util.dev()) # input 0~1
    # print(IQALoss.lower_better) False
    # print(URankerLoss.lower_better) False

    def uie_cond_fn(x_0, t, raw, label):
        # x_0: [-1, 1]
        # raw, label: [-1, 1]
        loss = 0
        with th.enable_grad():
            x_in = x_0.detach().requires_grad_(True)
            x_in_new = x_in.clamp(-1, 1)
            loss_mae = MAELoss(x_in_new, label)
            loss_ssim = SSIMLoss(x_in_new, label)
            loss_content1 = PLoss(x_in_new, raw, content_index=[3,4])
            loss_content2 = PLoss(x_in_new, label, content_index=[3,4])
            loss_content = loss_content1 + loss_content2
            loss_iqa = IQALoss(x_in_new)
            loss_uranker = URankerLoss(0.5*(x_in_new+1))

            loss = loss - loss_mae + loss_ssim - 0.005 * loss_content + 0.001 * (0.01*loss_iqa + 2*loss_uranker)
            if t[0].item() % 50 == 0:
                print(t[0].item(), loss_ssim.item(), loss_content.item(), loss_uranker.item(), loss_iqa.item())

            return th.autograd.grad(loss, x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_filenames = []
    all_sizes = []
    for i, data in enumerate(UIEDataloader):
        model_kwargs = {}
        cond_fn = lambda x,t : uie_cond_fn(x, t, raw=data[0].to(dist_util.dev()), label=data[1].to(dist_util.dev()))
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, _ = sample_fn(
            model_fn,
            (data[0].shape[0], 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = (normalize_img(sample) * 255).to(th.uint8)
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_filenames.extend(data[2])
        all_sizes.extend(data[3])
        logger.log(f"created {len(all_filenames)} samples")

    arr = np.concatenate(all_images, axis=0)
    if dist.get_rank() == 0:
        logger.log(f"saving images")
        save_images(arr, all_filenames, save_dir=output_path) 

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        uie_dataset="U45",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
