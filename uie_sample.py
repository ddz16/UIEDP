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
    # image is a numpy array of shape NHWC
    for i in range(image.shape[0]):
        save_single_image(image[i], save_name[i], save_dir)

def save_single_image(image, save_name, save_dir):
    # image is a numpy array of shape HWC
    # image = normalize_and_quantize(image)
    Image.fromarray(image).save(os.path.join(save_dir, save_name))

def save_process_images(all_res, save_dir, save_name):
    print(all_res[0].shape)
    batchsize = all_res[0].shape[0]
    for img_index in range(batchsize):
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i, each_time in enumerate(all_res):
            ax = axes.flatten()[i]
            ax.imshow(each_time[img_index])
            t = 1 if i==0 else i*250
            ax.set_title(f"T={t}")
            ax.axis("off")
        plt.tight_layout()

        output_path = os.path.join(save_dir, save_name[img_index])
        plt.savefig(output_path)


def resize_images(filenames, sizes, input_dir="data/", output_dir="data/diff/"):
    for i, filename in enumerate(filenames):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = Image.open(input_path)
        target_size = (int(sizes[i].split("_")[0]), int(sizes[i].split("_")[1]))
        resized_image = image.resize(target_size, resample=Image.BICUBIC)
        resized_image.save(output_path)

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

def normalize_img_1(img):
    if th.max(img) > 1 or th.min(img) < -1:
        # img: b x c x h x w
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h*w)
        im_max = th.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = th.min(temp_img, dim=2)[0].view(b, c, 1)

        temp_img = (2*temp_img - (im_min+im_max)) / (im_max - im_min + 1e-7)
        
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

    # logger.log("loading classifier...")
    # classifier = Discriminator()
    # classifier.load_state_dict(
    #     dist_util.load_state_dict("/home/iscas/ddz/UIE/classifier_model.pth", map_location="cpu")
    # )
    # classifier.to(dist_util.dev())
    # classifier.eval()

    logger.log("loading UIE dataset...")
    UIEDataset = UIEBDataset("/home/iscas/ddz/UIE/data/UIEB/raw-T90", 
                             "/home/iscas/ddz/UIE/data/UIEB/All_Results/UIEC2Net/T90",
                             "/home/iscas/ddz/UIE/data/UIEB/reference-890",
                             "/home/iscas/ddz/UIE/data/UIEB/raw-T90-depth",
                             image_size=args.image_size)
    UIEDataloader = DataLoader(UIEDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.log("creat UIE loss...")
    # MSELoss = MyMSELoss()
    MAELoss = MyMAELoss()
    SSIMLoss = MSSSIM(val_range=2.0)
    # SSIMLoss = pyiqa.create_metric('ssim', as_loss=True, device=dist_util.dev())
    PSNRLoss = pyiqa.create_metric('psnr', as_loss=True, data_range=2.0, device=dist_util.dev())
    PLoss = PerceptualLoss()
    # CLoss = ColorLoss()
    # ILoss = IlluminationLoss()

    # LPIPSLoss = pyiqa.create_metric('lpips-vgg', as_loss=True, device=dist_util.dev())
    # IQALoss = pyiqa.create_metric('musiq-koniq', as_loss=True, device=dist_util.dev())
    # URankerLoss = pyiqa.create_metric('uranker', as_loss=True, device=dist_util.dev()) # input 0~1
    # print(PSNRLoss.lower_better) False
    # print(LPIPSLoss.lower_better) True
    # print(IQALoss.lower_better) False
    # print(URankerLoss.lower_better) False

    def uie_cond_fn(x_0, t, raw, label, gt):
        # x_0: [-1, 1]
        # raw, label: [-1, 1]
        # print(t)
        loss = 0
        # classifier_scale = 12000. # 60000 - (999 - t[0].item()) * 60
        with th.enable_grad():
            x_in = x_0.detach().requires_grad_(True)
            x_in_new = x_in.clamp(-1, 1)

            loss_mae = MAELoss(x_in_new, label) #loss_mae = MAELoss(x_in_new, label, trans) # UIELoss(0.5*(x_in+1).clamp(0, 1), y_hat) 
            loss_psnr = PSNRLoss(x_in_new, label) # PSNRLoss(0.5*(x_in+1).clamp(0, 1), y_hat)
            loss_ssim = SSIMLoss(x_in_new, label)
            loss_content1 = PLoss(x_in_new, raw, content_index=[3,4])
            loss_content2 = PLoss(x_in_new, label, content_index=[3,4])
            loss_content = loss_content1 + loss_content2
            # loss_color = CLoss(x_in_new)
            # loss_illuminate = ILoss(x_in_new)
            # loss_iqa = IQALoss(x_in_new)
            # loss_uranker = URankerLoss(0.5*(x_in_new+1))
            # loss_uranker1 = URankerLoss(y)
            # loss_uranker2 = URankerLoss(y_hat)
            # outputs = classifier(x_in_new)
            # outputs = th.mean(outputs, dim=(1, 2, 3))
            # labels = th.ones_like(outputs).to(dist_util.dev())
            # loss_classifier = F.binary_cross_entropy_with_logits(outputs, labels)
            
            metric_psnr = PSNRLoss(x_in_new, gt)
            metric_ssim = SSIMLoss(x_in_new, gt)

            if args.use_ddim:
                iqa_weight = 0.0005
            else:
                iqa_weight = 0.002
                                                 #0.005                              (0.01*loss_iqa + loss_uranker)
            loss = loss - loss_mae + loss_ssim - 0.005 * loss_content #- 0.01 * loss_illuminate #- 0.1 * loss_content #+ 0.1 * loss_style #- 0.05 * loss_illuminate - 0.5 * loss_color #- 0.1 * loss_illuminate + 0.1 * loss_uranker #+ 0.5*loss_style - loss_illuminate - loss_color #+ 0.01 * loss_uranker#- 1.0 * loss_classifier # + 0.001 * loss_iqa

            if t[0].item() % 50 == 0:
                print(t[0].item(), loss_psnr.item(), loss_ssim.item(), loss_content.item())
                print("gt metric: ", metric_psnr.item(), metric_ssim.item())

            return th.autograd.grad(loss, x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_filenames = []
    all_sizes = []
    for i, data in enumerate(UIEDataloader):
        model_kwargs = {}
        cond_fn = lambda x,t : uie_cond_fn(x, t, raw=data[0].to(dist_util.dev()), label=data[1].to(dist_util.dev()), gt=data[2].to(dist_util.dev()))
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
        all_filenames.extend(data[4])
        all_sizes.extend(data[5])
        # gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_labels, classes)
        # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_filenames)} samples")

        # if i <= 3:
        #     all_res_new = []
        #     for each_time_sample in all_res:
        #         each_time_sample = (normalize_img(each_time_sample) * 255).to(th.uint8)
        #         # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #         each_time_sample = each_time_sample.permute(0, 2, 3, 1)
        #         each_time_sample = each_time_sample.contiguous()
        #         all_res_new.append(each_time_sample.cpu().numpy())
            
        #     save_process_images(all_res_new, save_dir="data/process/", save_name=data[4])


        # if i == 3:
        #     break

    arr = np.concatenate(all_images, axis=0)
    if dist.get_rank() == 0:
        logger.log(f"saving images")
        save_images(arr, all_filenames, save_dir="data/T90_1/")
        # resize_images(all_filenames, all_sizes)

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
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
