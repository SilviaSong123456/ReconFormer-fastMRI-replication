#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
import torch.fx
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from utils import evaluate
import h5py
from tqdm import tqdm
from data import transforms
import torchvision
import nibabel as nib
import os

def evaluate_recon(net_g, data_loader, args, writer = None, epoch = None):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            input, target, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
            mask = mask.to(args.device)
            masked_kspace = masked_kspace.to(args.device)
            output = net_g(input.to(args.device), masked_kspace, mask)

            target = target.to(args.device)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = transforms.complex_abs(output) * std + mean
            target = transforms.complex_abs(target) * std + mean
            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])

        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        print('No. Volume: ', len(outputs))
        if writer != None:
            writer.add_scalar('Dev_Loss/NMSE', metrics['nmse'], epoch)
            writer.add_scalar('Dev_Loss/SSIM', metrics['ssim'], epoch)
            writer.add_scalar('Dev_Loss/PSNR', metrics['psnr'], epoch)
    return metrics['val_loss'], metrics['nmse'], metrics['ssim'], metrics['psnr']

def test_recon_save(net_g, data_loader, args):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):

            input, target, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
            mask = mask.to(args.device)
            masked_kspace = masked_kspace.to(args.device)

            input = input.to(args.device)
            iter_data_time = time.time()
            output = net_g(input, masked_kspace, mask)
            t_comp = (time.time() - iter_data_time)
            #print('itr time: ', t_comp)

            target = target.to(args.device)
            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = transforms.complex_abs(output) * std + mean
            target = transforms.complex_abs(target) * std + mean
            input = transforms.complex_abs(input) * std + mean

            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'input': (input).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
                
#         metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
#         outputs_save = {}
#         targets_save = {}
#         input_save = {}
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])

        zero_filled_metrics = dict(
            nmse=[],
            ssim=[],
            psnr=[]
        )

        outputs_save = {}
        targets_save = {}
        input_save = {}

        os.makedirs("results/figures", exist_ok=True)
        os.makedirs("results/arrays", exist_ok=True)
        
        
        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            input = np.stack([tgt for _, tgt in sorted(inputs[fname])])
            outputs_save[fname] = output
            targets_save[fname] = target
            input_save[fname] = input
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
            
            # Zero-filled reconstruction metrics
            zero_filled_metrics['nmse'].append(
                evaluate.nmse(target, input)
            )
            zero_filled_metrics['ssim'].append(
                evaluate.ssim(target, input)
            )
            zero_filled_metrics['psnr'].append(
                evaluate.psnr(target, input)
            )

            # Save complete volume arrays for later analysis
            safe_name = os.path.splitext(str(fname))[0]

            np.savez_compressed(
                os.path.join("results/arrays", safe_name + "_reconstruction.npz"),
                zero_filled=input,
                reconformer=output,
                ground_truth=target
            )

            # Select the middle slice for visualization
            middle_slice = output.shape[0] // 2

            zf_slice = input[middle_slice]
            recon_slice = output[middle_slice]
            target_slice = target[middle_slice]
            error_slice = np.abs(recon_slice - target_slice)

            # Use the same grayscale range for fair comparison
            vmax = np.percentile(target_slice, 99.5)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(zf_slice, cmap="gray", vmin=0, vmax=vmax)
            axes[0].set_title("Zero-filled")

            axes[1].imshow(recon_slice, cmap="gray", vmin=0, vmax=vmax)
            axes[1].set_title("ReconFormer")

            axes[2].imshow(target_slice, cmap="gray", vmin=0, vmax=vmax)
            axes[2].set_title("Ground Truth")

            error_image = axes[3].imshow(error_slice, cmap="inferno")
            axes[3].set_title("Absolute Error")
            fig.colorbar(error_image, ax=axes[3], fraction=0.046, pad=0.04)

            for axis in axes:
                axis.axis("off")

            fig.suptitle(
                f"{fname} | Middle slice {middle_slice} | 8x acceleration",
                fontsize=12
            )

            plt.tight_layout()

            figure_path = os.path.join(
                "results/figures",
                safe_name + "_comparison.png"
            )

            plt.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            print("Saved figure:", figure_path)          
        
#         metrics = {metric: np.mean(values) for metric, values in metrics.items()}
#         print(metrics, '\n')
#         print('No. Slices: ', len(outputs))
            metrics = {
                metric: np.mean(values)
                for metric, values in metrics.items()
            }

            zero_filled_metrics = {
                metric: np.mean(values)
                for metric, values in zero_filled_metrics.items()
            }

            print("\nZero-filled baseline:")
            print(zero_filled_metrics)

            print("\nReconFormer:")
            print(metrics)

            print("\nNo. Volumes:", len(outputs))

    return metrics['val_loss'], metrics['nmse'], metrics['ssim'], metrics['psnr']