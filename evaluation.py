import os
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from skimage import data, io

from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
from util import util


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def calculateMean(vars):
    return sum(vars) / len(vars)

def save_img(path, img):
    fold, name = os.path.split(path)
    os.makedirs(fold, exist_ok=True)
    io.imsave(path, img)

def evaluateModel(epoch_number, model, opt, test_dataset, epoch, max_psnr, iters=None):
    
    model.netG.eval()
    
    if iters is not None:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s_iter%d.csv' % (epoch, iters))  # define the website directory
    else:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s.csv' % (epoch))  # define the website directory
    eval_results = {'mask': [], 'mse': [], 'psnr': [], 'fmse':[], 'ssim':[]}
    eval_results_fstr = open(eval_path, 'w')
    eval_results_fstr.writelines('img_path,mask_ratio,mse,psnr,fmse,ssim\n') 

    for i, data in enumerate(test_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # inference
        visuals = model.get_current_visuals()  # get image results
        output = visuals['attentioned']
        real = visuals['real']

        for i_img in range(real.size(0)):
            gt, pred = real[i_img:i_img+1], output[i_img:i_img+1]
            fore_nums = data['mask'][i_img].sum().item()
            img_pred = util.tensor2im(pred)
            img_gt = util.tensor2im(gt)
            mse_score_op = mean_squared_error(img_pred,img_gt )
            psnr_score_op = peak_signal_noise_ratio(img_pred, img_gt, data_range=255)
            fmse_score_op = mean_squared_error(img_pred, img_gt) * 256 * 256 / fore_nums
            ssim_score = ssim(img_pred[...,0], img_gt[...,0],data_range=255, multichannel=False)
            
            # if epoch >= 100:
            # pred_rgb = util.tensor2im(pred)
            img_path = data['img_path'][i_img]
            basename, imagename = os.path.split(img_path)
            basename = basename.split('/')[-2]
            save_img(os.path.join('evaluate', 'results',basename, imagename.split('.')[0] + '.png'), img_pred)
                
            # update calculator
            eval_results['mse'].append(mse_score_op)
            eval_results['psnr'].append(psnr_score_op)
            eval_results['fmse'].append(fmse_score_op)         
            eval_results['ssim'].append(ssim_score) 
            eval_results['mask'].append(data['mask'][i_img].mean().item())
            eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (
                data['img_path'][i_img], 
                eval_results['mask'][-1], 
                mse_score_op, 
                psnr_score_op, 
                fmse_score_op, 
                ssim_score))
        if i + 1 % 100 == 0:
            # print('%d images have been processed' % (i + 1))
            eval_results_fstr.flush()
    eval_results_fstr.flush()

    all_mask_ratio = calculateMean(eval_results['mask'])
    all_mse = calculateMean(eval_results['mse'])
    all_psnr = calculateMean(eval_results['psnr'])
    all_fmse = calculateMean(eval_results['fmse'])
    all_ssim = calculateMean(eval_results['ssim'])
    eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (
        'total', 
        all_mask_ratio, 
        all_mse, 
        all_psnr, 
        all_fmse, 
        all_ssim))
    eval_results_fstr.close()
    print('MSE:%.3f, PSNR:%.3f, fMSE:%.3f, SSIM:%.3f' % (all_mse, all_psnr, all_fmse, all_ssim))
    return all_mse, all_psnr, resolveResults(eval_results)

def resolveResults(results):
    interval_metrics = {}
    mask, mse, psnr, fmse, ssim = np.array(results['mask']), np.array(results['mse']), np.array(results['psnr']), np.array(results['fmse']), np.array(results['ssim'])
    interval_metrics['0.00-0.05'] = [np.mean(mse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                    np.mean(psnr[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                    np.mean(fmse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                    np.mean(ssim[np.logical_and(mask <= 0.05, mask > 0.0)])]

    interval_metrics['0.05-0.15'] = [np.mean(mse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                    np.mean(psnr[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                    np.mean(fmse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                    np.mean(ssim[np.logical_and(mask <= 0.15, mask > 0.05)])]

    interval_metrics['0.15-1.00'] = [np.mean(mse[mask > 0.15]),
                                    np.mean(psnr[mask > 0.15]),
                                    np.mean(fmse[mask > 0.15]),
                                    np.mean(ssim[mask > 0.15])]

    print(interval_metrics)
    return interval_metrics


if __name__ == '__main__':
    # setup_seed(6)
    opt = TrainOptions().parse()   # get training 
    test_dataset = CustomDataset(opt, is_for_train=False)
    test_dataset_size = len(test_dataset)
    print('The number of testing images = %d' % test_dataset_size)
    
    test_dataloader = test_dataset.load_data()
    print('The total batches of training images = %d' % len(test_dataset.dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    # writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    # evaluate for every epoch
    epoch = 0
    max_psnr=0
    epoch_mse, epoch_psnr, epoch_interval_metrics = evaluateModel(epoch, model, opt, test_dataloader, 'eval', max_psnr)
    print('end')
