import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
# from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader_new,create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options import TrainOptions
from data.process import get_processing_model
from util import set_random_seed

import wandb


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.isVal = True
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    set_random_seed()
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()

    opt.real_root = "/data3/xy/proj/bench/existing_dataset/genimage/stable_diffusion_v_1_4/train/0_real"
    opt.fake_root = "/data3/xy/proj/bench/existing_dataset/detection_dataset_random_compress/genimage/train/stable_diffusion_v_1_4"
    opt.checkpoints_dir = "/data3/chenweiyan/2024-12/2025-1/data/detection-method-ckpt/CNNSpot/genimage-sdv1.4-jpeg-aligned"
    
    val_opt.real_root = "/data3/xy/proj/bench/existing_dataset/genImage_test/stable_diffusion_v_1_4/0_real"
    val_opt.fake_root = "/data3/xy/proj/bench/existing_dataset/detection_dataset_random_compress/genimage/test/stable_diffusion_v_1_4/generate"
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="CNNSpot-genimage-sdv1.4-jpeg-aligned",

    # track hyperparameters and run metadata
    config={
    "learning_rate": opt.lr,
    "architecture": opt.name,
    "dataset": "genimage-sdv1.4-jpeg-aligned",
    "epochs": opt.niter,
    "batch_size": opt.batch_size,
    "blur_prob": opt.blur_prob,
    "jpg_prob": opt.jpg_prob,
    "crop_size": opt.CropSize,
    
    }
)
    
    
    data_loader = create_dataloader_new(opt)
    dataset_size = len(data_loader)
    # print('#training images = %d' % dataset_size)

    # train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    # val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    opt = get_processing_model(opt)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                wandb.log({"train_loss": model.loss})
                # train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            # model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap, r_acc, f_acc = validate(model.model, val_opt)[:4]
        # val_writer.add_scalar('accuracy', acc, model.total_steps)
        # val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}, r_acc: {}, f_acc: {}".format(epoch, acc, ap. r_acc, f_acc))
        wandb.log({"val_acc": acc, "val_ap": ap, "val_r_acc": r_acc, "val_f_acc": f_acc})

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
    wandb.finish()  

