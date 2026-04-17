from __future__ import print_function
import argparse
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from shutil import copyfile


import utils.semantic_seg as transform
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss

from tqdm import tqdm
import models.network as models
from config import get_config
from models.vit import SwinUnet
from mean_teacher import losses, ramps
from models.MTUNET import MTUNet
from models.model import FGMC


from utils import mkdir_p
from tensorboardX import SummaryWriter

from utils.newutils import DiceLoss
from utils.ssim import SSIM
from utils.utils import multi_validate, update_ema_variables, multi_validate_mt
from utils.visual import visualize

import torch.distributed as dist

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=250,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')
parser.add_argument('--data', default='',
                    help='input data path')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.995, type=float)
parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--wlabeled', action="store_true")
parser.add_argument('--scale', action="store_true")
parser.add_argument('--presdo', action="store_true")
parser.add_argument('--tcsm2', action="store_true")
parser.add_argument('--autotcsm', action="store_true")
parser.add_argument('--multitcsm', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--test_mode', action="store_true")
parser.add_argument('--retina', action="store_true")
parser.add_argument('--kvasir', action="store_true")
parser.add_argument('--clinic', action="store_true")
parser.add_argument('--lits', action="store_true")
parser.add_argument('--la', action="store_true")

parser.add_argument('--mode', default='sup', help='sup')
parser.add_argument('--ignore_index', default=255, type=int)

# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)

#
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float, default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=400.0, help='consistency_rampup')

#
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--base-lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')


# new
parser.add_argument('--downsample', default=True, type=bool, help='downsample')
parser.add_argument('--out-dim', default=128, type=float, help='momentum')
parser.add_argument('--in_dim', default=96, type=float, help='momentum')

parser.add_argument('--temp', default=0.1, type=float, help='momentum')
parser.add_argument('--epoch_semi', default=100, type=int, help='momentum')
parser.add_argument('--select_num', default=2000, type=int, help='momentum')
parser.add_argument('--seed', default=3407, type=int, help='momentum')

parser.add_argument('--step_save', default=20, type=float, help='momentum')

parser.add_argument('--stride', default=8, type=float, help='momentum')

parser.add_argument('--pos-thresh-value', default=0.7, type=float, help='momentum')
parser.add_argument('---weight-unsup', default=0.01, type=float, help='momentum')

# parser.add_argument('--capacity', default=1, type=int, help='momentum')

parser.add_argument('--reduction', default='mean', type=str, help='momentum')

parser.add_argument('--dist-url', default='tcp://127.0.0.1:12475', type=str, help='momentum')
parser.add_argument('--world-size', default=1, type=int, help='momentum')
parser.add_argument('--rank', default=0, type=int, help='momentum')

# swin_unet 需要的
parser.add_argument('--cfg', default='./swin_tiny_patch4_window7_224_lite.yaml', type=str, help='momentum')
parser.add_argument('--opts', default=None, type=str, help='momentum', nargs='+')
parser.add_argument('--zip', action='store_true', help='momentum')

parser.add_argument('--cache_mode', default='part', help='momentum')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

feature_bank = []
pseudo_label_bank = []
step_count = 0

def main():
    global best_acc
    print(args)
    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    copyfile("train_tcsm_mean.py", args.out + "/train_tcsm_mean.py")

    if args.retina:
        mean = [22, 47, 82]
    elif args.kvasir:
        mean = [60, 82, 142]
    elif args.clinic:
        mean = [46.9, 68.8, 102.1]
    else:
        mean = [140, 150, 180]

    if args.retina:
        std = [22, 47, 82]
    elif args.kvasir:
        std = [9.34, 8.31, 8.53]
    elif args.clinic:
        std = [7.41, 8.39, 11.3]
    else:
        std = [16.1, 14.75, 15.33]

    # Data augmentation
    # print(f'==> Preparing skinlesion dataset')
    transform_train = transform.Compose([
        transform.RandomRotationScale(),
        transform.RandomGaussianBlur(),
        # transform.RandomGaussianNoise(),

        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    transform_val = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    transform_for_semi = transform.Compose([

        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    if args.retina:
        import dataset.retina as dataset
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("./data/REFUGE/",
                                                                                                   num_labels=args.n_labeled,
                                                                                                   transform_train=transform_train,
                                                                                                   transform_val=transform_val,
                                                                                                   transform_forsemi=None)
    elif args.kvasir:
        import dataset.kvasir as dataset
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("../data/Kvasir/",
                                                                                                   num_labels=args.n_labeled,
                                                                                                   transform_train=transform_train,
                                                                                                   transform_val=transform_val,
                                                                                                   transform_forsemi=transform_for_semi)

    elif args.lits:
        import dataset.lits as dataset
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("../data/Lits/",
                                                                                                   num_labels=args.n_labeled,
                                                                                                   transform_train=transform_train,
                                                                                                   transform_val=transform_val,
                                                                                                   transform_forsemi=transform_for_semi)

    elif args.la:
        import dataset.lits as dataset
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("../data/LA/",
                                                                                                   num_labels=args.n_labeled,
                                                                                                   transform_train=transform_train,
                                                                                                   transform_val=transform_val,
                                                                                                   transform_forsemi=transform_for_semi)

    else:
        if args.test_mode:
            import dataset.skinlesion_test as dataset
            train_labeled_set, train_unlabeled_set, val_set = dataset.get_skinlesion_dataset("../data/skinlesion/",
                                                                                             num_labels=args.n_labeled,
                                                                                             transform_train=transform_train,
                                                                                             transform_val=transform_val,
                                                                                             transform_forsemi=transform_for_semi)
        else:
            import dataset.skinlesion as dataset
            train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset(
                "../data/skin600/",
                num_labels=args.n_labeled,
                transform_train=transform_train,
                transform_val=transform_val,
                transform_forsemi=transform_for_semi)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=2, drop_last=True)

    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=2, drop_last=True)

    # todo len
    print('train_label_set len:', train_labeled_set.__len__())
    print('train_unlabeled_set len:', train_unlabeled_set.__len__())

    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    # test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Model
    print("==> creating model")

    def create_MyModel(args, ema=False):
        model = FGMC(args)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    def create_MTUNET():
        model = MTUNet(out_ch=args.num_classes)
        model = model.cuda()

        return model

    def create_swin_transformer(config, ema=False):
        model = SwinUnet(config, img_size=224, num_classes=2)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_MyModel(args, ema=False)
    ema_model = create_MyModel(args, ema=True)

    config = get_config(args)
    mts_unet = create_swin_transformer(config, ema=False)
    mts_unet.load_from(config)

    mtt_unet = create_swin_transformer(config, ema=True)
    mtt_unet.load_from(config)

    for param in mtt_unet.parameters():
        param.detach_()

    cudnn.benchmark = True
    print('FGMC Model params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.93, 8.06]).cuda())

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer_mt = optim.Adam(mts_unet.parameters(), lr=args.lr, weight_decay=0.0001)

    # Resume
    if args.resume:
        print('==> Resuming from checkpoint..' + args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']

        print("epoch ", checkpoint['epoch'])

        model.load_state_dict(checkpoint['model'])
        mts_unet.load_state_dict(checkpoint['mts_model'])
        mtt_unet.load_state_dict(checkpoint['mtt_model'])

        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        val_loss, val_result1 = multi_validate(val_loader, model, criterion, use_cuda, args)  # mode='Valid Stats'
        val_loss, val_result2 = multi_validate(val_loader, ema_model, criterion, use_cuda, args)  # mode='Valid Stats'

        visualize(val_loader, model, criterion, 0, use_cuda, args)
        print('val model1')
        print("val_loss", val_loss)
        print("Val model: JA, AC, DI, SE, SP, mIou, ASD \n")
        print(", ".join("%.4f" % f for f in val_result1))

        print('val model2')
        print("val_loss", val_loss)
        print("Val model: JA, AC, DI, SE, SP, mIou, ASD \n")
        print(", ".join("%.4f" % f for f in val_result2))

        return

    writer = SummaryWriter("runs/" + str(args.out.split("/")[-1]))
    writer.add_text('Text', str(args))

    print('start train mt')

    ssim = SSIM()

    for epoch in tqdm(range(0, args.epochs)):
        # test
        if epoch != 0 and epoch % 5 == 0:
            val_loss, val_result = multi_validate(val_loader, model, criterion, use_cuda, args)
            test_loss, val_ema_result = multi_validate(val_loader, ema_model, criterion, use_cuda, args)
            #mts_loss, val_mts_result = multi_validate_mt(val_loader, mts_unet, criterion, use_cuda, args)
            #mtt_loss, val_mtt_result = multi_validate_mt(val_loader, mtt_unet, criterion, use_cuda, args)
            print('\n-------------------val model -------------------')
            print("Val model: JA, AC, DI, SE, SP, MIOU, ASD \n")
            print(", ".join("%.4f" % f for f in val_result))

            print('-------------------val ema model -------------------')
            print("Val model: JA, AC, DI, SE, SP, MIOU, ASD \n")
            print(", ".join("%.4f" % f for f in val_ema_result))
            #
            # print('-------------------val mts_unet -------------------')
            # print("Val model: JA, AC, DI, SE, SP, MIOU \n")
            # print(", ".join("%.4f" % f for f in val_mts_result))
            #
            # print('-------------------val mtt_unet -------------------')
            # print("Val model: JA, AC, DI, SE, SP, MIOU \n")
            # print(", ".join("%.4f" % f for f in val_mtt_result))

            step = args.val_iteration * (epoch)

            writer.add_scalar('Val/loss', val_loss, step)

            writer.add_scalar('Model/JA', val_result[0], step)
            writer.add_scalar('Model/AC', val_result[1], step)
            writer.add_scalar('Model/DI', val_result[2], step)
            writer.add_scalar('Model/SE', val_result[3], step)
            writer.add_scalar('Model/SP', val_result[4], step)

            # writer.add_scalar('Ema_model/JA', val_ema_result[0], step)
            # writer.add_scalar('Ema_model/AC', val_ema_result[1], step)
            # writer.add_scalar('Ema_model/DI', val_ema_result[2], step)
            # writer.add_scalar('Ema_model/SE', val_ema_result[3], step)
            # writer.add_scalar('Ema_model/SP', val_ema_result[4], step)
            # scheduler.step()

            # save model
            big_result = max(val_ema_result[0], val_result[0])
            is_best = big_result > best_acc
            best_acc = max(big_result, best_acc)
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'acc': val_result[0],
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'mtt_model': mtt_unet.state_dict(),
                'mts_model': mts_unet.state_dict(),
            }, is_best)
            print('now best AC = ', best_acc)

        # train
        # model.mode = 'sup'
        ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.93, 8.06]).cuda())
        dice_loss = DiceLoss(args.num_classes)

        train_MT(labeled_trainloader, unlabeled_trainloader, model, ema_model, mts_unet, mtt_unet, optimizer, optimizer_mt,
                  epoch, writer, ce_loss, ssim, dice_loss)

        # if (epoch + 1) % 600 == 0:
        #     lr = args.lr * 0.1 ** ((epoch + 1) // 600)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # lr = args.lr
        # writer.add_scalar('lr', lr, (epoch) * args.val_iteration)
    writer.close()

    print('Best acc:')
    print(best_acc)


def train_MT(labeled_trainloader, unlabeled_trainloader, model, ema_model, mts_unet, mtt_unet, optimizer, optimizer_mt,
              epoch, writer, ce_loss, ssim, dice_loss):
    global global_step
    global step_count
    global feature_bank
    global pseudo_label_bank

    # switch to train mode
    model.train()
    ema_model.train()

    for i_batch, (volume_batch, label_batch, names) in enumerate(labeled_trainloader):

        if i_batch >= 10:
            break

        label_batch[label_batch > 0] = 1	
        model.mode = 'sup'
        ema_model.mode = 'sup'

        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        try:
            images, _, ul1, br1, ul2, br2, flip, img_u, img_dct = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            images, _, ul1, br1, ul2, br2, flip, img_u, img_dct = unlabeled_train_iter.next()

        # targets_x[targets_x == 255] = 1
        img_u2 = img_u.clone()
        img_u3 = img_u.clone()
        img_dct1 = img_dct.clone()

        img_u2 = img_u2.cuda()
        img_u3 = img_u3.cuda()
        img_dct = img_dct.cuda()
        img_dct1 = img_dct1.cuda()

        model_inputs = img_dct
        ema_inputs = img_u2
        mts_model_inputs = img_dct1
        mtt_model_inputs = img_u3

        outputs = model(x_l=volume_batch)
        outputs_unlabel = model(x_l=model_inputs)

        outputs_mts = mts_unet(volume_batch)
        outputs_mts_unlabel = mts_unet(mts_model_inputs)

        with torch.no_grad():
            outputs_ema_unlabel = ema_model(x_l=ema_inputs)
            outputs_mtt_unlabel = mtt_unet(x_l=ema_inputs)

        loss_ce = ce_loss(outputs_mts, label_batch.long())
        loss_dice = dice_loss(outputs_mts, label_batch[:], softmax=True)
        supervised_loss_mt = 0.75 * loss_ce + 0.25 * loss_dice

        loss_ce = ce_loss(outputs, label_batch.long())
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
        supervised_loss = 0.75 * loss_ce + 0.25 * loss_dice

        sup_loss = supervised_loss_mt + supervised_loss

        consistency_weight = get_current_consistency_weight(epoch)

        consistency_loss1 = losses.softmax_kl_loss(outputs_unlabel, outputs_ema_unlabel)
        consistency_loss2 = losses.softmax_kl_loss(outputs_mts_unlabel, outputs_mtt_unlabel)

        consistency_loss3 = losses.softmax_kl_loss(outputs_unlabel, outputs_mtt_unlabel)
        consistency_loss4 = losses.softmax_kl_loss(outputs_mts_unlabel, outputs_ema_unlabel)

        consistency_loss = 0.5 * consistency_loss1 + 0.5 * consistency_loss2 + 0.5 * consistency_loss3 + 0.5 * consistency_loss4

        model.mode = 'semi'
        ema_model.mode = 'semi'
        images = images.cuda()

        model_img = images[:, 0, :, :, :]
        ema_model_img = images[:, 1, :, :, :]

        output_ul1, pseudo_logits_1, pseudo_label1 = model(x_ul=model_img, dropout=True)


        with torch.no_grad():

            output_ul2, pseudo_logits_2, pseudo_label2 = ema_model(x_ul=ema_model_img, dropout=True)

        # ---------calculate contr loss---------- #
        output_feature_list1 = []
        output_feature_list2 = []
        pseudo_label_list1 = []
        pseudo_label_list2 = []
        pseudo_logits_list1 = []
        pseudo_logits_list2 = []
        for idx in range(volume_batch.size(0)):
            output_ul1_idx = output_ul1[idx]
            output_ul2_idx = output_ul2[idx]
            pseudo_label1_idx = pseudo_label1[idx]
            pseudo_label2_idx = pseudo_label2[idx]
            pseudo_logits_1_idx = pseudo_logits_1[idx]
            pseudo_logits_2_idx = pseudo_logits_2[idx]

            if flip[0][idx] == True:
                output_ul1_idx = torch.flip(output_ul1_idx, dims=(2,))
                pseudo_label1_idx = torch.flip(pseudo_label1_idx, dims=(1,))
                pseudo_logits_1_idx = torch.flip(pseudo_logits_1_idx, dims=(1,))
            if flip[1][idx] == True:
                output_ul2_idx = torch.flip(output_ul2_idx, dims=(2,))
                pseudo_label2_idx = torch.flip(pseudo_label2_idx, dims=(1,))
                pseudo_logits_2_idx = torch.flip(pseudo_logits_2_idx, dims=(1,))
            output_feature_list1.append(
                output_ul1_idx[:, ul1[0][idx] // 8:br1[0][idx] // 8, ul1[1][idx] // 8:br1[1][idx] // 8].permute(1,
                                                                                                                2,
                                                                                                                0).contiguous().view(
                    -1, output_ul1.size(1)))
            output_feature_list2.append(
                output_ul2_idx[:, ul2[0][idx] // 8:br2[0][idx] // 8, ul2[1][idx] // 8:br2[1][idx] // 8].permute(1,
                                                                                                                2,
                                                                                                                0).contiguous().view(
                    -1, output_ul2.size(1)))
            pseudo_label_list1.append(pseudo_label1_idx[ul1[0][idx] // 8:br1[0][idx] // 8,
                                      ul1[1][idx] // 8:br1[1][idx] // 8].contiguous().view(-1))
            pseudo_label_list2.append(pseudo_label2_idx[ul2[0][idx] // 8:br2[0][idx] // 8,
                                      ul2[1][idx] // 8:br2[1][idx] // 8].contiguous().view(-1))
            pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[0][idx] // 8:br1[0][idx] // 8,
                                       ul1[1][idx] // 8:br1[1][idx] // 8].contiguous().view(-1))
            pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[0][idx] // 8:br2[0][idx] // 8,
                                       ul2[1][idx] // 8:br2[1][idx] // 8].contiguous().view(-1))
        output_feat1 = torch.cat(output_feature_list1, 0)  # [n, c]
        output_feat2 = torch.cat(output_feature_list2, 0)  # [n, c]
        pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0)  # [n,]
        pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0)  # [n,]
        pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0)  # [n,]
        pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0)  # [n,]
        assert output_feat1.size(0) == output_feat2.size(0)
        assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
        assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

        # concat across multi-gpus
        b, c, h, w = output_ul1.size()
        selected_num = args.select_num
        output_ul1_flatten = output_ul1.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        output_ul2_flatten = output_ul2.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        selected_idx1 = np.random.choice(range(b * h * w), selected_num, replace=False)
        selected_idx2 = np.random.choice(range(b * h * w), selected_num, replace=False)
        output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
        output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
        output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected],
                                               0)  # [2*kk, c]
        output_ul_all = concat_all_gather(output_ul_flatten_selected)  # [2*N, c]

        pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
        pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
        pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected],
                                                  0)  # [2*kk]
        pseudo_label_all = concat_all_gather(pseudo_label_flatten_selected)  # [2*N]

        feature_bank.append(output_ul_all)
        pseudo_label_bank.append(pseudo_label_all)
        # step_save = 20
        if step_count > args.step_save:
            # feature_bank[0] = feature_bank[0].cpu()
            # pseudo_label_bank[0] = pseudo_label_bank[0].cpu()

            feature_bank = feature_bank[1:]
            pseudo_label_bank = pseudo_label_bank[1:]
        else:
            step_count += 1
        output_ul_all = torch.cat(feature_bank, 0)
        pseudo_label_all = torch.cat(pseudo_label_bank, 0)

        eps = 1e-8
        pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / args.temp  # [n, 1]
        pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / args.temp  # [n, 1]

        # compute loss1
        b = 3000

        def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
            # print("gpu: {}, i_1: {}".format(gpu, i))
            mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float()  # [n, b]
            neg1_idx = (output_feat1 @ output_ul_idx.T) / args.temp  # [n, b]
            logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
            return logits1_neg_idx

        def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
            # print("gpu: {}, i_1_0: {}".format(gpu, i))
            mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float()  # [n, b]
            neg1_idx = (output_feat1 @ output_ul_idx.T) / args.temp  # [n, b]
            neg1_idx = torch.cat([pos, neg1_idx], 1)  # [n, 1+b]
            mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().cuda(), mask1_idx], 1)  # [n, 1+b]
            neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0]  # [n, 1]
            logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
            return logits1_neg_idx, neg_max1

        N = output_ul_all.size(0)
        logits1_down = torch.zeros(pos1.size(0)).float().cuda()
        for i in range((N - 1) // b + 1):
            # print("gpu: {}, i: {}".format(gpu, i))
            pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]
            output_ul_idx = output_ul_all[i * b:(i + 1) * b]
            if i == 0:
                logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1,
                                                                              output_ul_idx, pseudo_label_idx,
                                                                              pseudo_label1_overlap)
            else:
                logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx,
                                                                    pseudo_label_idx, pseudo_label1_overlap,
                                                                    neg_max1)
            logits1_down += logits1_neg_idx

        logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)

        # pos_mask_1 = ((pseudo_logits2_overlap > args.pos_thresh_value) & (
        #             pseudo_logits1_overlap < pseudo_logits2_overlap)).float()

        pos_mask_1 = (pseudo_logits2_overlap > -10000).float()

        loss1 = -torch.log(logits1 + eps)
        loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)
        # loss1 = loss1.mean()

        contr_loss = loss1

        # loss_unsup = contr_weight * loss2 + consistency_weight * consistency_loss
        # curr_losses = {}
        # curr_losses['contr_loss'] = args.weight_unsup * contr_loss
        # curr_losses['consist'] = consistency_weight * consistency_loss
        # curr_losses['sup'] = supervised_loss
        # print(curr_losses)
        # curr_losses['loss2'] = self.weight_unsup * loss2

        loss = sup_loss + consistency_weight * consistency_loss + args.weight_unsup * contr_loss

        optimizer.zero_grad()
        optimizer_mt.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_mt.step()

        iter_num = i_batch + epoch * len(labeled_trainloader)
        # print(iter_num)

        update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        update_ema_variables(mts_unet, mtt_unet, args.ema_decay, iter_num)

        # writer.add_scalar('losses/loss_dis', loss_dis, iter_num)
        # writer.add_scalar('losses/loss_seg', total_loss, iter_num)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    with torch.no_grad():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
    return output

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        probs_u = probs_u[:, 1, :, :]

        Lx = F.cross_entropy(outputs_x, targets_x.long(), weight=torch.FloatTensor([1.93, 8.06]).cuda())

        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        # self.tmp_model = models.WideResNet(num_classes=NUM_CLASS).cuda()
        # self.tmp_model = models.DenseuNet(num_classes=NUM_CLASS).cuda()
        self.tmp_model = models.DenseUnet_2d().cuda()
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
