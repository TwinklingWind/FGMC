import numpy as np
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import MSELoss

from models.segnet import Extractor, Classifier, Projector
from utils.ssim import SSIM

mse_loss = MSELoss


class FGMC(nn.Module):
    def __init__(self, args):
        super(FGMC, self).__init__()
        self.mode = args.mode

        self.extractor = Extractor()
        self.classifier = Classifier()
        # self.sdfGenerator = SDFGenerator()

        self.epoch_semi = args.epoch_semi
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.downsample = args.downsample
        self.projector = Projector(self.in_dim, self.out_dim, self.downsample)


    def forward(self, x_l=None, x_ul=None, dropout=True):
        if self.mode == 'test' or self.mode == 'sup':
            enc = self.extractor(x_l)
            enc = F.interpolate(enc, scale_factor=2, mode='bilinear', align_corners=True)
            cla = self.classifier(enc, dropout=dropout)

            # sdf = self.sdfGenerator(enc, dropout=dropout)
            return cla

        elif self.mode == 'semi':
            # with torch.no_grad():
            # x_ul_ori

            # enc_ul_ori = self.extractor(x_ul_ori)
            # cla_ul_ori = self.classifier(enc_ul_ori, dropout=dropout)

            # sdf_ul_ori = self.sdfGenerator(enc_ul_ori_c)
            # cla_ul_ori = F.interpolate(cla_ul_ori, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            # sdf_ul_ori = F.interpolate(sdf_ul_ori, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            # dis_to_mask = torch.sigmoid(-1800 * sdf_ul_ori)
            # outputs_soft = torch.softmax(cla_ul_ori, dim=1)

            # cla_ul_ori = F.interpolate(cla_ul_ori, 4, mode='bilinear', align_corners=True)

            enc_ul = self.extractor(x_ul)

            if self.downsample:
                enc_ul = F.avg_pool2d(enc_ul, kernel_size=2, stride=2)

            output_ul = self.projector(enc_ul)  # [b, c, h, w]

            output_ul = F.normalize(output_ul, 2, 1)

            # compute pseudo label
            logits = self.classifier(enc_ul)  # [batch_size, num_classes, h, w]

            pseudo_logits = F.softmax(logits, 1).max(1)[0].detach()  # [batch_size, h, w]
            pseudo_label = logits.max(1)[1].detach()  # [batch_size, h, w]

            return output_ul, pseudo_logits, pseudo_label
        else:
            raise ValueError("No such mode {}".format(self.mode))

    def get_backbone_params(self):
        return self.extractor.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'sup':
            return chain(self.extractor.get_module_params(), self.classifier.parameters(), self.sdfGenerator.parameters())
        elif self.mode == 'semi':
            return chain(self.extractor.get_module_params(), self.classifier.parameters(), self.projector.parameters(), self.sdfGenerator.parameters())
        else:
            raise ValueError("No such mode {}".format(self.mode))
