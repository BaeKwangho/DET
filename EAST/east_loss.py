
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss

    
class EASTLoss(nn.Module):
    """
    """

    def __init__(self,
                 eps=1e-6,
                 **kwargs):
        super(EASTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        l_score, l_geo, l_mask = labels[1:]
        f_score = predicts['f_score']
        f_geo = predicts['f_geo']

        dice_loss = self.dice_loss(f_score, l_score, l_mask)

        #smoooth_l1_loss
        channels = 8
        l_geo_split = torch.split(
            l_geo, num_or_sections=channels + 1, axis=1)
        f_geo_split = torch.split(f_geo, num_or_sections=channels, axis=1)
        smooth_l1 = 0
        for i in range(0, channels):
            geo_diff = l_geo_split[i] - f_geo_split[i]
            abs_geo_diff = torch.abs(geo_diff)
            smooth_l1_sign = torch.less_than(abs_geo_diff, l_score)
            smooth_l1_sign = torch.cast(smooth_l1_sign, dtype='float32')
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + \
                (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
            out_loss = l_geo_split[-1] / channels * in_loss * l_score
            smooth_l1 += out_loss
        smooth_l1_loss = paddle.mean(smooth_l1 * l_score)

        dice_loss = dice_loss * 0.01
        total_loss = dice_loss + smooth_l1_loss
        losses = {"loss":total_loss, \
                  "dice_loss":dice_loss,\
                  "smooth_l1_loss":smooth_l1_loss}
        return losses