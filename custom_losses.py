# Adapted from PLNLP: https://github.com/zhitao-wang/PLNLP/blob/19923b5f9ffe94c257ca354ade849071560a92f4/plnlp/loss.py\
# a dump of pairwise loss functions to help improve SWEAL
import torch


def auc_loss(pos_out, neg_out, neg_ratio):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, neg_ratio))
    return torch.square(1 - (pos_out - neg_out)).sum()


def hinge_auc_loss(pos_out, neg_out, neg_ratio):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, neg_ratio))
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()


def weighted_auc_loss(pos_out, neg_out, weight):
    # TODO: accept weight from CLI
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, 1))
    return (weight * torch.square(1 - (pos_out - neg_out))).sum()
