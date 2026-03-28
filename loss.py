import torch
from torch import nn
from torch.nn.functional import one_hot


class WCEDCELoss(nn.Module):
    def __init__(self, num_classes=8, inter_weights=0.5, intra_weights=None, device='cuda'):
        super(WCEDCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=intra_weights)
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weights = inter_weights
        self.device = device

    def dice_loss(self, prediction, target, weights):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5

        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)

        intersection = (prediction * target)

        dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
        # print('dice: ', dice)
        dice_loss = 1 - dice.sum(0) / batchsize
        weighted_dice_loss = dice_loss* weights

        # print(dice_loss, weighted_dice_loss)
        return weighted_dice_loss.mean()

    def forward(self, pred, label):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        cel = self.ce_loss(pred, label)

        label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()

        if self.intra_weights == None:
            intra_weights = torch.zeros([self.num_classes]).to(self.device)
            for item in range(self.num_classes):

                intra_weights[item] = len(label.view(-1)) / (len(label[label == item].view(-1)) + 1e-5)
        else:
            intra_weights = self.intra_weights
        # print('weights: ', intra_weights)
        dicel = self.dice_loss(pred, label_onehot, intra_weights)
        # print('ce: ', cel, 'dicel: ', dicel)
        loss = cel * self.inter_weights + dicel * (1 - self.inter_weights)

        return loss