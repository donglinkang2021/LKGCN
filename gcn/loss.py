import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos, neg):
        loss = -F.logsigmoid(pos - neg)
        return loss.mean()
    
# refer https://discuss.pytorch.org/t/rmse-loss-function/16540/3
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


# mix BPRLoss and MSELoss
class MixLoss(nn.Module):
    def __init__(self):
        """
        MixLoss(BPRLoss, MSELoss)
        """
        super(MixLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bpr = BPRLoss()

    def forward(self, pos, neg, y):
        bprloss = self.bpr(pos, neg)
        mseloss = self.mse(pos, y)
        return bprloss + mseloss

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pos, neg, margin=1.0):
        loss = F.relu(neg - pos + margin)
        return loss.mean()