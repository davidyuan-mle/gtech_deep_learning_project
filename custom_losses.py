# import torch tools
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class LearnableCBFL(nn.Module):
    def __init__(self, num_classes, cls_num_list, beta_init=0.9999, gamma=2.0):
        """
        Args:
        """
        super(LearnableCBFL, self).__init__()
        self.num_classes = num_classes
        self.cls_num_list = cls_num_list
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float64))
        self.gamma = gamma

    def forward(self, input, target):
        """
        Args:
        """
        # Construct class weight graph
        beta = torch.clip(self.beta, min=0., max=1.)
        weights = (1. - beta) / (1. - torch.pow(beta, self.cls_num_list))
        weights = weights / torch.sum(weights)
        weights *= self.num_classes

        # Construct full loss now with gamma
        n_b, _ = input.shape

        # Mask and one hot outputs
        mask = F.one_hot(target, num_classes=self.num_classes)
        onehot_probs = torch.sum(mask * torch.exp(F.log_softmax(input, dim=1)), dim=1, keepdim=False)
        onehot_weights = torch.sum(mask * weights.repeat(n_b, 1), dim=1, keepdim=False)

        # Loss
        if self.gamma == 0.:
            loss = torch.mean(- onehot_weights * torch.log(onehot_probs))
        else:
            loss = torch.mean(- onehot_weights * torch.pow(1. - onehot_probs, self.gamma) * torch.log(onehot_probs))

        return loss

    def get_params(self):
        """
        Get learned beta and gamma during training
        """
        #return nn.functional.sigmoid(self.invsig_beta) # nn.functional.softplus(self.invsoftplus_gamma)
        return self.beta

### 
# CITATION START
### Code for vanilla CBFL taken from assignment 2 implementation by group member Henry Wang

def reweight(cls_num_list, beta=0.9999):
    """
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    # Calculate weighting per paper
    per_cls_weights = (1. - beta) / (1. - torch.pow(beta, cls_num_list))

    # Noramlise weights to one and scale by num_classes
    per_cls_weights = per_cls_weights / torch.sum(per_cls_weights)
    per_cls_weights *= len(cls_num_list)

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, num_classes, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        # Get batch and num classes
        n_b, _ = input.shape
        c = self.num_classes
        
        # Mask and one hot outputs
        mask = F.one_hot(target, num_classes=c)
        onehot_probs = torch.sum(mask * torch.exp(F.log_softmax(input, dim=1)), dim=1, keepdim=False)
        onehot_weights = torch.sum(mask * self.weight.repeat(n_b, 1), dim=1, keepdim=False)

        # Loss
        if self.gamma == 0.:
            loss = torch.mean(- onehot_weights * torch.log(onehot_probs))
        else:
            loss = torch.mean(- onehot_weights * torch.pow(1. - onehot_probs, self.gamma) * torch.log(onehot_probs))

        return loss

###
# CITATION END
###