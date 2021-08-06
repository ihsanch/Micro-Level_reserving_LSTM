import torch
import torch.distributions
import torch.nn as nn

def class_loss(Ind_pay_pred, Ind_pay_true, weights): #classification loss function per batch
    cross_entropy_loss = nn.BCELoss(reduction='sum') #Binary CrossEntropy
    current_loss = 0 #initialize loss
    total_weight = weights.sum(0) #compute the total number of observations used to evaluate the loss

    for i in range(11): #for each period (counting starts from 0 to 10, i.e., 11 periods to predict)
        pay_true_index = (weights[:, i] == 1).nonzero() #total number of observed target in period j
        if pay_true_index.size()[0] > 1: #if there is at least one target I_{k,j}!= NA, we evaluate the loss for period j
            current_loss += cross_entropy_loss(Ind_pay_pred[:, i][pay_true_index[:, 0]].squeeze(), Ind_pay_true[:, i][pay_true_index[:, 0]].squeeze())

    if total_weight.sum() == torch.zeros(1):
        #if for all the batch there is no observed target at the evaluation date (all period combined)
        #we return a zero torch
        loss = torch.zeros(1)
    else:
        loss = current_loss / total_weight.sum() #
    return loss

def regression_loss(pay_pred, pay_true, weights): #regression loss function per batch
    #pay_pred: predicted payments
    #pay_true: observed targets
    #weights: tensor with value 0 if Y_{k,j}=0 or NA, and 1 otherwise
    MSEloss = nn.MSELoss(reduction='sum') #Mean squared error function
    current_loss = 0
    total_weight = weights.sum(0)

    for i in range(11):
        pay_true_index = (weights[:, i] == 1).nonzero() #total number of non-zero observed target Y_{k,j} in period j
        if pay_true_index.size()[0] > 1:  #if there is at least one target Y_{k,j}!= 0 and NA, we evaluate the loss for period j
            current_loss += MSEloss(weights[:, i][pay_true_index[:, 0]] * pay_pred[:, i][pay_true_index[:, 0]], weights[:, i][pay_true_index[:, 0]] * pay_true[:, i][pay_true_index[:, 0]])  # * total_weight[i]

    if total_weight.sum() == torch.zeros(1):
        # if for all the batch there is no observed target at the evaluation date (all period combined)
        # we return a zero torch
        loss = torch.zeros(1)
    else:
        loss = current_loss / total_weight.sum()
    return loss