from torch import nn

def criterion(output, target):
    # return nn.CrossEntropyLoss()(output, target)
    return nn.BCELoss()(output, target)