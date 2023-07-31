import torch


def mean_pooling(output):
    # b x s x d
    return torch.mean(output, dim=1)

def cls_pooling(output):
    # b x s x d
    return output[:, 0, :]

def next_token_pooling(output):
    # b x s x d
    return output[:, -1, :]

def any_max_pooling(output):
    """
    A special pooling technique for binary classification
    where if any token is 1, then the whole
    """
    assert output.size(-1) == 2
    output2 = torch.softmax(output, dim=-1)
    i = torch.argmax(output2[:, :, 1], dim=1)
    return output[:, i, :]