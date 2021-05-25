from numpy.lib.type_check import _nan_to_num_dispatcher
import torch
import torch.nn.functional as F
from einops import rearrange

def dice(prediction, target, train=True):
    
    if train:
        prediction = torch.softmax(prediction, dim=1)
    else:
        prediction = torch.argmax(prediction, dim=1)
        prediction = F.one_hot(prediction)
        if len(prediction.shape) == 4:
            prediction = rearrange(prediction, 'b h w c -> b c h w')
        elif len(prediction.shape) == 5:
            prediction = rearrange(prediction, 'b h w d c -> b c h w d')

    target = F.one_hot(target.squeeze(1), num_classes=prediction.shape[1])
    if len(target.shape) == 4:
        target = rearrange(target, 'b h w c -> b c h w')
    elif len(target.shape) == 5:
        target = rearrange(target, 'b h w d c -> b c h w d')


    intersection = torch.sum(prediction * target, dim=tuple(range(2,len(target.shape)))) 
    total_area   = torch.sum(prediction + target, dim=tuple(range(2,len(target.shape))))
    
    return torch.mean( 2 * intersection / total_area, dim=0)
