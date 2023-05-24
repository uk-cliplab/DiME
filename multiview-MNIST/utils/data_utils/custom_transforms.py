import torch

"""
Custom pytorch data transforms
"""

class AddUniformNoise(object):
    """
    Adds noise that's sampled uniformly from [noise_min, noise_max)
    """
    def __init__(self, noise_min=0., noise_max=1.):
        self.noise_min = noise_min
        self.noise_max = noise_max
        
    def __call__(self, tensor):
        return tensor + (self.noise_min + torch.rand(tensor.size()) * (self.noise_max - self.noise_min))
    
    def __repr__(self):
        return self.__class__.__name__ + '(noise_min={0}, noise_max={1})'.format(self.noise_min, self.noise_max)

class Clamp(object):
    """
    Clamps a tensors value to be between [min_val,  max_val]
    """
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, tensor):
        return torch.clamp(tensor, min=self.min_val, max=self.max_val)
    
    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class UnitNormalize(object):
    """
    Normalizes a tensor to lie within [0, 1]
    """
    def __init__(self):
        pass
    
    def __call__(self, tensor):
        tensor = tensor - torch.min(tensor)
        tensor = tensor / torch.max(tensor)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__