import torch
def find_pair(image_bg,image_fg_original,digit_label):
    image_fg = torch.zeros_like(image_bg)
    for i,im in enumerate(image_bg):
        i_label = digit_label[i]
        all_digits_same_label = (digit_label == i_label).nonzero(as_tuple=True)[0]
        perm = torch.randperm(all_digits_same_label.size(0))
        idx = perm[0]
        
        image_fg[i] = image_fg_original[all_digits_same_label[idx]]
    return image_bg, image_fg