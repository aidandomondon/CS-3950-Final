import torch;

# sum of feature map differences for all images in a given batch
def sum_feature_map_changes(old_featmaps, new_featmaps):
    return torch.linalg.matrix_norm(old_featmaps - new_featmaps, dim=(2, 3)).sum();
