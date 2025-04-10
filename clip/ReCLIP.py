import torch
import clip
import socket
import torch.nn.functional as F
from scipy.sparse import eye
from scipy.sparse import linalg as s_linalg
import numpy as np
from torch import nn

clip_full_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

clip_small_templates = [
    "a photo of a {}"
]
# setup device
if(torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class CLIP_LN_V(nn.Module):
    def __init__(self, class_names, architecture = "ViT-L/14", templates = clip_full_templates, learnable_classifier = False):
        super().__init__()
        
        # Load the CLIP checkpoint
        self.base_model, self.preproces = clip.load(name = architecture, device = device)
        
        # Load the CLIP templates
        self.templates = templates
        
        # Produce the text embeddings based on class names
        with torch.no_grad():
            class_embeddings = self.encode_text(class_names).detach()
        
        if learnable_classifier:
            self.classification_weight = nn.Parameter(class_embeddings, requires_grad=True)
            self.learnable_params = [self.classification_weight]
        
        else:
            self.classification_weight = class_embeddings
            self.learnable_params = []
        
        # Setup parameters for training
        self.setup_parameters()
    
    def setup_parameters(self):
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        
        # Visual related layer norms
        for m in self.base_model.visual.modules():
            if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
                self.learnable_params.append(m.weight)
                self.learnable_params.append(m.bias)
                