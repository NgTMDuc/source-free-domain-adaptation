
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

_tokenizer = _Tokenizer()
# default templates provided by CLIP for ImageNet
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

# default template provided by CLIP 
clip_small_templates = [
    'a photo of a {}.',
]

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.clip_model = clip_model

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_prompt(self,args,ent_interval,adv_dic,pesu_label):
        batch_ent = torch.sum(-pesu_label * torch.log(pesu_label + args.epsilon), dim=1).mean(0)
        batch_ent = batch_ent.numpy().tolist()
        # for i in range(len(adv_dic)):
        if (batch_ent>=ent_interval[0] and batch_ent<ent_interval[1]):
            newctx_con = adv_dic[0]
            print(newctx_con)
        elif (batch_ent>=ent_interval[1] and batch_ent<ent_interval[2]):
            newctx_con = adv_dic[1]
            print(newctx_con)
        elif (batch_ent>=ent_interval[2] and batch_ent<ent_interval[3]):
            newctx_con = adv_dic[2]
            print(newctx_con)
        elif (batch_ent>=ent_interval[3] and batch_ent<ent_interval[4]):
            newctx_con = adv_dic[3]
            print(newctx_con)
        elif(batch_ent>=ent_interval[4]):
            newctx_con = adv_dic[3]
            print(newctx_con)
        elif(batch_ent<=ent_interval[0]):
            newctx_con = adv_dic[0]
            print(newctx_con)
        # List_rd = classname
        # space = " "
        # indices = indices.squeeze(0).cpu().numpy().tolist()
        # newctx = [List_rd[i] for i in indices]
        # newctx_con = space.join(newctx)
        n_ctx = len(newctx_con.split(" "))
        prompt = tokenize(newctx_con).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        # self.ctx_init_state = torch.cat((ctx_vectors,self.ctx_init_state[n_ctx:self.ctx_init_state.shape[0],:].float()), 0)
        ctx_init_state_temp = torch.cat((self.ctx_init_state[0:2,:].float(),ctx_vectors),0)
        self.ctx_init_state = torch.cat((ctx_init_state_temp,self.ctx_init_state[3:,:].float()),0)
        # print(self.ctx_init_state.shape)
        # self.ctx_init_state = torch.cat((ctx_vectors,self.ctx_init_state[n_ctx:self.ctx_init_state.shape[0],:].float()), 0)
        # self.n_ctx = self.ctx_init_state.shape[0]
        # print(newctx_con)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.text_features_wd = None
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def reset_prompt(self,args,ent_interval,adv_dic,pesu_label):
        self.prompt_learner.reset_prompt(args,ent_interval,adv_dic,pesu_label)


    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        # torch.save(prompts, 'learned_prompts.pt')
        # torch.save(prompts, 'learned_tokenized.pt')
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        # text_features_save = text_features.clone().detach()
        # torch.save(text_features_save, 'learned_features.pt')
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits,text_features
    

    def inference_with_updated_logits(self, logits_new):
        # with torch.no_grad():
            # image_features = self.image_encoder(image.type(self.dtype))
        # image_features = all_clip_feature_prompt
        text_features = self.get_text_features()
        # text_features_save = text_features.clone().detach()
        # torch.save(text_features_save, 'learned_features.pt')
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()

        return logits_new,text_features

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

class CLIP_LN_V(nn.Module):
    def __init__(self, device, class_names, architecture = "ViT-L/14", templates = clip_full_templates, learnable_classifier = None):
        super().__init__()
        self.device = device
        # Load the CLIP checkpoint
        self.base_model, self.preprocess = load(
            download_root = DOWNLOAD_ROOT, name = architecture, device = device
        )
        
        self.templates = templates
        
        # Produce the text embedding
        with torch.no_grad():
            class_embeddings = self.encode_text(class_names).detach()
            
        # # classification are set to not learnable by default, learnable_params is a dict of parameters for optimizer to know which parameter to update
        if(learnable_classifier):
            self.classification_weight = nn.Parameter(class_embeddings, requires_grad=True)
            self.learnable_params = [self.classification_weight]
        else:
            self.classification_weight = class_embeddings
            self.learnable_params = []
        
        self.setup_parameters()
        
    
    def setup_parameters(self):
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        
        for m in self.base_model.visual.modules():
            if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
                self.learnable_params.append(m.weight)
                self.learnable_params.append(m.bias)
    
    def encode_text(self, classnames):
        num_class = len(classnames)
        zeroshot_weights = [] # expected size: [class_num * 1]
        for classname in classnames:
            if (isinstance(classname, list)):
                all_prompts = [template.format(classna) for template in self.templates for classna in classname]
            else:
                all_prompts = [template.format(classname) for template in self.templates]
            
            all_tokens = tokenize(all_prompts).to(self.device)
            class_embeddings  = self.base_model.encode_text(all_tokens)
            
            # Normalize -> Average -> Normalize
            class_embeddings = F.normalize(class_embeddings, p = 2, dim = -1)
            class_embeddings = class_embeddings.mean(dim = 0)
            class_embeddings = F.normalize(class_embeddings, p = 2, dim = -1)
            
            zeroshot_weights.append(class_embeddings)
        
        zeroshot_weights = torch.stack(
            zeroshot_weights, dim = 1
        ).to(self.device)
    

        return zeroshot_weights

    def encode_image(self, image):
        image_features = self.base_model.encode_image(image)
        image_features = F.normalize(image_features, p = 2, dim = 1)
        return image_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        self.classification_weight = F.normalize(self.classification_weight, p=2, dim=0)
        logits = 100. * image_features @ self.classification_weight
        return logits, image_features

class CLIP_LN_T(nn.Module):
    def __init__(self, device, architecture='ViT-L/14', templates= clip_small_templates):
        super().__init__()
        
        # Load the CLIP checkpoint
        self.base_model, self.preprocess = load(download_root=DOWNLOAD_ROOT, name = architecture, device = device)
        
        # Load the templates provied by CLIP
        self.templates = templates
        self.short_templates = [self.templates[0]]
        self.num_template = len(self.templates)
        
        # set-up parameters
        self.learnable_params = []
        self.setup_parapmeters()
        
    def setup_parameters(self):
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.learnable_params = []
        
        for m in self.base_model.transformer.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.requires_grad_(True)
                self.learnable_params.append(m.weight)
                self.learnable_params.append(m.bias)
        self.base_model.ln_final.requires_grad_(True)
        self.learnable_params.append(self.base_model.ln_final.weight)
        self.learnable_params.append(self.base_model.ln_final.bias)
        
    # by default, use short template. Unless full_templates=True (used for producing projection matarix and clustering centriods)
    def encode_text(self, classnames, full_templates=False):
        # select template
        if(full_templates):
            curr_template = self.templates
            # collect all prompts
            zeroshot_weights = [] # expected size: [class_num * template_num]
            num_class = len(classnames)
            for classname in classnames:
                if(isinstance(classname, list)):
                    # prepare token then embedding
                    all_prompts = [template.format(classna) for template in self.templates for classna in classname]
                else:
                    # prepare token then embedding
                    all_prompts = [template.format(classname) for template in self.templates]
                    
                all_tokens = tokenize(all_prompts).to(self.device) # [template_num, 77]
                class_embeddings = self.base_model.encode_text(all_tokens) # [num_prompts, 768]
                # normalize, average, normalize again
                class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
                class_embeddings = class_embeddings.mean(dim=0) # [num_class, 768]
                class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
                zeroshot_weights.append(class_embeddings)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
            return zeroshot_weights
        else:
            curr_template = self.short_templates
            curr_num_template = len(curr_template)
            # collect all prompts
            all_prompts = [] # expected size: [class_num * template_num]
            num_class = len(classnames)
            for classname in classnames:
                all_prompts.extend([template.format(classname) for template in curr_template])
            all_tokens = tokenize(all_prompts).to(self.device) # [class_num * template_num, 77]

            # class embeddings
            class_embeddings = self.base_model.encode_text(all_tokens) # [num_prompts, 768]
            class_embeddings = class_embeddings.view(num_class, curr_num_template, -1)
            
            # normalize, average, normalize again
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            class_embeddings = class_embeddings.mean(dim=1) # [num_class, 768]
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            class_embeddings = class_embeddings.transpose(0,1)
            return class_embeddings

    def encode_image(self, image):
        image_features = self.base_model.encode_image(image)
        image_features = F.normalize(image_features, p=2, dim=1)
        return image_features

    def forward(self, image, class_names):
        image_features = self.encode_image(image)
        class_embedding = self.encode_text(class_names) # use generated text embeddings for classification
        logits = 100. * image_features @ class_embedding
        return logits, image_features

class LabelPropagationCluster(nn.Module):
    def __init__(self, device, classification_weight, dataset_size, k=10,  alpha=0.99, cut_dim=768):
        super().__init__()
        
        self.classification_weight = classification_weight
        
        # parameters