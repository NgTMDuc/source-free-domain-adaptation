import torch
import clip
import socket
import torch.nn.functional as F
from scipy.sparse import eye
import json
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
        self.base_model, self.preproces, _ = clip.load(name = architecture, device = device)
        
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
    
    def encode_text(self, classnames):
        # Collect all the prompts
        zeroshot_weights = []
        num_class = len(classnames)
        for classname in classnames:
            if isinstance(classname, list):
                all_prompts = [
                    template.format(classna) for template in self.templates for classna in classname
                ]
            else:
                all_prompts = [template.format(classname) for template in self.templates]
            

            all_tokens = clip.tokenize(
                all_prompts
            ).to(device)
            class_embeddings = self.base_model.encode_text(all_tokens)

            # normalize -> average -> normalize
            class_embeddings = F.normalize(class_embeddings, p = 2, dim = -1)
            class_embeddings = class_embeddings.mean(dim = 0)
            class_embeddings = F.normalize(class_embeddings, p = 2, dim = -1)
            zeroshot_weights.append(class_embeddings)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights

    def encode_image(self, image):
        image_features = self.base_model.encode_image(image)
        image_features = F.normalize(image_features, p = 2, dim = -1)

        return image_features

    def forward(self, image):
        image_features = self.encode_image(image)
        self.classification_weight = F.normalize(self.classification_weight, p = 2, dim = 0)
        logits = 100. * image_features @ self.classification_weight
        return logits, image_features

class CLIP_LN_T(nn.Module):
    def __init__(self, architecture = "ViT-L/14", templates = clip_small_templates):
        super().__init__()

        # Load the CLIP checkpoint
        self.base_model, self.preprocess,_ = clip.load(name = architecture, device = device)

         # load the templates provided by CLIP
        self.templates = templates
        self.short_templates = [self.templates[0]] # ReCLIP-T uses short template for training and use more templates for providing projection matarix and clustering centriods
        self.num_template = len(self.templates)

        # setup parameteres
        self.learnable_params = []
        self.setup_parapmeters()
    # make everything forzen except text encoder layer norm
    def setup_parapmeters(self):
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.learnable_params = []
        # visual related layer norms
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
                    
                all_tokens = clip.tokenize(all_prompts).to(device) # [template_num, 77]
                class_embeddings = self.base_model.encode_text(all_tokens) # [num_prompts, 768]
                # normalize, average, normalize again
                class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
                class_embeddings = class_embeddings.mean(dim=0) # [num_class, 768]
                class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
                zeroshot_weights.append(class_embeddings)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
            return zeroshot_weights
        else:
            curr_template = self.short_templates
            curr_num_template = len(curr_template)
            # collect all prompts
            all_prompts = [] # expected size: [class_num * template_num]
            num_class = len(classnames)
            for classname in classnames:
                all_prompts.extend([template.format(classname) for template in curr_template])
            all_tokens = clip.tokenize(all_prompts).to(device) # [class_num * template_num, 77]

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

# Label propagatin module producing pseudo labels
class LabelPropagationCluster(nn.Module):
    def __init__(self, classification_weight, dataset_size, k = 10, alpha = 0.99, cut_dim = 769):
        super().__init__()

        # Text generated classification weight
        self.classification_weight = classification_weight

        # Parameters
        self.feat_dim = classification_weight.size(0)
        self.num_class = classification_weight.size(1)
        self.num_neighbor = k
        self.dataset_size = dataset_size
        self.image_per_class = self.dataset_size // self.num_class
        self.alpha = alpha
        self.cut_dim = cut_dim

        # Container for pseudo labels, features, etc
        self.all_feat = []
        self.idx_map = []
        self.all_labels = {}
        self.pseudo_labels = {
            i: 0 for i in range(self.dataset_size)
        }
        self.confidence = {i: 0 for i in range(self.dataset_size)}

        # Build projection
        self.update_projection(classification_weight)
        self.update_centroids(classification_weight.t())
    
    def forward(self, x, idx, label):
        # update features into memory
        idx = list(idx.cpu().numpy())
        label = list(label.cpu().numpy())
        self.all_feat.append(x.detach())
        self.idx_map.extend(idx)
        bs = len(label)
        for i in range(bs):
            self.all_labels[idx[i]] = label[i]
    
    # Use SVD method to compute projection matrix
    def update_projection(self, classification_weight = None):
        # Classificaiton weight 
        if classification_weight is not None:
            classification_weight = classification_weight
        else:
            classification_weight = self.centriods.t()

        U, S, V = torch.svd(classification_weight.to(torch.float32)) # U [768, class]
        self.projection_matrix = nn.Parameter((U[:,1:self.cut_dim] @ U[:,1:self.cut_dim].t()).to(torch.float32), requires_grad=False) # [768, 768]
    # update clustering centriods
    def update_centroids(self, centriods):
        self.centriods = centriods

    # return pseudo labels for examples of given indices
    def get_pseudo_label(self, idx):
        idx = list(idx.cpu().numpy())
        pseudo_labels = [self.pseudo_labels[i] for i in idx]
        pseudo_confdence = [self.confidence[i] for i in idx]
        return pseudo_labels, pseudo_confdence

    # calculate closed form solution for label propagation
    def cg_diffusion(self, qsims, Wn, alpha = 0.99, maxiter = 20, tol = 1e-6):
        Wnn = eye(Wn.shape[0]) - alpha * Wn
        out_sims = []
        for i in range(qsims.shape[0]):
            f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
            out_sims.append(f.reshape(-1,1))
        out_sims = np.concatenate(out_sims, axis = 1)
        ranks = np.argsort(-out_sims, axis = 0)
        return ranks, out_sims

    # main function for label propagation
    def perform_label_propagation(self, clear_cache=True, cluster_centriod=False):    
        # stack all feat
        self.all_feat_stack = torch.cat(self.all_feat, dim=0) # [all_feat]
        
        # assertion
        num_record = self.all_feat_stack.size(0)
        assert(len(self.idx_map) == num_record)
        assert(len(self.all_labels) == num_record)
        
        # prepare features
        all_points = torch.cat([self.centriods, self.all_feat_stack], dim=0).detach()
        all_points_original = all_points.cpu().to(torch.float32)
        all_points_project = all_points @ self.projection_matrix
        all_points_project = (F.normalize(all_points_project, p=2, dim=-1)).cpu().to(torch.float32)
        num_example = all_points_project.size(0)
        
        # affinty matrix
        A = (all_points_project @ all_points_project.t() + 1) / 2
        # remove diagonal
        A = A * (1 - torch.eye(num_example)) 
        # only keep topk nearest neighbors
        topk_val, topk_idx = torch.topk(A, self.num_neighbor, dim=1)
        topA = torch.zeros_like(A).scatter_(1, topk_idx, topk_val)
        # symmentric
        W = (topA + topA.t()) / 2
        # normalize 
        D = torch.diag(torch.diag((W @ torch.ones(num_example, num_example)) ** (-0.5)))
        nW = D @ (W @ D)
        # need to use scipy to solve linear system pW = Y
        nw = nW.numpy()

        # produce labels
        Y = np.zeros((self.num_class, num_example))
        for i in range(self.num_class):
            Y[i,i] = 1

        # perform cg optimization
        _, out_sims = self.cg_diffusion(Y, nw, self.alpha)

        # prediction
        prediction = np.argmax(out_sims, axis=1) # [sample.]

        # entropy
        eps = 1e-7  # small epsilon to avoid log(0)
        out_sims_normalized = (out_sims.T / out_sims.sum(axis=1)).T  # row normalize ranks
        out_sims_normalized = np.clip(out_sims_normalized, eps, 1.0)  # clip values to avoid log(0)
        entropy = -out_sims_normalized * np.log(out_sims_normalized)
        ent_confidence = 1 - entropy.sum(axis=1) / np.log(self.num_class)

        # calculate clustering centriods to update ReCLIP-V classification weights
        if(cluster_centriod):
            new_centriods = self.centriods.clone()
            for class_id in range(self.num_class):
                current_class = (prediction == class_id)
                num_current_class = np.sum(current_class)
                current_class_conf = ent_confidence[current_class] # [num_current_class]
                current_class_feat = all_points_original[current_class] # [num_current_class, 768]
                sample_order = np.argsort(current_class_conf)[::-1]
                sample_order = sample_order[:int(num_current_class)].copy()
                current_centriods = current_class_feat[sample_order]
                current_centriods = torch.mean(current_centriods, dim=0)
                current_centriods = F.normalize(current_centriods.to(torch.float32), p=2, dim=0)
                new_centriods[class_id,:] = current_centriods
            
        prediction = prediction[self.num_class:] # first num_class entries correspond to text embeddings of each class
        ent_confidence = ent_confidence[self.num_class:] # first num_class entries correspond to text embeddings of each class

        # save predictions & confidence
        for i in range(len(prediction)):
            self.pseudo_labels[self.idx_map[i]] = prediction[i]
            self.confidence[self.idx_map[i]] = ent_confidence[i]

        # pseudo label result
        pseudo_label_acc = np.mean([self.pseudo_labels[i] == self.all_labels[i] for i in self.idx_map])
        
        # clean up the bucket for next round
        if(clear_cache):
            self.all_feat = []
            self.idx_map = []

        if(cluster_centriod):
            return pseudo_label_acc, new_centriods, self.pseudo_labels, out_sims
        else:
            return pseudo_label_acc, self.pseudo_labels, out_sims

class ReCLIP(nn.Module):
    def __init__(self, 
                 cfg,
                #   test_dataset, 
                  dataset_size):
        super().__init__()
        self.cfg = cfg
        # self.test_dataset = test_dataset
        self.dataset_size = dataset_size

        # with open("./clip/prompts/clip_prompts", "r") as filename:
            # names_prompts = json.load(filename)
            # class_names = names_prompts[f"{cfg.SETTING.DATASET}_{}"][]["classes"]
            # templates = names_prompts[cfg.SETTING.DATASET]["templates"]
        class_names = []
        with open(cfg.name_file) as f:
            for line in f:
                class_names.extend([i for i in line.split()])
        templates = clip_small_templates
        
        # Load the ReCLIP-V model
        self.class_names = class_names

        self.v_model = CLIP_LN_V(
            class_names = class_names,
            templates = templates,
            architecture = cfg.PROPOSAL.ARCH,
            learnable_classifier = False 
        )

        if torch.cuda.is_available():
            self.v_model.to(device)

        
        ### Optimizer for ReCLIP-V visual-encoder layer-norm paramters
        self.v_optimizer = torch.optim.SGD(
            self.v_model.learnable_params, 1e-3, weight_decay = 1e-4, momentum=0.9
        )

        # Load the ReCLIP-T model
        self.t_model = CLIP_LN_T(
            architecture = cfg.PROPOSAL.ARCH, templates = templates
        )

        ### Optmizer for ReCLIP-T text-encoder layer-norm paramters
        self.t_optimizer = torch.optim.SGD(
            self.t_model.learnable_params, 1e-3, weight_decay = 1e-4, momentum = 0.9
        )

        self.v_label_propagation = LabelPropagationCluster(
            self.v_model.classification_weight, self.dataset_size, k = cfg.PROPOSAL.NEIGHBOR_SIZE, alpha = cfg.PROPOSAL.ALPHA, cut_dim= cfg.PROPOSAL.CUT_DIM
        )
        self.t_label_propagation = LabelPropagationCluster(
            self.v_model.classification_weight, self.dataset_size, k = cfg.PROPOSAL.NEIGHBOR_SIZE, alpha = cfg.PROPOSAL.ALPHA, cut_dim= cfg.PROPOSAL.CUT_DIM
        )

    def generate_pseudo_labels(self, test_dataset):
        for (image, inputs_test_augs), label, tar_idx in test_dataset:
            image_input = image.to(device)
            label = label.to(device).view(-1)

            # Forward pass of the ReCLIP-T
            t_logits, t_feature = self.t_model(image_input, self.class_names)
            # TODO: Compute the accuracy
            # t_acc = None
            
            # Forward pass of the ReCLIP-V
            v_logits, v_feature = self.v_model(image_input)
            # TODO: Compute the accuracy
            # v_acc = None
            
            # Update the label propagation mododules with visual features collected from ReCLIP-V and ReCLIP-T
            self.t_label_propagation(t_feature, tar_idx, label)
            self.v_label_propagation(v_feature, tar_idx, label)

            # Combined logits for prediction
            c_logits = (t_logits + v_logits) / 2
            # TODO: Compute the accuracy
            # c_acc = None
        
        # After collecting features from all samples, perform label propagation
        # Label propagation function returns the accuracy of pseudo labels generated by ReCLIP-T
        pt_acc, pseudo_labels, out_sims = self.t_label_propagation.perform_label_propagation(clear_cache=True)
        pv_acc, centriods, pseudo_labels, out_sims = self.v_label_propagation.perform_label_propagation(clear_cache=True, cluster_centriod=True)

        # Update the classification weights with clustering centroids
        self.v_model.classification_weight = centriods.t()
    
        return c_logits, pseudo_labels, out_sims
    
    def update_ReCLIP_modules(self, test_dataset):
        criterion = torch.nn.CrossEntropyLoss(reduction = "none")
        for (images, inputs_test_augs), label, tar_idx in test_dataset:
            image_input = images.to(device)
            label = label.to(device).view(-1)

            # Forward pass of ReCLIP-T
            t_logits, _ = self.t_model(image_input, self.class_names)

            # Forward pass of ReCLIP-V
            v_logits, _ = self.v_model(image_input)

            # Get pseudo labels for ReCLIP-T, based on current example idx
            t_pseudo_labels, _ = self.t_label_propagation.get_pseudo_label(tar_idx)
            t_pseudo_labels = torch.LongTensor(t_pseudo_labels).to(device)

            # Get pseudo labels for ReCLIP-V, based on current example idx
            v_pseudo_labels, _ = self.v_label_propagation.get_pseudo_label(tar_idx)
            v_pseudo_labels = torch.LongTensor(v_pseudo_labels).to(device)

            # Use the commonly agreed pseudo labels for training
            confidence_map = (v_pseudo_labels == t_pseudo_labels)

            # Update the 2 modules if there exits commonly labels
            if torch.sum(confidence_map) > 0:
                self.t_optimizer.zero_grad()
                t_loss = torch.sum(criterion(t_logits, t_pseudo_labels) * confidence_map) / torch.sum(confidence_map)
                t_loss.backward()
                self.t_optimizer.step()

                self.v_optimizer.zero_grad()
                v_loss = torch.sum(criterion(v_logits, v_pseudo_labels) * confidence_map) / torch.sum(confidence_map)
                v_loss.backward()
                self.v_optimizer.step()
            
            # Update the projection matrix and the classification weight in ReCLIP-T
            # For ReCLIP-V, it uses  clustering centriods for classification -> don't need update
        with torch.no_grad():
            classificaition_weight_t = self.t_model.encode_text(self.class_names, full_templates = True)
            self.t_label_propagation.update_projection(classificaition_weight_t)
            self.t_label_propagation.update_centroids(classificaition_weight_t.t())




