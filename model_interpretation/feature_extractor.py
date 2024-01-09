import time
import random
import torch
from tqdm import tqdm
from functools import partial
from PIL import Image
import timm
from torchvision.datasets import ImageNet
from torchvision.datasets import MNIST
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import gc

def subset_imagenet(samples, nc):
    '''Select nc samples per class from the ImageNet validation set.'''

    state = random.getstate()
    random.seed(42)

    indices = []
    for idx_c in range(1000):
        class_indices = random.sample(range(50), nc)
        class_indices = [idx_c*50+ind for ind in class_indices]
        indices += class_indices

    samples = [samples[ind] for ind in indices]

    random.setstate(state)

    return samples

def load_model_pt(name):
    '''Load pre-trained Pytorch model.'''

    if name=='efficientnet':
        from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

        weights = EfficientNet_V2_L_Weights.DEFAULT
        preprocess = weights.transforms(antialias=True)
        model = efficientnet_v2_l(weights=weights)
        @torch.no_grad()
        def get_features(batch):
            map = model.features(batch)
            features = model.avgpool(map)
            return features.squeeze((2,3)), map

    elif name=='regnet':
        from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights

        weights = RegNet_Y_32GF_Weights.IMAGENET1K_V2
        #weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
        #weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
        preprocess = weights.transforms(antialias=True)
        model = regnet_y_32gf(weights=weights)
        @torch.no_grad()
        def get_features(batch):
            x = model.stem(batch)
            map = model.trunk_output(x)
            features = model.avgpool(map)
            return features.squeeze((2,3)), map

    elif name=='vit':
        from torchvision.models import vit_l_16, ViT_L_16_Weights

        weights = ViT_L_16_Weights.IMAGENET1K_V1
        #weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        #weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        preprocess = weights.transforms(antialias=True)
        model = vit_l_16(weights=weights)
        @torch.no_grad()
        def get_features(batch):       
            def hook(module, args, output): 
                storage['output'] = output
                return None

            storage = {}
            handler = model.encoder.register_forward_hook(hook)
            res = model(batch)
            map = storage['output']
            features = map[:,0]  # class token
            handler.remove()
            return features, map
    
    
    model.eval()

    img = preprocess(Image.new('RGB', (224, 224)))
    n_features = get_features(img[None])[0][0].numel()

    return model, get_features, preprocess, n_features

def load_model_timm(name, tag):
    '''Load timm pre-trained model.'''
    model_name = name+'.'+tag

    model = timm.create_model(model_name, pretrained=True, 
                                num_classes=0)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    preprocess = timm.data.create_transform(**data_cfg)
    @torch.no_grad()
    def get_features(batch):
        return model(batch)
    
    model.eval()

    return model, get_features, preprocess, model.num_features

def transform_composition(img, transf1, transf2):
    return transf2(transf1(img))

def extract_features(ds, name, tag, bs, device='cpu'):

    model, get_features, preprocess, n_features = load_model_timm(name, tag)
    model.to(device)

    if ds.transform is None:
        ds.transform = preprocess
    else:
        # Compose dataset current transforms with preprocessing
        ds.transform = partial(transform_composition, transf1=ds.transform, transf2=preprocess)

    num_workers = 0
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers>0
    )

    iter(dl)
    print('Started')
    features_all = torch.zeros(len(ds), n_features)
    ti = time.time()
    for idx, (imgs, _) in tqdm(enumerate(dl), total=len(ds)//bs):
        #imgs = preprocess(imgs)
        imgs = imgs.to(device)
        features = get_features(imgs)
        features = features.to('cpu')

        features_all[idx*bs:(idx+1)*bs] = features
    dt = time.time()-ti
    print("Time: ", dt)

    return features_all, model, preprocess

@torch.no_grad()
def accuracy_pt(model, features, name, labels):
    '''Measure accuracy from extracted features.'''

    if name=='efficientnet':
        scores = model.classifier(features)
    elif name=='regnet':
        scores = model.fc(features)
    elif name=='vit':
        scores = model.heads(features)

    pred = torch.argmax(scores, axis=1)
    acc = (torch.tensor(labels)==pred).sum()/len(labels)

    return acc

@torch.no_grad()
def accuracy_timm(features, name, tag, labels):
    '''Measure accuracy from extracted features.'''

    model_name = name+'.'+tag
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    if name=='convnext_base':
        scores = model.head.fc(features)
    elif 'vit_base_patch16' in name:
        scores = model.head(features)

    pred = torch.argmax(scores, axis=1)
    acc = (torch.tensor(labels)==pred).sum()/len(labels)

    return acc

def get_good_models():

    convnext_models = [
                    'convnext_base.fb_in22k_ft_in1k',
                    #'convnext_base.clip_laion2b',
                    #'convnext_base.clip_laion2b_augreg',
                    'convnext_base.clip_laion2b_augreg_ft_in1k',
                    #'convnext_base.clip_laion2b_augreg_ft_in12k',
                    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
                    #'convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384',
                    #'convnext_base.clip_laiona',
                    #'convnext_base.clip_laiona_320',
                    #'convnext_base.clip_laiona_augreg_320',
                    #'convnext_base.clip_laiona_augreg_ft_in1k_384',
                    'convnext_base.fb_in1k',
                    'convnext_base.fb_in22k',
                    #'convnext_base.fb_in22k_ft_in1k_384'
                    ]
    
    vit_models = [
                #'vit_base_patch16_224.augreg2_in21k_ft_in1k',
                'vit_base_patch16_224.augreg_in1k',
                #'vit_base_patch16_224.augreg_in21k',
                'vit_base_patch16_224.augreg_in21k_ft_in1k',
                'vit_base_patch16_224.dino',
                'vit_base_patch16_224.mae',
                #'vit_base_patch16_224.orig_in21k_ft_in1k',
                #'vit_base_patch16_224.sam_in1k',
                #'vit_base_patch16_clip_224.datacompxl',
                #'vit_base_patch16_clip_224.dfn2b',
                #'vit_base_patch16_clip_224.laion2b',
                'vit_base_patch16_clip_224.laion2b_ft_in1k',
                #'vit_base_patch16_clip_224.laion2b_ft_in12k',
                'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
                #'vit_base_patch16_clip_224.metaclip_2pt5b',
                #'vit_base_patch16_clip_224.openai',
                'vit_base_patch16_clip_224.openai_ft_in1k',
                #'vit_base_patch16_clip_224.openai_ft_in12k',
                'vit_base_patch16_clip_224.openai_ft_in12k_in1k'
                ]
    
    return convnext_models, vit_models

def run_imagenet():

    root_data = 'I:/datasets_ssd/imagenet'
    root_features = 'data/'
    convnext_models, vit_models = get_good_models()

    ds = ImageNet(root_data, split='val')
    #ds.samples = subset_imagenet(ds.samples, nc=2)

    # Get sample filenames
    samples = [('/'.join(item[0].split('\\')[2:]),item[1]) for item in ds.samples]
    files, labels = zip(*samples)

    model_names = convnext_models+vit_models
    for model_name in model_names:
        meta = {'model':model_name,
                'files':list(files),
                'labels':list(labels)}

        name, tag = model_name.split('.')
        features_all, model, preprocess = extract_features(ds, name, tag, 256, device='cuda')
        out_file = f'{root_features}/{name}-{tag}.pt'
        
        print(f'{model_name} accuracy:', accuracy_timm(features_all, name, tag, labels))
        torch.save({'features':features_all, 'meta':meta}, out_file)

        torch.cuda.empty_cache()
        gc.collect()

def run_mnist():

    def to_rgb(img):
        return img.convert('RGB')

    root_data = 'K:/datasets/classification'
    root_features = 'data/'
    convnext_models, vit_models = get_good_models()

    ds = MNIST(root_data, train=False, transform=to_rgb)
    labels = ds.targets
    #ds.data = ds.data[:500]
    #labels = labels[:500]

    model_names = convnext_models+vit_models
    for model_name in model_names:
        meta = {'model':model_name,
                'labels':list(labels)}

        name, tag = model_name.split('.')
        features_all, model, preprocess = extract_features(ds, name, tag, 256, device='cuda')
        out_file = f'{root_features}/{name}-{tag}_mnist.pt'
        
        # Check accuracy
        mapper = PCA(n_components=0.95, whiten=True)
        features_all_pca = mapper.fit_transform(features_all)

        logreg = LogisticRegression(C=0.1, max_iter=1000)
        logreg.fit(features_all_pca, labels)
        acc = logreg.score(features_all_pca, labels)
        print(f'{model_name} accuracy:', acc)

        torch.save({'features':features_all, 'meta':meta}, out_file)

        torch.cuda.empty_cache()
        gc.collect()


if __name__=='__main__':
    run_imagenet()