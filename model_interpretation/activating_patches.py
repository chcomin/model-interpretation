import time
import math
import heapq
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtrainer.module_util import Hook, ReceptiveField
from torchtrainer.img_util import create_grid
from torch.utils.data import DataLoader

class PriorityQueue:
    def __init__(self, n):
        """Class for generating a priority queue. Only the largest n values are kept.

        Args:
            n (int): Number of values to keep.
        """
        self.heap = []
        self.n = n
        self.end = -math.inf  # Smallest value in the queue

    def push(self, val, item):
        if val>self.end:
            heap = self.heap
            heapq.heappush(heap, (val, item))
            if len(heap)==(self.n+1):
                self.pop()
                self.end = heap[0][0]  # Update smallest value

    def pop(self):
        val, item = heapq.heappop(self.heap)
        return val, item
    
    def data(self):
        return sorted(self.heap, reverse=True)
    
    def items(self):
        data = self.data()
        vals, items = zip(*data)
        
        return list(items)
    
    def __len__(self):
        return len(self.heap)
    
def attach_hooks(modules):
    '''Attach forward hooks to a list of modules to get activations.'''

    hooks = []
    for module in modules:
        hooks.append(Hook(module))

    return hooks

@torch.no_grad()
def get_maximum_activations(model, module_names, ds, n=10, bs=16, device='cuda'):
    """Get patches that maximally activates each feature channel of the model.
    `n` patches are returned for each feature.

    Args:
        model: Pytorch model
        module_names: list containing module names
        ds: the dataset
        n (int, optional): number of maximum activations to store. Defaults to 10.

    Returns:
        act_data: dictionary where act_data[module_name][channel_index] contains information
        regarding the top n activations of that channel of the module.
    """

    model.eval()
    model.to(device)
    # Translate module names to objects
    modules = []
    for name in module_names:
        modules.append(model.get_submodule(name))

    dl = DataLoader(ds, bs, shuffle=False)
    #dl = DataLoader(ds, bs, shuffle=False, num_workers=7, persistent_workers=True)
    #iter(dl)
    t1 = time.time()

    hooks = attach_hooks(modules)
    # Apply model to an image to create priority queues for each channel.
    # The queues are used to save only the top n activations
    x = ds[0][0][None].to(device)
    _ = model(x)
    module_data_storage = {}
    for name, module, hook in zip(module_names, modules, hooks):
        act = hook.activation
        num_channels = act.shape[1]
        queues = [PriorityQueue(n) for _ in range(num_channels)]

        module_data_storage[name] = {'hook':hook, 'queues':queues}

    #torch.cuda.synchronize()
    #torch.cuda.cudart().cudaProfilerStart()
    #torch.cuda.nvtx.range_push("main-loop")
    print("Capturing maximum activations...")
    for idx_batch, (imgs, _) in enumerate(dl):
        print('\r\033[K'+f'Batch {idx_batch}', end='')
        #torch.cuda.nvtx.range_push("forward")
        _ = model(imgs.to(device))
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("calculations")
        for name, module_data in module_data_storage.items():
            hook = module_data['hook']
            queues = module_data['queues']
            # For each module, go through the activations for each channel
            act = hook.activation
            num_imgs, num_channels, nr, nc = act.shape
            '''for idx_in_batch in range(num_imgs):
                idx_img = idx_batch*bs + idx_in_batch
                for channel in range(num_channels):
                    idx_max = act[idx_in_batch, channel].argmax().item()
                    r = idx_max//nc
                    c = idx_max - r*nc
                    max_act_loc = (r, c)
                    max_act = act[idx_in_batch, channel, r, c].item()

                    # Save maximum value of activation, the index of the image and the
                    # pixel location of the activation
                    queues[channel].push(max_act, (idx_img, max_act_loc))'''
            # Maximum activation for each image and channel
            max_act_ic, indices_ic = act.reshape(num_imgs, num_channels, -1).max(dim=2)
            # Pixel position of the maximum activations
            r = indices_ic//nc
            c = indices_ic - r*nc
            # Index of the images in the dataset
            #idx_img = torch.arange(idx_batch*bs, idx_batch*bs+num_imgs).reshape(-1,1).expand(-1,num_channels)
            max_act_top, indices_top = max_act_ic.topk(n, dim=0, largest=True, sorted=False)
            
            r = r.to('cpu')
            c = c.to('cpu')
            max_act_top = max_act_top.to('cpu')
            indices_top = indices_top.to('cpu')
            for channel in range(num_channels): 
                for idx in range(n):
                    max_act = max_act_top[idx,channel]
                    idx_img_in_batch = indices_top[idx,channel]
                    ra, ca = r[idx_img_in_batch, channel], c[idx_img_in_batch, channel]
                    idx_img = idx_batch*bs + idx_img_in_batch

                    # Copy data to CPU and include in queues
                    queues[channel].push(max_act.item(), (idx_img.item(), (ra.item(), ca.item())))       
            
    #    torch.cuda.nvtx.range_pop()
    #torch.cuda.nvtx.range_pop()
    #torch.cuda.synchronize()
    #torch.cuda.cudart().cudaProfilerStop()
    # Put (max_act, (idx_img, max_act_loc)) of the n largest activations
    # in act_data[module_name][channel_index]
    print('t1: ', time.time()-t1)
    act_data = {}
    for name, module_data in module_data_storage.items():
        hook = module_data['hook']
        queues = module_data['queues']
        act_data[name] = {}
        for idx, queue in enumerate(queues):
            act_data[name][idx] = queue.data()

        hook.remove()

    return act_data

def maximum_activating_patches(model, module_names, ds, n=10, bs=16, threshold=1e-8, eps=1e-8, device='cuda'):
    """Get image patches from a dataset that maximally activate the features of the model. 

    *Note: if an image contains more than one pixel that maximally activate a channel, only one
    of the pixels is stored.

    Args:
        model: Pytorch model
        module_names: list containing module names
        ds: the dataset
        n (int, optional): number of maximum activations to store. Defaults to 10.
        threshold (float, optional): only returns patches having activation larger than this. 
        eps (float, optional): epsilon to detect non-zero gradient values. Defaults to 1e-8.

    Returns:
        patch_data: dictionary where patch_data[name][channel][idx_act] contains the patches for
        a [module][module_channel][i-th largest activation]. Each value is a dictionary containing
        storage = {'patch_img':patch_img,    # the image of the patch
                    'patch_grad':patch_grad, # the gradient of the patch
                    'bbox':bbox,             # bounding box where the patch was extracted
                    'idx_img':idx_img,       # index of the image in the dataset
                    'max_act':max_act,       # value of the respecive activation
                    'max_act_loc':max_act_loc} # pixel in the feature map of the maximum activation
    """

    model.eval()
    model.to(device)

    #t1 = time.time()
    act_data = get_maximum_activations(model, module_names, ds, n, bs, device=device)
    #print('t1: ', time.time()-t1)
    t2 = time.time()
    '''Get receptive field sizes. Not used because the receptive field size can change depending
    on the position of the activation. For instance, suppose two x2 nearest neighbors interpolations 
    are done in succession. For a convolution with kernel size 3, sometimes the receptive field
    size will be 1 and sometimes 2. '''
    '''rf_size_map = {}
    receptive_field = ReceptiveField(model)
    x = ds[0][0][None].to(device)
    size = x.shape[-2:]
    for name, channel_data in act_data.items():
        bbox_rf, _ = receptive_field.receptive_field_bbox(name, num_channels=x.shape[1], img_size=size)
        rf_size = bbox_rf[2]-bbox_rf[0]+1, bbox_rf[3]-bbox_rf[1]+1
        rf_size_map[name] = rf_size'''

    #receptive_field = ReceptiveField(model)
    print("\nGetting patches...")
    patch_data = {}
    for name, channel_data in act_data.items():
        module = model.get_submodule(name)
        hook = Hook(module, detach=False)
        patch_data[name] = {}
        for channel, max_acts in channel_data.items():
            print('\r\033[K'+f'Channel {channel} of module {name}', end='')
            patch_data[name][channel] = {}
            for idx_act, max_act_item in enumerate(max_acts):
                max_act, (idx_img, max_act_loc) = max_act_item
                if max_act>=threshold:
                    img, _ = ds[idx_img]
                    img = img.to(device)
                    # For a given module, channel and maximum activation, pass the image through the
                    # model and get the pixel corresponding to the maximum activation
                    img.requires_grad_(True)
                    _ = model(img[None])
                    acts = hook.activation
                    pixel = acts[0, channel, max_act_loc[0], max_act_loc[1]]

                    # Calculate the gradient of the pixel with respect to the image
                    pixel.backward()
                    img_grad = img.grad
                    # Get bounding box where gradients are not zero.
                    bbox_grad = get_bbox(img_grad, eps)

                    # Can also get bbox and center of receptive field, but takes more time
                    '''img_size = img.shape[-2:]
                    bbox_rf_act, center_rf_act  = receptive_field.receptive_field_bbox(name, num_channels=img.shape[0], img_size=img_size, 
                                                                                     pixel=max_act_loc)  '''
                    
                    # Do many sanity checks
                    '''bbox_rf, _ = receptive_field.receptive_field_bbox(name, num_channels=img.shape[0], img_size=img_size)
                    size_rf = bbox_rf[2]-bbox_rf[0]+1, bbox_rf[3]-bbox_rf[1]+1
                    if bbox_rf_act[0]==0 or bbox_rf_act[1]==0 or bbox_rf_act[2]==img_size[0]-1 or bbox_rf_act[3]==img_size[1]-1:
                        # If receptive field is at the border
                        pass
                    else:
                        size_rf_act = bbox_rf_act[2]-bbox_rf_act[0]+1, bbox_rf_act[3]-bbox_rf_act[1]+1
                        # Check if the size of the layer receptive field is the same as the size for this specific activation
                        if size_rf_act==size_rf:
                            diff_check = 2
                            r0, c0, r1, c1 = bbox_rf_act
                            center_rf_bbox = r0+size_rf[0]//2, c0+size_rf[1]//2
                            if abs(center_rf_bbox[0]-center_rf_act[0])>diff_check or abs(center_rf_bbox[1]-center_rf_act[1])>diff_check:
                                print(f"Warning, center of receptive field of module {name}, channel {channel} and activation {idx_act} should be {center_rf_bbox}, but got {center_rf_act}.")    
                        else:
                            print(f"Warning, receptive field size of module {name} is {size_rf}, but got size {size_rf_act} for {idx_act}-th activation of channel {channel}.")

                        size_rf_grad = bbox_grad[2]-bbox_grad[0]+1, bbox_grad[3]-bbox_grad[1]+1
                        if size_rf_grad!=size_rf:
                            print(f"Receptive field size of module {name} is {size_rf}, but gradient rf has size {size_rf_grad} for {idx_act}-th activation of channel {channel}.")
                    size_rf_act = bbox_rf_act[2]-bbox_rf_act[0]+1, bbox_rf_act[3]-bbox_rf_act[1]+1
                    size_rf_grad = bbox_grad[2]-bbox_grad[0]+1, bbox_grad[3]-bbox_grad[1]+1
                    if size_rf_grad!=size_rf_act:
                        print(f"Receptive field size of module {name} is {size_rf_act}, but gradient rf has size {size_rf_grad} for {idx_act}-th activation of channel {channel}.")'''


                    # Capture image patch and gradients. Need to save on cpu due to memory constraints
                    r0, c0, r1, c1 = bbox_grad
                    patch_img = img.detach()[:, r0:r1+1, c0:c1+1].to('cpu')
                    patch_grad = img_grad[:, r0:r1+1, c0:c1+1].to('cpu')

                    # Get center by shifting half the known rf size from one of the points of the activation rf
                    # Since the corners of the activation rf can be outside of the image, we need to do some checks
                    '''if bbox_rf_act[0]==0:
                        c_r = bbox_rf_act[2]-size_rf_grad[0]//2
                    else:
                        c_r = bbox_rf_act[0]+size_rf_grad[0]//2
                    if bbox_rf_act[1]==0:
                        c_c = bbox_rf_act[3]-size_rf_grad[1]//2
                    else:
                        c_c = bbox_rf_act[1]+size_rf_grad[1]//2
                    center = c_r, c_c'''
                    
                    storage = {'patch_img':patch_img,
                            'patch_grad':patch_grad,
                            'idx_img':idx_img,
                            'max_act':max_act,
                            'max_act_loc':max_act_loc,
                            'bbox_grad':bbox_grad}
                    patch_data[name][channel][idx_act] = storage

        hook.remove()
    print('t2: ', time.time()-t2)

    return patch_data
    
def translate_coords(point, input_size, output_size):
    """Translate coordinate of a point from one resolution to another, avoiding rounding issues. 
    This function considers that the left border of the first pixel of a signal is at position 0 
    and the right border of the last pixel is at position input_size[i] for dimension i. This is 
    the same assumption when using Pytorch's interpolate function with align_corners=False.
    For instance, for input_size=(2,) and output_size=(8,), the pixel with index 1 at the input
    has position 4*(1+0.5)=6.0 on the output (and not 4*1=4, as would be expect by just multiplying
    the position by the interpolation factor). 

    Args:
        point (tupe): Point to translate
        input_size (tuple): Size of the input signal
        output_size (tuple): Size of the output signal

    Returns:
        tuple: The new point
    """

    point_out = []
    for p, inp_s, out_s in zip(point, input_size, output_size):
        point_out.append(int(out_s*(p+0.5)/inp_s))

    return point_out

def get_bbox(img_grad, eps=1e-8):
    """Get bounding box of nonzero values of a tensor."""

    inds = torch.nonzero(img_grad.abs()>=eps)[:,1:]
    r0, c0 = inds.amin(dim=0)
    r1, c1 = inds.amax(dim=0)
    r0, c0, r1, c1 = r0.item(), c0.item(), r1.item(), c1.item()
    bbox = (r0, c0, r1, c1)

    return bbox

#*** Functions for plotting the results ***

def show_patches(patch_data, module_names=None, n=5, transform=None, reescale_each=False, tile_size=(100,100), width=12):
    """Show patches returned by the function maximum_activating_patches.

    Args:
        patch_data (dict): Dictionary of patches.
        module_names (List[str], optional): Which items from patch_data to plot. Defaults to all.
        n (int, optional): Number of patches to plot. Defaults to 5.
        tile_size (tuple, optional): Size of each image tile in pixels. Defaults to (100,100).
        transform (func, optional): Transform to apply to each patch before plotting. This is useful
        if the patches come from a dataset that was normalized. Defaults to None.
        reescale_each (bool, optional): If True, the gradient of each patch is reescaled between [0,255] 
        individually. If False, a global maximum and minimum is calculated for all gradients
        in a module, which preserves the scale between gradients. Defaults to False.
        width (float): Width of the generated matplotlib figures.

    Returns:
        List[torch.tensor]: List containing the generated images.
    """

    if module_names is None:
        module_names = patch_data.keys()
    if transform is None:
        transform = lambda x:x

    img_grids = []
    for name in module_names:
        channel_data = patch_data[name]
        if reescale_each:
            value_range = None
        else:
            value_range = _get_limits(channel_data, name)

        tensors = []
        texts = []
        for channel, max_acts in channel_data.items():
            for idx_act, max_act_item in max_acts.items():
                patch_img = max_act_item['patch_img']
                max_act = max_act_item['max_act']
                patch_grad = max_act_item['patch_grad']

                patch_img = transform(patch_img)
                patch_grad = _normalize_img(patch_grad, value_range)

                text = f'{channel} - {max_act:.2f}'

                tensors.append(patch_img)
                tensors.append(patch_grad)
                texts.append(text)
                texts.append('')

        img_grid = create_grid(tensors, nrow=2*n, container_shape=tile_size, texts=texts, padding=2, text_height=12)

        nr, nc, _ = img_grid.shape
        ratio = nr/nc
        fig, ax = plt.subplots(figsize=(width, width*ratio))
        ax.imshow(img_grid)
        ax.set_title(name)
        ax.set_axis_off()

        img_grids.append(img_grid)

    return img_grids

def show_patches_mpl(patch_data, module_names=None, n=5, plot_size=(1.5,1.5), transform=None):
    """Plot patch_data instance. Each patch is plotted on its own axis. Very slow.

    Args:
        patch_data (dict): The data
        module_names (list, optional): List of modules to plot. If None, plot all modules in patch_data. Defaults to None.
        n (int, optional): Number of activations of each channel to plot. Defaults to 5.
        plot_size (tuple, optional): Size of each plot. Defaults to (1.5,1.5).
        cmap (str, optional): Colormap to use. Defaults to 'RdYlGn'.
    """

    if module_names is None:
        module_names = patch_data.keys()
    if transform is None:
        transform = lambda x:x

    for name in module_names:
        channel_data = patch_data[name]
        num_channels = len(channel_data)
        fig, axes = plt.subplots(nrows=num_channels, ncols=2*n, figsize=(2*n*plot_size[0],num_channels*plot_size[1]), squeeze=False)
        for channel, max_acts in channel_data.items():
            for idx_act, max_act_item in max_acts.items():
                patch_img = max_act_item['patch_img']
                patch_grad = max_act_item['patch_grad']
                max_act = max_act_item['max_act']

                patch_img = transform(patch_img)
                patch_img = patch_img.permute(1, 2, 0)
                patch_grad = _normalize_img(patch_grad, None).permute(1, 2, 0)

                axi = axes[channel,2*idx_act]
                axg = axes[channel,2*idx_act+1]
                axi.imshow(patch_img, 'gray')
                #axi.imshow(patch_grad_rgba, cmap, norm=norm)
                axi.set_title(f'{channel} - {max_act:.2f}', size=8)
                axi.set_axis_off()
                axg.imshow(patch_grad)
                axg.set_axis_off()

def show_patches_mpl_gray(patch_data, module_names=None, n=5, plot_size=(1.5,1.5), cmap='RdYlGn'):
    """Plot patch_data instance. ***Not maintained***

    Args:
        patch_data (dict): The data
        module_names (list, optional): List of modules to plot. If None, plot all modules in patch_data. Defaults to None.
        n (int, optional): Number of activations of each channel to plot. Defaults to 5.
        plot_size (tuple, optional): Size of each plot. Defaults to (1.5,1.5).
        cmap (str, optional): Colormap to use. Defaults to 'RdYlGn'.
    """

    if module_names is None:
        module_names = patch_data.keys()

    for name in module_names:
        channel_data = patch_data[name]
        num_channels = len(channel_data)
        fig, axes = plt.subplots(nrows=num_channels, ncols=2*n, figsize=(2*n*plot_size[0],num_channels*plot_size[1]), squeeze=False)
        for channel, max_acts in channel_data.items():
            for idx_act, max_act_item in max_acts.items():
                patch_img = max_act_item['patch_img']
                patch_grad = max_act_item['patch_grad']
                max_act = max_act_item['max_act']

                vmin, vmax = get_bounds(patch_grad)
                #patch_grad_rgba, norm = build_rgba(patch_grad, cmap=cmap, s=3)

                axi = axes[channel,2*idx_act]
                axg = axes[channel,2*idx_act+1]
                axi.imshow(patch_img, 'gray')
                #axi.imshow(patch_grad_rgba, cmap, norm=norm)
                axi.set_title(f'{channel} - {max_act:.2f}', size=8)
                axi.set_axis_off()
                axg.imshow(patch_grad, cmap, vmin=vmin, vmax=vmax)
                axg.set_axis_off()
                

        #fig.suptitle(name)
        #fig.tight_layout()

def get_bounds(img):
    """Get symmetric maximum bounds of an image."""

    v = max([abs(img.min()), abs(img.max())])
    vmin, vmax = -v, v

    return vmin, vmax

def _build_rgba(scalar_field, cmap='RdYlGn', s=10, u=0.5):
    '''Map values using a colormap while also changing the transparency of the mapped values
     depending on the value of the scalar_field. Values close to zero become more transparent.'''

    cm = plt.matplotlib.colormaps[cmap]
    vmax = np.abs(scalar_field).max()
    # Normalize to range [-1,1]
    scalar_field_11 = scalar_field/vmax
    # Normalize to range [0,1]
    scalar_field_01 = scalar_field_11/2 + 0.5
    # Calculate alpha values [-1,1]->[0,1]->sigmoid
    alpha = np.abs(scalar_field_11)
    alpha = 1/(1+np.exp(-s*(alpha-u)))
    img = cm(scalar_field_01, alpha=alpha)
    # Create norm that can be used for colorbars
    norm = plt.matplotlib.colors.CenteredNorm(halfrange=vmax)

    return img, norm

def _get_limits(channel_data, module_name):
    """Get minimum and maximum values for layer data."""

    low = math.inf
    high = -math.inf
    for channel, max_acts in channel_data.items():
        for idx_act, max_act_item in max_acts.items():
            patch_grad = max_act_item['patch_grad']
            low = min([low, patch_grad.min()])
            high = max([high, patch_grad.max()])

    return low, high

def _normalize_img(img, value_range=None):
    """Normalize image to the range [0,255].

    Args:
        img (torch.tensor): Input image
        value_range (tuple[int,int], optional): If None, the image intensities are mapped as
        [img.min(),img.max()]->[0,255]. Otherwise, the mapping is [value_range[0],value_range[1]]->[0,255] 
        Defaults to None.

    Returns:
        torch.tensor: The normalized image
    """

    img = img.clone()
    if value_range is None:
        low, high = img.min(), img.max()
    else:
        low, high = value_range
        img.clamp_(min=low, max=high)
    
    img.sub_(low).div_(max(high - low, 1e-5))
    img = (255*img).to(torch.uint8)

    return img

def test():
    import torch.nn as nn

    # Conv filters
    filters = torch.tensor([
        [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]],
        [[1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]],
        [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]],
    ]).float()

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 3, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
            self.conv3 = nn.Conv2d(3, 1, 3, padding=1, bias=False)

            # Set specific filters
            with torch.no_grad():
                self.conv1.weight[:,0] = filters
                self.conv2.weight[:,0] = filters
                self.conv2.weight[:,1] = filters
                self.conv2.weight[:,2] = filters
                self.conv3.weight[0] = filters

        def forward(self, x):

            return self.conv3(self.conv2(self.conv1(x)))

    class Dataset(torch.utils.data.Dataset):

        def __init__(self, x):
            self.x = x
        
        def __getitem__(self, idx):
            return x[idx], None
        
        def __len__(self):
            return len(self.x)

    # Batch of images equal to the filters
    x = torch.tensor([
        [[[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]],
        [[[1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]]],
        [[[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]]],
    ]).float()
    ds = Dataset(x)
    model = SimpleModel()

    patch_data = maximum_activating_patches(model, ['conv1', 'conv2', 'conv3'], ds, n=3, device='cuda')
    plot_data(patch_data, n=3)

if __name__=="__main__":

    from functools import partial
    from torchvision.datasets import ImageNet
    from torchvision.models import resnet50, ResNet50_Weights

    def transform_inv(x, mean, std):

        mean = torch.tensor(mean, device=x.device).reshape(3, 1, 1)
        std = torch.tensor(std, device=x.device).reshape(3, 1, 1)
        x_or = 255.*(std*x + mean)
        x_or = x_or.to(torch.uint8)

        return x_or.permute(1, 2, 0)

    def load_data():

        root = 'I:/datasets_ssd/imagenet'

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()

        categories = weights.meta["categories"]
        preprocess = weights.transforms()

        imagenet = ImageNet(root, split='val', transform=preprocess)
        transform = partial(transform_inv, mean=preprocess.mean, std=preprocess.std)

        return model, imagenet, categories, transform
        
    module_names = ('relu', 'layer1')
    model, ds, categories, transform = load_data()
    ds = [ds[idx] for idx in range(0, 50000, 10)]
    maximum_activating_patches(model, module_names, ds, n=5, bs=512, device='cuda')