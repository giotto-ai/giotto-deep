# This code is taken from the fastai2_extension package.
# acknowlegment to rsomani95

import PIL
import torchvision
from fastai.vision.all import *
from gdeep.gradcam.utility import *

class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()


class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

def create_test_img(learn, f, return_img=True):
    img = PILImage.create(f)
    tsf = torchvision.transforms.ToTensor()
    x = tsf(img)
    x = x.reshape(1,*x.shape)
    if return_img: return img,x
    return x

def to_cuda(*args): return [o.cuda() for o in args]

def get_label_idx(learn:Learner, preds:torch.Tensor,
                  label:Union[str,int,None]):
    """Either:
        * Get the label idx of a specific `label`
        * Get the max pred using `learn.loss_func.decode` and `learn.loss_func.activation`
        * Only works for `softmax` activations as the backward pass requires a scalar index
        * Throws a `RuntimeError` if the activation is a `sigmoid` activation
        """
    if label is not None:
        # if `label` is a string, check that it exists in the vocab
        # and return the label's index
        if isinstance(label,str):
            if not label in learn.dls.vocab: raise ValueError(f"'{label}' is not part of the Learner's vocab: {learn.dls.vocab}")
            return learn.dls.vocab.o2i[label], label
        # if `label` is an index, return itself
        elif isinstance(label,int): return label, learn.dls.vocab[label]
        else: raise TypeError(f"Expected `str`, `int` or `None`, got {type(label)} instead")
    else:
        # if no `label` is specified, check that `learn.loss_func` has `decodes`
        # and `activation` implemented, run the predictions through them,
        # then check that the output length is 1. If not, the activation must be
        # sigmoid, which is incompatible
        if not hasattr(learn.loss_func, 'activation') or\
            not hasattr(learn.loss_func, 'decodes'):
            raise NotImplementedError(f"learn.loss_func does not have `.activation` or `.decodes` methods implemented")
        decode_pred = compose(learn.loss_func.activation, learn.loss_func.decodes)
        label_idx   = decode_pred(preds)
        if len(label_idx) > 1:
            raise RuntimeError(f"Output label idx must be of length==1. If your loss func has a sigmoid activation, please specify `label`")
        return label_idx, learn.dls.vocab[label_idx][0]

def get_target_layer(learn: Learner,
                     target_layer:Union[nn.Module, callable, None]) -> nn.Module:
    if target_layer is None:
        if has_pool_type(learn.model[0]):
            warnings.warn(f"Detected a pooling layer in the model body. Unless this is intentional, ensure that the feature map is not flattened")
        return learn.model[0]
    elif isinstance(target_layer, nn.Module):
        return target_layer
    elif callable(target_layer):
        return target_layer(learn.model)

def compute_gcam_items(learn: Learner,
                       x: TensorImage,
                       label: Union[str,int,None] = None,
                       target_layer: Union[nn.Module, callable, None] = None
                       ):
    """Compute gradient and activations of `target_layer` of `learn.model`
        for `x` with respect to `label`.
        
        If `target_layer` is None, then it is set to `learn.model[:-1]`
        """

    if torch.cuda.is_available():
        learn.model, x = to_cuda(learn.model, x)
    target_layer = get_target_layer(learn, target_layer)
    with HookBwd(target_layer) as hook_g:
        with Hook(target_layer) as hook:
            preds       = learn.model.eval()(x)
            activations = hook.stored
            label_idx, label = get_label_idx(learn,preds,label)
        #print(preds.shape, label, label_idx)
        #print(preds)
        preds[0, label_idx].backward()
        gradients = hook_g.stored

    preds = getattr(learn.loss_func, 'activation', noop)(preds)

    # remove leading batch_size axis
    gradients   = gradients  [0]
    activations = activations[0]
    preds       = preds.detach().cpu().numpy().flatten()
    return gradients, activations, preds, label

def compute_gcam_map(gradients, activations) -> torch.Tensor:
    """Take the mean of `gradients`, multiply by `activations`,
        sum it up and return a GradCAM feature map
        """
    # Mean over the feature maps. If you don't use `keepdim`, it returns
    # a value of shape (1280) which isn't amenable to `*` with the activations
    gcam_weights = gradients.mean(dim=[1,2], keepdim=True) # (1280,7,7)   --> (1280,1,1)
    gcam_map     = (gcam_weights * activations) # (1280,1,1) * (1280,7,7) --> (1280,7,7)
    gcam_map     = gcam_map.sum(0)              # (1280,7,7) --> (7,7)
    return gcam_map

def plt2pil(fig) -> PIL.Image.Image:
    """Convert a matplotlib `figure` to a PILImage"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_img = PIL.Image.open(buf).convert('RGB')
    plt.close('all')
    return pil_img

def plt_decoded(x, ctx):
    'Processed tensor --> plottable image, return `extent`'
    x_decoded = TensorImage(x[0])
    extent = (0, x_decoded.shape[1], x_decoded.shape[2], 0)
    x_decoded.show(ctx=ctx)
    return extent

def plot_gcam(learn, img:PILImage, x:tensor, gcam_map:tensor,
              full_size=True, alpha=0.6, dpi=100,
              interpolation='bilinear', cmap='magma', **kwargs):
    'Plot the `gcam_map` over `img`'
    fig,ax = plt.subplots(dpi=dpi, **kwargs)
    if full_size:
        extent = (0, img.width,img.height, 0)
        show_image(img, ctx=ax)
    else:
        extent = plt_decoded(x, ax)
    
    show_image(gcam_map.detach().cpu(), ctx=ax,
               alpha=alpha, extent=extent,
               interpolation=interpolation, cmap=cmap)

    return plt2pil(fig)

@patch
def gradcam(self: Learner,
            item: Union[PILImage, os.PathLike],
            target_layer: Union[nn.Module, callable, None] = None,
            labels = None,
            show_original=False, img_size=None, alpha=0.5,
            cmap = 'magma',
            font_path=None, font_size=None, grid_ncol=4,
            **kwargs
            ):
    '''Plot Grad-CAMs of all specified `labels` with respect to `target_layer`
        Key Args:
        * `item`: a `PILImage` or path to a file. Use like you would `Learner.predict`
        * `target_layer`: The target layer w.r.t which the Grad-CAM is produced
        Can be a function that returns a specific layer of the model
        or also a direct reference such as `learn.model[0][2]`. If `None`,
        defaults to `learn.model[0]`
        * `labels`: A string, int index, or list of the same w.r.t which the Grad-CAM
        must be plotted. If `None`, the top-prediction is plotted if the
        model uses a Softmax activation, else it must be specified.
        * `show_original`: Show the original image without the heatmap overlay
        * `font_path`: (Optional, recommended) Path to a `.ttf` font to render the text
        * `font_size`: Size of the font rendered on the image
        * `grid_ncol`: No. of columns to be shown. By default, all maps are shown in one row
        '''
    img, x = create_test_img(self,item)
    
    
    if not isinstance(labels, list): labels=[labels]
    if img_size is not None: img=img.resize(img_size)
    if grid_ncol is None: grid_ncol = 1+len(labels) if show_original else len(labels)
    
    gcams = defaultdict()
    
    results = []
    for label in labels:
        grads, acts, preds, _label = compute_gcam_items(self, x, label, target_layer)
        gcams[label] = compute_gcam_map(grads, acts)
        preds_dict = {l:pred for pred,l in zip(preds, self.dls.vocab)}
        pred_img = plot_gcam(self, img, x, gcams[label], alpha=alpha, cmap=cmap)
        pred_img.draw_labels(f"{_label}: {preds_dict[_label]* 100:.02f}%",
                             font_path=font_path, font_size=font_size, location="top")
        results.append(pred_img)
    if show_original:
        img = img.resize(results[0].size)
        results.insert(0, img.draw_labels("Original", font_path=font_path, font_size=font_size, location="top"))
    return make_img_grid(results, img_size=None, ncol=grid_ncol)


