import torch, argparse, os, sys, ntpath
import torch.nn.functional as F
from termcolor import colored
from torchsummary import summary
from torch import nn
from torchviz import make_dot
from time import sleep
import numpy as np


file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path,os.pardir))
from src.resnet_model.resnet_dense import ResNet34 as ResNet
from src.effnet_models.mobilenetv2 import MobileNetV2 as MobNet
from src.effnet_models.shufflenetv2 import ShuffleNetV2 as ShufNet
from src.effnet_models.mobilenetv2_cifar100 import MobileNetV2 as MobnetCifar100

QUANTIZE_BITS = 32
PRUNE_RATIO = 1.0
QUANT_ALGO = None
MODEL_PIECES = ['conv1','bn1','layer1','layer2','layer3','layer4','conv2','bn2','shortcut']
verbose = False

def gen_file_lines(p):
    with open(p) as f:
        for line in f:
            yield line.rstrip('\n').split()

def get_loss(c):
    i = 1
    d = {}
    for item in c:
        if 'Loss:' in item:
            d[i] = float(item[item.index('Loss:')+1])
            i +=1
    return d

def calc_model_sparsity(m,param_name=['weight']):
    total_elems = 0
    nonzero_elems = 0
    for name, w in m.named_parameters():
         if name.strip().split('.')[-1] in param_name:
             n_e = torch.numel(w)
             not_zero = list(torch.nonzero(w).shape)[0]
             total_elems += n_e
             nonzero_elems += not_zero
    if verbose:
        print("Total model weights: {}\nNonzero weights: {}\nZero value weights: {}\nCompression ratio: {}"\
                .format(total_elems,nonzero_elems,total_elems-nonzero_elems,round(nonzero_elems/total_elems,7)))
    return total_elems, nonzero_elems

def model_weight_distribution(m,param_names=['weight']):
    all_weights = np.array([])
    for name,w in m.named_parameters():
        if name.strip().split('.')[-1] in param_names:
            all_weights= np.append(all_weights, w.flatten().data.numpy())
    return all_weights

def copy_model_weights(model, W_flat, W_shapes, param_name=['weight']):
    offset = 0
    if isinstance(W_shapes, list):
        W_shapes = iter(W_shapes)
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            numel = W.numel()
            W.data.copy_(W_flat[offset: offset + numel].view(shape))
            offset += numel
            
def l0proj(model, k, normalized=True, param_name=['weightA', "weightB", "weightC"]):
    W_shapes = []
    res = []
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name:
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))
    res = torch.cat(res, dim=0)
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = round(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes

def layers_n(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    lay = 1
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_n = W.data.view(-1)
            if W_n.dim() > 0:
                if not normalized:
                    res[layer_name] = W_n.shape[0]
                else:
                    res[layer_name] = float(W_n.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_n.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0
    return res, count_res

def layers_unique(model, param_name=['weight'], normalized=True):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_nz = W.data
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.data.unique().shape[0]
                else:
                    res[layer_name] = float(W_nz.data.unique().shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.data.unique().shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0
    return res, count_res

def param_list(model, param_name):
    for n, w in model.named_parameters():
        if n.split(".")[-1] in param_name:
            yield w

def layers_nnz(model, normalized=True, param_name=['weight']):
    res = {}
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            layer_name = name
            W_nz = torch.nonzero(W.data)
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.shape[0]
                else:
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
                count_res[layer_name] = W_nz.shape[0]
            else:
                res[layer_name] = 0
                count_res[layer_name] = 0

    return res, count_res

def is_valid_file(parser,arg):
    if not os.path.exists(os.path.join(os.getcwd(),arg)):
        parser.error(colored('The file {} does not exist!'.format(arg),'red'))
        exit()
    return arg

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def weigh(model, name = ''):
    print(colored('Weighing {}'.format(name),'cyan'))
    summary(model, input_size = (3,32,32))
    weight_name = ["weight"]
    #print("\tNUM PARAMS {}".format(sum(p.numel() for p in model.parameters())))
    layers = layers_nnz(model, param_name=weight_name)[1]
    layers_n = layers_n(model, param_name=["weight"])[1]
    all_num = sum(layers_n.values())
    sparse_factor = int(all_num * PRUNE_RATIO)
    model_size = sparse_factor * QUANTIZE_BITS
    #print("\t MODEL SIZE {} bits".format(model_size))
    #print("\t = {} MB".format(round((model_size / 8388608),2)))
    weight_bits = [QUANTIZE_BITS for _ in layers] if QUANT_ALGO == "model_size_quant" else None
    #print("\t weight bits {}".format(weight_bits))
    layernnz = lambda m: list(layers_nnz(m, param_name=weight_name)[1].values())
    param_list = lambda m: param_list(m, param_name=weight_name)

def showattr(m, specials = False):
    spec = [x for x in dir(m) if x[0]=='_']
    if specials: print(spec)
    print([x for x in dir(m) if x not in spec])

def load_model(f,classes=10,architecture='resnet'):
    if architecture=='resnet':
        m = ResNet(classes)
    elif architecture=='mobilenet':
        if classes == 10:
            m = MobNet(alpha=1.0)
        else:
            m = MobnetCifar100()
    m.load_state_dict(torch.load(f), strict = False)
    return m

def dense_tensor_to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def iter_and_sparsify_tensors(model_weights):
    x = 1

def new_filename(old, form = 'sparse'):
        n = old.rfind('/')
        return '{}{}_{}.pth'.format(old[:n],\
                old[n:].split('.')[0],form)

def save_sparse(model,name = None):
    prunes = ['weight','bias']
    from collections import OrderedDict
    new_stdic = OrderedDict()
    for k, v in model.state_dict().items():
        new_stdic[k] = dense_tensor_to_sparse(v)
        continue
        if k[:5]=='layer':
            b = k.split('.')
            if ((b[2]!='shortcut') and (b[-1] in prunes)):
                new_stdic[k] = dense_tensor_to_sparse(v)
                continue
        new_stdic[k] = v
    return new_stdic
    # torch.save(new_stdic,name)
    # if verbose: print(colored("Sparse model saved at: {}".format(name),'cyan'))

def main():
    global verbose
    parser = argparse.ArgumentParser(description="A script to weigh and prune models saved as .pth files")
    parser.add_argument('model', metavar = 'path_to_model',
            help = 'Path from working directory to the model saved as .pth file',
            type = lambda x: is_valid_file(parser,x))
    parser.add_argument('--architecture',default='resnet',help="Architecture of the model argument (resnet,mobilenet)")
    parser.add_argument('--dataset',default='cifar10',help="Dataset used for the training of the model argument. (Only important\
            for loading the model architecture but called dataset for consistency)")
    parser.add_argument('--compression_ratio', action = 'store_true', default = False)
    parser.add_argument('-v', '--verbose', action = 'store_true', default = False)
    args = parser.parse_args()
    if args.verbose: verbose = True
    
    model_classes = 10 if args.dataset[-2:]=='10' else 100
    model = load_model(args.model,classes=model_classes,architecture=args.architecture)
    
    if args.compression_ratio:
        if verbose: print(colored("Calculating compression ratio","yellow"))
        tot,nnz = calc_model_sparsity(model)
        del tot,nnz
        if verbose: print(colored("Done","cyan"))

    if verbose: print(colored('All Done','green'))

if __name__=='__main__':
    main()
