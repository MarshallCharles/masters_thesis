import torch, time, os, sys, argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from termcolor import colored
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.resnet_model.resnet_dense import ResNet34 as ResNet
from src.effnet_models.mobilenetv2 import MobileNetV2 as MobNet
from src.effnet_models.mobilenetv2_cifar100 import MobileNetV2 as MobnetCifar100
from src.effnet_models.shufflenetv2 import ShuffleNetV2 as ShufNet
from src.attack import ryax_attacker,Carlini_Wagner
from src.evaluate import acc_call,single_predict
from src.data_worker import get_data, select_gpu
from contextlib import contextmanager

verbose = False

def test_timer(f):
    def wrap(*args,**kwargs):
        time1 = time.time()
        ret = f(*args,**kwargs)
        time2 = time.time()
        return ret,time2-time1
    return wrap

@contextmanager
def logger(args):
    f = open(args.log,'a')
    f.write('\n{}\nTest starting at {}\n'.format('='*50,datetime.now()))
    for k,v in args.__dict__.items():
        f.write('{}: {}\n'.format(k,v))
    try:
        yield f
    finally:
        f.write('END TEST\n{}'.format('='*50))
        f.close()

def is_valid_file(arg):
    l=arg
    if type(arg)!=list:
        l = [arg]
    for item in l:
        assert os.path.isfile(os.path.join(os.getcwd(),item)),colored('{} file does not exist\nExiting'.format(item),'red')
        assert item[-4:]=='.pth',colored('{} not valid .pth file'.format(item),'red')
    return arg

def is_valid_arch(arg):
    l=arg
    if type(arg)!=list:
        l=[arg]
    for i in range(len(l)):
        assert l[i] in ['mobilenet','resnet','shufflenet'],colored('{} architecture not supported'.format(l[i]),'red')
    return arg

def validate_and_build_models(args):
    return load_source_model(is_valid_file(args.source_model),is_valid_arch(args.source_arch),args.dataset),\
            load_target_models(is_valid_file(args.target_models.split(',')),is_valid_arch(args.target_arch.split(',')),args.dataset)

def load_source_model(f,a,dataset='cifar10'):
    classes = 10 if dataset=='cifar10' else 100
    if a=='resnet':
        model = ResNet(classes)
    elif a=='mobilenet':
        if classes==10:
            model = MobNet(alpha=1.0)
        else:
            model = MobnetCifar100()
    model.load_state_dict(torch.load(f), strict = False)
    return model

def load_target_models(files,archs,dataset='cifar10'):
    classes = 10 if dataset=='cifar10' else 100
    ret = []
    for i in range(len(files)):
        if archs[i]=='resnet':
            model = ResNet(classes)
        elif archs[i]=='mobilenet':
            if classes==10:
                model = MobNet(alpha=1.0)
            else:
                model = MobnetCifar100()
        model.load_state_dict(torch.load(files[i]), strict = False)
        ret.append([files[i],archs[i],model])
    return ret

def transfer_test(source,targets,dataloader,atk,batch_size=128,iscuda=False):
    source.eval()
    for i in range(len(targets)):
        targets[i][2].eval()
        targets[i].append(0)
    niter,source_acc = 0,0
    for data, target in tqdm(dataloader,total = round((10000)/batch_size)):
        indx_target = target.clone()
        niter += data.shape[0]
        if iscuda:
            data,target = data.cuda(),target.cuda()
        attacked_data,_ = atk(source,data,target)
        if iscuda:
            attacked_data = attacked_data.cuda()
        with torch.no_grad():
            source_out,_ = single_predict(source,attacked_data) 
            source_acc += acc_call(source_out,indx_target)
            for t in range(len(targets)):
                targ_out,_ = single_predict(targets[t][2],attacked_data)
                targets[t][3] += acc_call(targ_out,indx_target)
    source_acc /= niter
    for t in range(len(targets)):
        targets[t][3] /= niter
    return source_acc,targets

def main():
    global verbose
    parser = argparse.ArgumentParser(description="Transfer attack")
    parser.add_argument('--verbose',default=True,type=bool,help='print verbose') 
    parser.add_argument('--gpu', default="0", help="index of GPUs to use")
    parser.add_argument('--batch_size',default=128,type=int,help='batch size')
    parser.add_argument('--ngpu',default=1,type=int, help='number of GPUs to use')
    parser.add_argument('--seed',default=117,type=int,help="random seed (default: 117)")
    parser.add_argument('--log',default='logs/transfer_log',help = 'log file for the test')
    parser.add_argument("--dataset",default='cifar10',type=str,help='which dataset')
    parser.add_argument("--source_model", default = None, help="Source model for attack")
    parser.add_argument("--source_arch",default = 'resnet',help='architecture for source model')
    parser.add_argument("--target_models", default = None,help="A list of the model files to attack" )
    parser.add_argument("--target_arch",default=None,help="A list of the target architectures")
    parser.add_argument("--atk_algo",default='pgd',help="Adversarial attack for the test")
    parser.add_argument("--epsilon",default=8,type=int,help="attack radius")
    parser.add_argument("--steps",default=7,type=int,help="attack iterations")
    args = parser.parse_args() 
    if args.verbose: verbose = True 
    
    if verbose: print(colored('Configuring Hardware & Models','yellow'))
    args.gpu = select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    assert args.atk_algo in ['pgd','fgsm','cw'],colored("Attack not supported",'red')
    assert args.dataset in ['cifar10','cifar100'],colored("Dataset not supported",'red')
    args.classes = 10 if args.dataset=='cifar10' else 100
    #args.target_names = args.target_models.split(',')
    #args.arch_names = args.target_arch.split(',')
    
    source_model,target_models = validate_and_build_models(args)
    
    if args.cuda:
        for i in range(len(target_models)):
            target_models[i][2] = torch.nn.DataParallel(target_models[i][2], device_ids=range(args.ngpu))
            target_models[i][2] = target_models[i][2].cuda()
        source_model = torch.nn.DataParallel(source_model, device_ids=range(args.ngpu))
        source_model.cuda()
    if verbose: 
        print(colored('Done Model Config','cyan'))
        print(colored('Configuring Data & Attacker','yellow'))

    test_data = get_data(test=True, batch_size=args.batch_size, dataset=args.dataset,download=False)

    if args.atk_algo != 'cw':
        attacker = ryax_attacker(
                epsilon=args.epsilon,
                steps=args.steps,
                model=source_model,
                iscuda=args.cuda,
                attack_algo=args.atk_algo,
                cifar=args.classes)

    else:
        attacker = Carlini_Wagner(
                epsilon=args.epsilon,
                steps=args.steps,
                cifar=args.classes,
                model=source_model,
                iscuda=args.cuda)

    attack_lambda = attacker.get_attack()
    if verbose: print(colored('Done Config','cyan'))
    
    with logger(args) as LOG:
        if verbose: print(colored("Starting test. Will generate and evaluate on the fly, in batches","yellow"))
        source_acc, target_models = transfer_test(
                                        source_model,
                                        target_models,
                                        test_data,
                                        attack_lambda,
                                        batch_size=args.batch_size,
                                        iscuda=args.cuda)
        
        delim = '---------------'
        msg = delim+'\n'+'SOURCE MODEL\nModel: {}\nArch: {}\nAcc: {}\n'.format(args.source_model,args.source_arch,source_acc)
        for f in range(len(target_models)):
            msg += delim+'\nTARGET {}\nModel: {}\nArch: {}\nAcc: {}\n'.format(f,target_models[f][0],target_models[f][1],target_models[f][3])
        LOG.write(msg)
    if verbose:print(colored('Done','green')) 

if __name__=='__main__':
    main()
