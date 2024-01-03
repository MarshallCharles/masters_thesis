import os, sys, torch, time, argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from termcolor import colored
from datetime import datetime
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.resnet_model.resnet_dense import ResNet34 as ResNet
from src.effnet_models.mobilenetv2 import MobileNetV2 as MobNet
from src.effnet_models.mobilenetv2_cifar100 import MobileNetV2 as MobnetCifar100
from src.effnet_models.shufflenetv2 import ShuffleNetV2 as ShufNet
from src.weigh_prune import l0proj, layers_nnz, layers_n, param_list
from src.attack import ryax_attacker,Carlini_Wagner
from src.evaluate import inference_test, acc_eps_test
from src.data_worker import get_data, select_gpu
from contextlib import contextmanager

verbose = False
default_eps = [round(x,2) for x in np.arange(0,1.01,0.05)]

def test_timer(f):
    def wrap(*args,**kwargs):
        time1 = time.time()
        ret = f(*args,**kwargs)
        time2 = time.time()
        return ret,time2-time1
    return wrap

@contextmanager
def logger(f):
    f = open(f,'a')
    f.write('\n{}\nTest starting at {}\n'.format('='*50,datetime.now()))
    try:
        yield f
    finally:
        f.write('END TEST\n{}'.format('='*50))
        f.close()

class bulk_tester:
    def __init__(self,parser,args):
        self.parser = parser
        self.args = args
        self.current_comm = None
        self.benign_model = None
        if args.dataset != 'None':
            self.load_data = lambda: get_data(test=True, batch_size=self.args.batch_size, dataset=self.args.dataset,download=False)
            self.data = self.load_data()
        if args.config[-4:]=='.pth':
            self.args.model = args.config
            self.model = self.load_model(self.args)
            if args.cuda:
                self.model_to_cuda() 
            self.attacker = ryax_attacker(epsilon=self.args.epsilon,
                    steps=self.args.attack_iterations,
                    model=self.model,
                    iscuda=self.args.cuda,
                    attack_algo=self.args.atk_algo)
            self.attack_lambda = self.attacker.get_attack()
            if self.args.acc_eps_test is not False:
                self.eps_attacker = ryax_attacker(epsilon=self.args.epsilon,
                        steps=self.args.attack_iterations,
                        model=self.model,
                        iscuda=self.args.cuda,
                        attack_algo=self.args.atk_algo,
                        set_elbow=True)
            if self.args.benign_model != 'None':
                self.benign_model = self.load_model(self.args,benign=True)
                if self.args.cuda:
                    self.benign_model_to_cuda()

        self.test_iteration = 0

    def run_experiment_return_string(self):
        ret = 'Model: {}\nData: {}\n'.format(self.args.config,self.args.dataset)
        if self.args.acc_test:
            if verbose: print(colored('Running inference on test set','yellow'))
            message,t = inference_test(self.model,self.data,self.args.batch_size,atk_algo=self.attack_lambda,
                    iscuda=self.args.cuda,iter_cap=self.args.test_set_iteration_cap,
                    comb_test = self.args.comb_benign_acc_test, benign_model = self.benign_model,speed_test_batches = self.args.speed_test_batches)
            ret += 'Running accuracy and speed test (with adversary too if specified)\n{}\nTotal elapsed time = {}s\n'.format(message,t)
        if self.args.acc_eps_test:
            if verbose: print(colored('Running epsilon elbow test','yellow'))
            message2,t2 = acc_eps_test(self.model,self.load_data,self.eps_attacker, iscuda=self.args.cuda,benign=self.doing_benign,bm=self.benign_model)
            ret += 'Running epsilon elbow test with pgd.\nRESULT: {}\nTotal elapsed time = {}s\n'.format(message2,t2) 
        self.test_iteration +=1
        return ret

    def tests_from_custom_config(self,cmd):
        self.setup_test_from_config_line(cmd)
        if verbose:
            s=''
            for item in cmd:
                s += item + ' '
            print(colored('Running: {}'.format(s),'yellow'))
        ret = self.run_experiment_return_string()
        return 'Running Command: {}\n{}'.format(self.current_comm,ret)

    def get_test_commands(self,config):
        with open(config,'r') as f:
            for line in f:
                self.current_comm = line.rstrip()
                command=[x[2:] if x[:2]=='--' else x for x in line.rstrip().split(' ')]
                if not ((os.path.isfile(os.path.join(os.getcwd(),command[0])))and(command[0][-4:]=='.pth')):
                    print("Received Command: {}\nSkipping this line (this may be perfectly fine, it could be a blank line, if not, check config)".format(self.current_comm))
                    continue
                yield command
    
    def setup_test_from_config_line(self,line):
        self.args.model = line[0]
        for k in list(self.args.__dict__):
            if k in line:
                self.args.__dict__[k] = type(self.parser.get_default(k))(line[line.index(k)+1])
        self.model = self.load_model(self.args)
        self.load_data = lambda: get_data(test=True, batch_size=self.args.batch_size, dataset=self.args.dataset,download=False)
        self.data = self.load_data()
        if self.args.cuda:
            self.model_to_cuda()
        classes = 10 if self.args.dataset.lower()=='cifar10' else 100
        if self.args.atk_algo is not None:
            if self.args.atk_algo.lower()=='cw':
                self.attacker = Carlini_Wagner(
                        epsilon=self.args.epsilon,
                        steps=self.args.attack_iterations,
                        cifar=classes,
                        model=self.model,
                        iscuda=self.args.cuda)
            else:
                self.attacker = ryax_attacker(
                        epsilon=self.args.epsilon,
                        steps=self.args.attack_iterations,
                        model=self.model,
                        iscuda=self.args.cuda,
                        attack_algo=self.args.atk_algo,
                        cifar=classes)

            self.attack_lambda = self.attacker.get_attack()
        
        if self.args.acc_eps_test is not False:
            if self.args.atk_algo.lower()=='cw':
                self.eps_attacker = Carlini_Wagner(
                        epsilon=self.args.epsilon,
                        steps=self.args.attack_iterations,
                        cifar=classes,
                        model=self.model,
                        iscuda=self.args.cuda,
                        set_elbow=True)
            else:
                self.eps_attacker = ryax_attacker(epsilon=self.args.epsilon,
                        steps=self.args.attack_iterations,
                        model=self.model,
                        iscuda=self.args.cuda,
                        attack_algo=self.args.atk_algo,
                        set_elbow=True)
        if self.args.comb_benign_acc_test: 
            self.benign_model = self.load_model(self.args,benign=True)
            if self.args.cuda:
                self.benign_model_to_cuda()
                self.doing_benign = True
        else:
            self.doing_benign = False
            self.benign_model = None


    def benign_model_to_cuda(self):
        self.benign_model = torch.nn.DataParallel(self.benign_model, device_ids=range(self.args.ngpu))
        self.benign_model.cuda()
    def model_to_cuda(self): 
        self.model = torch.nn.DataParallel(self.model, device_ids=range(self.args.ngpu))
        self.model.cuda()

    @staticmethod
    def load_model(args,benign=False):
        args.architecture = args.architecture.lower()
        classes = 10 if args.dataset=='cifar10' else 100
        assert args.architecture in ['resnet','mobilenet','shufflenet'],"Model Architecture not supported"
        if args.architecture=='resnet':
            model = ResNet(classes)
        elif args.architecture=='mobilenet':
            if classes==10:
                model = MobNet(alpha=args.effnet_alpha)
            else:
                model = MobnetCifar100()
        elif args.architecture=='shufnet':
            model = ShufNet(alpha=args.effnet_alpha,net_size=groups)
        if benign:
            model.load_state_dict(torch.load(args.benign_model), strict = False)
        else:
            model.load_state_dict(torch.load(args.model), strict = False)
        return model
    
def is_valid_file(parser,arg):
    assert os.path.isfile(os.path.join(os.getcwd(),arg)),colored('Config file does not exist\nExiting','red')
    return arg

def main():
    global verbose
    parser = argparse.ArgumentParser(description="Testing module by Charles Marshall for my research with Ryax Technologies")
    parser.add_argument('config',metavar='config_file',
            help = 'test config file, or a .pth file',
            type = lambda x: is_valid_file(parser,x))
    parser.add_argument('--log',default='models/model_test_log',help = 'log file for the test')
    parser.add_argument('--verbose',default=False,type=bool,help='print verbose') 
    parser.add_argument('--gpu', default="0", help="index of GPUs to use")
    parser.add_argument('--ngpu',default=1,type=int, help='number of GPUs to use')
    parser.add_argument('--seed',default=117,type=int,help="random seed (default: 117)")
    parser.add_argument('--dataset', default='None',type=str,help="Dataset to use")
    parser.add_argument('--batch_size',default=1,type=int,help='batch size for tests')
    parser.add_argument('--acc_test', default=False,type=bool,help='Test and report accuracy')
    parser.add_argument('--comb_benign_acc_test', default=False,type=bool,help='combine score with the benign model and current model')
    parser.add_argument('--benign_model',default='None',type=str)
    parser.add_argument('--adv_acc_test',default=False,type=bool,help='Test with perturbations, report accuracy')
    parser.add_argument('--atk_algo',default='None',type=str,help='Attack for the adversarial accuracy test')
    parser.add_argument('--epsilon',default=4.0,type=float,help='Epsilon for adversarial accuracy test')
    parser.add_argument('--attack_iterations',default=7,type=int,help='Amt of iterations for the atack')
    parser.add_argument('--acc_eps_test',default=False,type=bool,help='Find the elbow for the value of epsilon vs model accuracy')
    parser.add_argument("--architecture",default='resnet',type=str,help='Which architecture is the model specified in --model. Be careful of the dataset also')  
    parser.add_argument("--effnet_alpha",default=1.0,type=float,help='Alpha value for mobilenet')
    parser.add_argument("--shufflenet_groups",default=3,type=int,help="Group number for shufflenet")
    parser.add_argument("--test_set_iteration_cap",default=10000,type=int,help="Cap on the iteration number for the test")
    parser.add_argument("--speed_test_batches", default = 0, help = "This code is getting so sloppy... June 9th 2020")
    
    args = parser.parse_args()

    if args.verbose: verbose = True 
    args.gpu = select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    with logger(args.log) as f:
        test_manager = bulk_tester(parser,args)
        if args.config[-4:]=='.pth':
            f.write(test_manager.run_experiment_return_string())
        else:
            test_commands = test_manager.get_test_commands(args.config)
            for command in test_commands:
                f.write(test_manager.tests_from_custom_config(command))

if __name__=='__main__':
    main()
    
