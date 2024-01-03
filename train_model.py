import argparse, pickle, os, time, sys, datetime, re, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.weigh_prune import l0proj, layers_nnz, layers_n, param_list, calc_model_sparsity
from src.attack import ryax_attacker
from src.evaluate import validation_test
from src.data_worker import get_data, select_gpu
import numpy as np
from src.resnet_model.resnet_dense import ResNet34 as ResNet
from src.effnet_models.mobilenetv2 import MobileNetV2 as MobNet
from src.effnet_models.mobilenetv2_cifar100 import MobileNetV2 as MobnetCifar100
from src.effnet_models.shufflenetv2 import ShuffleNetV2 as ShufNet
from termcolor import colored

LOG = None
weight_name = ["weight"]
supported_models = ['resnet','mobilenet','shufflenet']
verbose = False
STEPS_LIST = None

def set_log(pth):
    global LOG
    LOG = pth+'training_log' if pth[-1]=='/' else pth+'/training_log'
    if not os.path.isdir(pth):
        os.mkdir(pth)
    with open(LOG,'a') as f:
        f.write("Test starting at {}\n\n".format(datetime.datetime.now()))
        for i in range(3):
            f.write("="*50)
            f.write("\n")

def write_to_log(text):
    if verbose: print(text)
    with open(LOG,'a') as f:
        f.write(text)
        f.write('\n')

def params_to_log(args):
    t="===============TEST PARAMETERS==============="
    with open(LOG,'a') as f:
        f.write(t+'\n')
        for k,v in args.__dict__.items():
            f.write("{}: {}\n".format(k,v))
        f.write("="*50)
        f.write("\n\n")
    if verbose:
        print(t)
        for k,v in args.__dict__.items():
            print("{}: {}".format(k,v))
        print("="*50)

def save_model(model, name, pth):
    from collections import OrderedDict
    if name is None:
        name = pth + 'NEWMODEL.pth' if pth[-1]=='/' else pth+'/NEWMODEL.pth'
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, name)

def get_trainer_model(args):
    arch_type = args.architecture.lower()
    n_classes = 10 if args.dataset=='cifar10' else 100
    assert arch_type in supported_models,colored("Model architecture not supported. Exiting","red")
    if arch_type == 'resnet':
        model = ResNet(n_classes)
    elif arch_type == 'mobilenet':
        if n_classes==10:
            model = MobNet(alpha=args.effnet_alpha,num_classes=n_classes)
        else:
            model = MobnetCifar100()
    elif arch_type == 'shufnet':
        model = ShufNet(net_size=args.effnet_alpha)
    if ((not args.raw_train) and (args.model != None)):
        model.load_state_dict(torch.load(args.model), strict = False)
    return model

def ensure_len(arr,l):
    while len(arr)!=l:
        if len(arr)>l:
            del arr[-1]
        elif len(arr)<l:
            arr.append(arr[-1])
    return arr

def acc_call(output, target, type="vanilla"):
    pred = output.data.max(1)[1]
    correct = pred.cpu().eq(target).sum().item()
    acc = correct * 1.
    return acc

def train(model, epoch, data_loader,optimizer,defense_atk=None,\
    log_interval=10, iscuda=False,criterion=F.cross_entropy,prune_algo=None,\
    proj_interval=1,layernnz=None, param_list=None,adv_coeff = 0.5):
    model.train()
    loss_avg, acc_avg,nb_data = 0., 0.,0
    for batch_idx, (data, target) in enumerate(data_loader):
        nb_data += len(data)
        indx_target = target.clone()
        if iscuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if defense_atk is None:
            data_adv = data
            output_adv = model(data_adv)
            loss = criterion(output_adv, target)
        else:
            data_adv,trash = defense_atk(model,data,target)
            if iscuda:
                data_adv = data_adv.cuda()
            output_adv = model(data_adv)
            output = model(data)
            loss = ((1-adv_coeff) * criterion(output, target)) + (adv_coeff * criterion(output_adv, target))
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        if batch_idx % proj_interval == 0:
            with torch.no_grad():
                if prune_algo is not None:
                    prune_algo(model)
        acc = acc_call(output_adv, indx_target)
        acc_avg += acc
        msg = 'Train Epoch: {} [{}/{}] Loss: {:.5f} Acc: {:.4f} lr: {:.2e}'.format(
            epoch,batch_idx*len(data),len(data_loader.dataset),\
            loss_avg/nb_data, acc_avg/nb_data,optimizer.param_groups[0]['lr'])
        if batch_idx%log_interval==0:
            write_to_log(msg)
    if prune_algo is not None:
        prune_algo(model)
        total_elems, nonzero_elems = calc_model_sparsity(model)
        write_to_log('Total Parameters: {}\nZero Parameters: {}\nCompression Ratio: {}'.format(\
                        total_elems,total_elems-nonzero_elems,nonzero_elems/total_elems))
    return acc_avg/nb_data

def main():
    global verbose
    parser = argparse.ArgumentParser(description="Training Module by Charles Marshall for my research with Ryax Technologies")
    parser.add_argument("--architecture",default='resnet', help='Model architecture. For now mobilenet or resnet')
    parser.add_argument("--effnet_alpha",type=float, default=1.0,help='Alpha value for mobilenet')
    parser.add_argument("--shufflenet_groups",type=int,default=3,help="Group number for shufflenet")
    parser.add_argument("--raw_train", action="store_true")
    parser.add_argument("--model", default=None,help="model to continue training")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay factor")
    parser.add_argument('--learning_rate', type=float, default=5e-2, help='learning rate (default: 1e-3)')
    parser.add_argument('--decreasing_lr', default='80,120,150', help="decreasing strategy")
    parser.add_argument('--gpu', default="0", help="index of GPUs to use")
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--seed', type=int, default=117, help="random seed (default: 117)")
    parser.add_argument('--test_interval', type=int, default=1, help="how many epochs to wait before another test")
    parser.add_argument('--save_dir', default='default_test_data/', help="folder to save the log")
    parser.add_argument('--dataset', default='cifar10', help="dataset to use for training")
    parser.add_argument('--log_interval', type=int, default=20, help="how many batches to wait before logging training status")
    parser.add_argument('--ada_train_attack',action='store_true',help='use adaptive-train-adaptive-attack')
    parser.add_argument('--start_epsilon',type=float,default=0.0,help='Initial epsilon value')
    parser.add_argument('--retain_adv_loss',type=float,default=0.0,help='Retain the same adversarial loss')
    parser.add_argument('--maximum_epsilon',default=16.,type=float,help='Maximum epsilon value')
    parser.add_argument('--defense_algo', default=None, help='adversarial algo for defense')
    parser.add_argument('--defense_radius', type=float, default=None, help='perturbation radius for defend phase')
    parser.add_argument('--defense_iterations', type=int, default=8, help="defend iteration for the adversarial sample computation")
    parser.add_argument('--attack_algo', default='pgd', help="attack algorithm for testing")
    parser.add_argument('--attack_radius', type=int, default=4, help="attack radius for testing")
    parser.add_argument('--attack_iterations', type=int, default=8, help="attack iterations for testing")
    parser.add_argument('--prune_algo', default=None, help="pruning method")
    parser.add_argument('--prune_ratio', type=float, default=1, help='sparsity ratio from 0-1')
    parser.add_argument('--prune_interval', type=int, default=2, help='Amt of epochs between pruning')
    parser.add_argument('--verbose', action='store_true', help='print verbose')
    args = parser.parse_args()

    if args.verbose: verbose = True
    set_log(args.save_dir)
    
    if verbose:
        print(colored("Configuring Hardware",'yellow'))

    args.gpu = select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if verbose:
        print(colored("Done",'green'))
        print(colored('Assembling Model and Data','yellow'))

    train_loader = get_data(train=True, batch_size=args.batch_size, dataset=args.dataset)
    test_loader = get_data(test=True, batch_size=args.batch_size, dataset=args.dataset)
    model = get_trainer_model(args)

    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
        model.cuda()

    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,\
                    weight_decay=args.weight_decay, momentum=0.9)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    params_to_log(args)
    
    adv_coeffs = [0.5]*args.epochs
    defense_lambda = None
    if verbose: 
        print(colored('Done','green'))
        print(colored('Configuring Attacker/Defender','yellow'))

    attacker = ryax_attacker(
            attack_algo=args.attack_algo,
            epsilon=args.attack_radius,
            steps=args.attack_iterations,
            iscuda=args.cuda,
            set_adaptive=False)
    attack_lambda = attacker.get_attack()
    
    if args.defense_algo != None:
        defender = ryax_attacker(
                epochs=args.epochs,
                max_epsilon=args.maximum_epsilon,
                epsilon=args.defense_radius,
                steps=args.defense_iterations,
                model = model,
                iscuda=args.cuda,
                attack_algo=args.defense_algo,
                set_adaptive=args.ada_train_attack,
                initial_epsilon = args.start_epsilon,
                step_interval=3,
                start_steps=4,
                step_adder=4)

        if args.ada_train_attack: 
            adv_coeffs = [x for x in np.arange(0.5,1,0.5/args.epochs)]
            adv_coeffs = ensure_len(adv_coeffs,args.epochs) 
        
        defense_lambda = defender.get_attack()

    if args.retain_adv_loss != 0.0:
        adv_coeffs = [args.retain_adv_loss]*(args.epochs+1)

    if verbose: 
        print(colored('Done','green'))
        print(colored('Config done, starting an initial test','yellow'))
    statement,acc_benign,acc_adv = validation_test(model, test_loader,atk_algo=attack_lambda,iscuda=args.cuda)
    write_to_log("Pre Training Score:\n{}".format(statement))

    if verbose: 
        print(colored('Done','green'))
        print(colored('Configuring Pruning Scheme (if applicable)','yellow'))

    adaprune = False
    layers = layers_nnz(model, param_name=weight_name)[1]
    all_num = sum(layers_n(model, param_name=["weight"])[1].values())
    sparse_factor = int(all_num * args.prune_ratio)
    write_to_log("Model Size: {}".format(sparse_factor * 32))
    layernnz = lambda m: list(layers_nnz(m, param_name=weight_name)[1].values())
    param_list = lambda m: param_list(m, param_name=weight_name)
    print(all_num)
    if ((args.prune_algo == "l0proj") and (args.prune_ratio<1)):
        prune_lambda = lambda m: l0proj(m, sparse_factor, normalized=False, param_name=weight_name)
    elif ((args.prune_algo == "adal0proj") and (args.prune_ratio<1)):
        total_elems, nonzero_elems = calc_model_sparsity(model)
        compr_ratio = round(nonzero_elems/total_elems,7)
        adaprune=True
        decreasing_prune_ratio = [x for x in np.arange(args.prune_ratio,compr_ratio,(compr_ratio-args.prune_ratio)/(args.epochs/args.prune_interval))]
        while len(decreasing_prune_ratio)>args.epochs:
            del decreasing_prune_ratio[-1]
    else:
        prune_lambda = None

    if verbose:
        print(colored('Done','green'))
        print(colored("Begin Training Procedure", "cyan"))
    
    '''TRAINING LOOP'''

    BEGIN_TIME = time.time()
    AVG_TIME_PER_EPOCH,best_training_accuracy,best_testing_acc,best_dist = 0,0.,0,np.inf
    for epoch in range(args.epochs):
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        if adaprune:
            sparse_factor = int(all_num * decreasing_prune_ratio[-(epoch+1)])
            prune_lambda = lambda m: l0proj(m,sparse_factor,normalized=False,param_name=weight_name)

        train_accuracy = train(model, epoch, train_loader, optimizer,\
                defense_atk=defense_lambda,log_interval=args.log_interval,\
                iscuda=args.cuda,param_list=param_list, layernnz=layernnz,\
                criterion=F.cross_entropy,prune_algo=prune_lambda,\
                proj_interval=args.prune_interval,adv_coeff=adv_coeffs[epoch])

        if args.ada_train_attack:
            defender.train_step(model,epoch)
            defense_lambda = defender.get_attack()
            write_to_log('Attack Updated\nEpsilon = {}\nSteps = {}\nAdv Coeff = {}'.format(defender.epsilon,defender.steps,adv_coeffs[epoch]))

        total_elapse_time = time.time() - BEGIN_TIME
        speed_epoch = total_elapse_time / (epoch + 1)
        estimated_total = speed_epoch * args.epochs

        if verbose: print("Epoch {} done, epochs averaging {} sec. Total time passed: {}, estimated total: {}".\
                    format(epoch+1,speed_epoch, total_elapse_time, estimated_total))
        
        if train_accuracy>best_training_accuracy:
            temp_name ='latest_'+args.model if args.model!=None else 'latest_model.pth'
            save_model(model,temp_name,args.save_dir)
            best_training_accuracy = train_accuracy
        
        if epoch % args.test_interval==0:
            statement,acc_benign,acc_adv = validation_test(model, test_loader,atk_algo=attack_lambda,iscuda=args.cuda)
            write_to_log("Epoch {} score:\n{}".format(epoch+1,statement))
            
            if acc_adv > best_testing_acc:
                best_testing_acc = acc_adv
                save_model(model,args.model,args.save_dir)
        save_model(model,'most_recent_model.pth',args.save_dir) 
    write_to_log("Total elapsed time: {}sec".format(time.time()-BEGIN_TIME))

if __name__=='__main__':
    main()
