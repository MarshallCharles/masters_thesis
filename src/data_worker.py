import torch, sys, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from termcolor import colored

file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path,os.pardir))
DATAPATH =  os.path.join(os.path.join(file_path,os.pardir),'data')
DATASETS = {'cifar10':datasets.CIFAR10,'cifar100':datasets.CIFAR100}

def config_dataset(d,train,dwnld = True):
    d_root = os.path.join(DATAPATH,d)
    if train:
        return DATASETS[d](root=d_root,\
            train=True, download=dwnld,\
            transform=transforms.Compose([
            transforms.Pad(4),\
            transforms.RandomCrop(32),\
            transforms.RandomHorizontalFlip(),\
            transforms.ToTensor(),\
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
    else:
        return DATASETS[d](root=d_root,\
            train=False, download=dwnld,\
            transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))

def get_data(train = False, test = False, batch_size = 1, dataset = 'cifar10',download = True):
    if dataset.lower() not in DATASETS:
        print(colored("Error, dataset not supported",'red'))
        exit()
    if ((train and test) or (not train)and(not test)):
        print(colored("Error, choose train or test data once at a time",'red'))
        exit()
    else:
        return torch.utils.data.DataLoader(\
            config_dataset(dataset,train,download),\
            batch_size=batch_size, shuffle=test)

def select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]
        if len(ideal_gpus) < num_gpu:
            print(colored("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu),'red'))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

if __name__=='__main__':
    print("Testing to make sure this module can retrieve the proper data")
    data = get_data(train = True, batch_size = 50, dataset = 'cifar10')
    print("Result: {}".format(data))
