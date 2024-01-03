import os, sys, torch, time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import RandomSampler, DataLoader
from matplotlib import pyplot as plt
from src.resnet_model.resnet_dense import ResNet34 as ResNet
from src.effnet_models.mobilenetv2 import MobileNetV2 as MobNet
from src.effnet_models.shufflenetv2 import ShuffleNetV2 as ShufNet
from src.data_worker import get_data, select_gpu

CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class Model_Interpreter:
    def __init__(self,**kwargs):
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        self.model = kwargs['model'] if 'model' in kwargs else None
        self.data_pulls = 0
        self.data = None
        self.classes = 10
        self.modelfiles = []

    def __str__(self):
        return "===Model Interpreter===\n\
                Model Files: {}\n\
                Data: {}\n\
                Amt Data Pulls: {}\n\
                Vebose: {}".format(self.modelfile,self.data,self.data_pulls)

    def load_data(self, batch_size = 1,dataset='cifar10'):
        self.dataset = dataset
        self.classes = 100 if dataset=='cifar100' else 10
        self.data = get_data(test=True, batch_size=batch_size, dataset=dataset)

    def load_model(self,f,arch, alpha=1.0,groups=3):
        self.modelfiles.append(f)
        if arch=='resnet':
            model = ResNet(self.classes)
        elif arch=='mobilenet':
            model = MobNet(alpha=alpha)
        elif arch=='shufnet':
            model = ShufNet(alpha=alpha,net_size=groups)
        else:
            print("Model Architecture not supported. From /src/imaging.py line 43")
            exit()
        model.load_state_dict(torch.load(f), strict = False)
        return model

    @staticmethod
    def tensor2image(img):
        return np.transpose(img.numpy(), (1, 2, 0))

    @staticmethod
    def show_img(img, name = None):
        plt.imshow(img)
        plt.title(name)
        plt.show()

    @staticmethod
    def attack_eps(rho):
        std = [1., 1., 1.]
        channels = [rho/255./s for s in std]
        return channels

    @staticmethod
    def attack_range():
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        channels = []
        for i in range(len(std)):
            channels.append([-mean[i]/std[i], (1-mean[i])/std[i]])
        return channels

    def tensor_clamp(self, t, mini, maxi, in_place=True):
        if not in_place:
            res = t.clone()
        else:
            res = t
            idx = res.data < mini
            res.data[idx] = mini[idx]
            idx = res.data > maxi
            res.data[idx] = maxi[idx]
        return res

    @staticmethod
    def edgecut(data, mini, maxi, in_place=True):
        if not in_place:
            res = data.clone()
        else:
            res = data
            idx = res.data < mini
            res.data[idx] = mini
            idx = res.data > maxi
            res.data[idx] = maxi
        return res

    def linfball_proj(self, center, radius, t, in_place=True):
        return self.tensor_clamp(t, mini=center - radius, maxi=center + radius, in_place=in_place)

    def predict(self, model, x, y, criterion=F.cross_entropy, ret = 'binary'):
        model.eval()
        with torch.no_grad():
            t1 = time.time()
            output = model(x).tolist()[0]
            t2 = time.time()
        if self.verbose:
            print("===Class Predictions===")
            print(*[CIFAR_CLASSES[idx]+': '+str(round(val,2)) for idx,val in enumerate(output)],sep='\n')
            print("Predicted Class: {}\nTime: {} seconds".format(CIFAR_CLASSES[output.index(max(output))],round(t2-t1,4)))
        if ret=='binary':
            return int(y.tolist()[0]==output.index(max(output)))
        elif ret=='timed':
            return t2-t1
        elif ret=='vector':
            return output
        else:
            return int(output.index(max(output)))

    def validation_test(self,model,ret = False,max_iter =np.inf,iscuda=False):
        stop = int(max_iter) if max_iter< len(list(self.data)) else 0
        correct = 0
        niter = 0
        for data, target in self.data:
            if iscuda:
                data,target=data.cuda(),target.cuda()
            correct += self.predict(model, data, target)
            niter +=1
            if niter==stop:
                break
        print("Test Complete\nAmt: {}\nAcc: {}".format(correct,correct/niter))
        if ret: return correct/niter

    def alt_atk(self,atk,model,data,target,rad,its):
        if atk=='fgsm':
            return self.fgsm(model,data,target,attack_radius=rad)
        else:
            return data
    
    def adv_validation(self,model,atk_algo='pgd',atk_rad=4,atk_iter=7,iscuda=False,criterion=F.cross_entropy): 
        std = torch.tensor([1., 1., 1.])
        l2dist, linfdist, niter, correct_adv = 0,0,0,0
        for data, target in self.data:
            indx_target = target.clone()
            if iscuda:
                data, target, std = data.cuda(), target.cuda(), std.cuda()
            if atk_algo=='pgd':
                data_adv = self.pgd(model, data, target, attack_radius=atk_rad, steps=atk_iter, iscuda=iscuda)
            else:
                data_adv = self.alt_atk(model,atk,data,target,atk_rad,atk_iter, iscuda=iscuda)
            with torch.no_grad():
                correct += self.predict(model,data_adv,target)
            niter+=1
            l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
            linfdist_channels = torch.max((data-data_adv).view(data.size(0), data.size(1), -1).abs(), dim=-1)[0] * std
            linfdist_temp = torch.max(linfdist_channels, dim=-1)[0]
            linfdist += linfdist_temp.sum().item()
        l2dist /= niter
        linfdist /= niter
        acc_adv = correct_adv/niter
        return acc_adv, "Adversarial validation set with {} attack, epsilon = {}, {} iterations\nAccuracy: {}\nAvg l2dist: {}\nAvg linfdist {}".format(atk_algo,atk_rad,atk_iter,acc_adv,l2dist,linfdist)

    def fgsm(self,model,x,y,attack_radius = 2,criterion=F.cross_entropy):
        eps = self.attack_eps(attack_radius)
        rgb = self.attack_range()
        x_adv = x.clone()
        x_adv.requires_grad = True
        loss_adv0 = criterion(model(x_adv), y, reduction='sum')
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        for i in range(len(eps)):
            alpha = eps[i]
            x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad0.data[:,i,:,:]))
            tmp = self.linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            x_adv[:,i,:,:].data.copy_(tmp.data)
            self.edgecut(x_adv[:,i,:,:], mini=rgb[i][0], maxi=rgb[i][1])
        return x_adv.data

    def pgd(self,model, x, y, attack_radius=2, steps=5, criterion=F.cross_entropy, randinit=True,iscuda=False):
        eps = self.attack_eps(attack_radius)
        eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
        rgb = self.attack_range()
        x_adv = x.clone() 
        if iscuda:
            x_adv = x_adv.cuda()
            eps_torch = eps_torch.cuda()
        x_adv.requires_grad = True
        if randinit:
            # pertub = torch.sign( torch.randn_like(x_adv) )
            x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch)
            for i in range(len(eps)):
                alpha = eps[i]
                # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
                self.linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
                self.edgecut(x_adv[:,i,:,:], mini=rgb[i][0], maxi=rgb[i][1])
        for num in range(steps):
            prev = x_adv.clone().detach()
            loss_adv = criterion(model(x_adv), y, reduction="sum")
            grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            with torch.no_grad():
                for i in range(len(eps)):
                    alpha = (eps[i] * 1.25) / steps
                    x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad.data[:,i,:,:]))
                    self.linfball_proj(center=x[:,i,:,:], radius=eps[i], t=x_adv[:,i,:,:])
                    self.edgecut(x_adv[:,i,:,:], mini=rgb[i][0], maxi=rgb[i][1])
                    # x_adv[:,i,:,:].data.fill_(torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1]))
        return x_adv.data

def main():
    IA = Model_Interpreter()
    IA.load_data(dataset='cifar10')
    mod = IA.load_model('TRAINED_MODELS/test1.pth')

if __name__=='__main__':
    main()
