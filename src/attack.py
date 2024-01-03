import torch, time
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from art.attacks.evasion import CarliniL2Method, CarliniLInfMethod
from art.classifiers import PyTorchClassifier

def timer(f):
    def wrap(*args,**kwargs):
        time1 = time.time()
        ret = f(*args,**kwargs)
        time2 = time.time()
        return ret,time2-time1
    return wrap


class ryax_attacker:
    default_attrs = {
            'epochs':None,
            'max_epsilon':16,
            'rads':None,
            'epsilon':4,
            'steps':7,
            'model':None,
            'criterion':F.cross_entropy,
            'iscuda':False,
            'attack_algo':'pgd',
            'set_adaptive':False,
            'step_interval':3,
            'start_steps':4,
            'initial_epsilon':1.0,
            'step_adder':4,
            'train_rads':None,
            'set_elbow':False,
            'eps_test_eps':[x for x in np.arange(0,0.5,.02)],
            'cifar':10
            }
    
    def __init__(self,**kwargs):
        for k,v in ryax_attacker.default_attrs.items():
            if k in kwargs:
                setattr(self,k,kwargs[k])
            else:
                setattr(self,k,ryax_attacker.default_attrs[k])
        if self.set_adaptive:
            self.set_step_scheme()
        if self.set_elbow:
            self.epsilon=self.eps_test_eps[0]
        if self.attack_algo in ['pgd','fgsm']:
            self.name = self.attack_algo
        else:
            self.name = 'cw'
   
    def set_step_scheme(self):
        self.train_rads=[x for x in np.arange(self.initial_epsilon,self.max_epsilon,(self.max_epsilon-self.initial_epsilon)/(self.epochs/3))]
        self.train_rads.append(self.max_epsilon)
        self.epsilon = self.train_rads[0]
        self.steps = self.start_steps

    def train_step(self,model,epoch):
        self.model = model
        if self.train_rads is not None:
            if ((epoch>0)and(epoch%self.step_interval==0)):
                self.epsilon=self.train_rads[self.train_rads.index(self.epsilon)+1]
                self.steps=self.start_steps
            else:
                self.steps+=4

    def elbow_test_step(self,model,niter):
        self.epsilon = self.eps_test_eps[niter]*255

    def get_attack(self):
        if self.attack_algo=='pgd':
            return lambda m,d,t: self.pgd(m,d,t,randinit=True)
        elif self.attack_algo=='fgsm':
            return lambda m,d,t: self.fgsm(m,d,t)
        else:
            return None
            
    @timer
    def fgsm(self,model,x,y):
        eps = self.attack_eps(self.epsilon)
        rgb = self.attack_range()
        x_adv = x.clone()
        x_adv.requires_grad = True
        if self.iscuda:
            x_adv=x_adv.cuda()
        loss_adv0 = self.criterion(model(x_adv), y, reduction='sum')
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        for i in range(len(eps)):
            alpha = eps[i]
            x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad0.data[:,i,:,:]))
            tmp = self.linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
            x_adv[:,i,:,:].data.copy_(tmp.data)
            self.edgecut(x_adv[:,i,:,:], mini=rgb[i][0], maxi=rgb[i][1])
        return x_adv.data

    @timer
    def pgd(self,model, x, y, randinit=True):
        eps = self.attack_eps(self.epsilon)
        eps_torch = torch.tensor(eps).view(1, len(eps), 1, 1)
        rgb = self.attack_range()
        x_adv = x.clone()
        if self.iscuda:
            eps_torch,x,y,x_adv= eps_torch.cuda(),x.cuda(),y.cuda(),x_adv.cuda()
        x_adv.requires_grad = True
        if randinit:
            # pertub = torch.sign( torch.randn_like(x_adv) )
            x_adv.data.add( (2.0 * torch.rand_like(x_adv) - 1.0) * eps_torch )
            for i in range(len(eps)):
                alpha = eps[i]
                # x_adv[:,i,:,:].data.add_(alpha * pertub[:,i,:,:])
                self.linfball_proj(center=x[:,i,:,:], radius=alpha, t=x_adv[:,i,:,:])
                self.edgecut(x_adv[:,i,:,:], mini=rgb[i][0], maxi=rgb[i][1])
        for _ in range(self.steps):
            loss_adv = self.criterion(model(x_adv), y, reduction="sum")
            grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            with torch.no_grad():
                for i in range(len(eps)):
                    alpha = (eps[i] * 1.25) / self.steps
                    x_adv[:,i,:,:].data.add_(alpha * torch.sign(grad.data[:,i,:,:]))
                    self.linfball_proj(center=x[:,i,:,:], radius=eps[i], t=x_adv[:,i,:,:])
                    self.edgecut(x_adv[:,i,:,:], mini=rgb[i][0], maxi=rgb[i][1])
                    # x_adv[:,i,:,:].data.fill_(torch.clamp(x_adv[:,i,:,:], min=rgb[i][0], max=rgb[i][1]))
        return x_adv.data
    
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

    @staticmethod
    def linfball_proj(center, radius, t, in_place=True,iscuda = False):
        def tensor_clamp(t,mini,maxi,flag=True):
            if not flag:
                res = t.clone()
            else:
                res = t
                idx = res.data < mini
                res.data[idx] = mini[idx]
                idx = res.data > maxi
                res.data[idx] = maxi[idx]
            return res
        return tensor_clamp(t,
                center - radius,
                center + radius, 
                flag=in_place)

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

class Carlini_Wagner(ryax_attacker):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        assert self.model is not None,'Must specify a model for CW attack'
        self.set_model()
        self.name = 'cw'
         
    def set_model(self,**kwargs):
        self.model = kwargs['model'] if 'model' in kwargs else self.model
        if self.iscuda is True:
            self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.classifier = PyTorchClassifier(
                loss = self.criterion,
                optimizer = self.optimizer,
                model = self.model,
                input_shape = (3, 32, 32),
                nb_classes = self.cifar)

    def get_attack(self):
        CWattack = CarliniLInfMethod(self.classifier)
        CWattack.max_iter = self.steps
        CWattack.eps = self.epsilon
        return lambda d: CWattack(x=d)
    
    def cw_elbow_step(self,model,niter):
        self.epsilon = self.eps_test_eps[niter]*255
