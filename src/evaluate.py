import torch, time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.data_worker import get_data

ATTACKS = ['pgd','fgsm']

def test_timer(f):
    def wrap(*args,**kwargs):
        time1 = time.time()
        ret = f(*args,**kwargs)
        time2 = time.time()
        return ret,time2-time1
    return wrap

#default_eps = [round(x,2) for x in np.arange(0,1.01,0.05)]
#new_eps = [x for x in np.arange(0,0.5,.02)]

@test_timer
def acc_eps_test(model,load_data,attacker,iscuda=False,benign=False,bm=None):
    import statistics as st
    res = {} 
    for i in range(len(attacker.eps_test_eps)):
        ep_res = []
        for k in range(10):
            dataloader = load_data()
            correct = 0
            niter = 0 
            for data, target in dataloader:
                indx_target = target.clone()
                niter += data.shape[0]
                attack = attacker.get_attack()
                if attacker.name=='cw':
                    data_adv = attack(data) if i>0 else data.clone()
                else:
                    data_adv,atk_time = attack(model,data,target) if i>0 else (data.clone(),1)
                if iscuda:
                    data_adv = data_adv.cuda()
                    target = target.cuda()
                if benign is False:
                    worst_out,_ = single_predict(model,data_adv)
                    correct += acc_call(worst_out, indx_target)
                else:
                    correct += combined_predict(model,bm,data_adv,indx_target)
                
            ep_res.append(correct/niter) 
        avg = st.mean(ep_res)
        res[attacker.epsilon] = avg
        print("Eps: {}, Avg: {}".format(attacker.epsilon,avg))
        if res[attacker.epsilon] <= .10:
            return res
        if attacker.name=='cw':
            attacker.cw_elbow_step(model,i)
        else:
            attacker.elbow_test_step(model,i)

def acc_call(output, targ):
    pred = output.data.max(1)[1]
    correct = pred.cpu().eq(targ).sum().item()
    return correct

def validation_test(model,dataloader,atk_algo=None,iscuda=False,criterion=F.cross_entropy,iter_cap=np.inf):
    model.eval()
    std = torch.tensor([1., 1., 1.])
    correct_benign, correct_adv, loss_benign, loss_adv = 0,0,0,0
    l2dist, linfdist, niter = 0,0,0
    for data, target in dataloader:
        indx_target = target.clone()
        niter += data.shape[0]
        if iscuda:
            data, target = data.cuda(), target.cuda()
            std = std.cuda()

        with torch.no_grad():
            output_benign = model(data)
        correct_benign += acc_call(output_benign,indx_target)
        loss_benign += criterion(output_benign,target).data.item()

        if atk_algo==None:
            correct_adv = correct_benign
            loss_adv = loss_benign
        else:
            data_adv,trash = atk_algo(model, data, target)
            
            with torch.no_grad():
                output_adv = model(data_adv)
            correct_adv += acc_call(output_adv,indx_target)
            loss_adv += criterion(output_adv,target).data.item()
        l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
        linfdist_channels = torch.max((data-data_adv).view(data.size(0), data.size(1), -1).abs(), dim=-1)[0] * std
        linfdist_temp = torch.max(linfdist_channels, dim=-1)[0]
        linfdist += linfdist_temp.sum().item()
        if niter>=iter_cap:
            break
    loss_benign /= niter
    loss_adv /= niter
    l2dist /= niter
    linfdist /= niter
    acc_benign = correct_benign/niter
    acc_adv = correct_adv/niter
    return "Test Set:  avg loss: {},  accuracy: {}/{} = {}%\nAdv Set: avg loss: {},  accuracy: {}/{} = {}%,  L2dist: {},  Linf: {}"\
            .format(round(loss_benign,2),correct_benign,niter,round(acc_benign,2),round(loss_adv,2),correct_adv,niter,round(acc_adv,2),\
            round(l2dist,2),round(linfdist,2)), acc_benign, acc_adv


def correct_guess(tns,lab):
    x,y = tns.tolist()[0],int(lab)
    if int(x.index(max(x)))==lab:
        return 1
    return 0

@test_timer
def single_predict(model,data):
    return model(data)

def combined_predict(m,bm,data,lab):
    o1 = m(data) 
    o2 = bm(data)
    ans = o1 + o2
    pred = ans.data.max(1)[1]
    correct = pred.cpu().eq(lab).sum().item()
    return correct

@test_timer
def inference_test(model,dataloader, batch_size,atk_algo=None, iscuda=False, criterion=F.cross_entropy,iter_cap=10000,
        comb_test = False, benign_model = None, speed_test_batches = 0):
    
    if speed_test_batches != 0:
        assert type(speed_test_batches) is int
        stb_time = time.time()
        stb_times = 0
        stb_rounds = 1

    model.eval()
    std = torch.tensor([1., 1., 1.])
    correct_benign, correct_adv, loss_benign, loss_adv,correct_comb = 0,0,0,0,0
    l2dist, linfdist, niter = 0,0,0
    time_benign,time_adv,atk_time = 0,0,0
    for data, target in tqdm(dataloader,total=round((iter_cap-1)/batch_size)):
        indx_target = target.clone()
        niter += data.shape[0]
        if iscuda:
            data, target, std = data.cuda(), target.cuda(), std.cuda()
        with torch.no_grad():
            output_benign, tb = single_predict(model,data)
            time_benign += tb
        correct_benign += acc_call(output_benign,indx_target)
        loss_benign += criterion(output_benign,target).data.item()
        if atk_algo==None:
            correct_adv = correct_benign
            loss_adv = loss_benign
        else:
            if atk_algo.__qualname__[:7]=='Carlini':
                data_adv = atk_algo(data.cpu())
            else:
                data_adv,atk_time = atk_algo(model, data, target) 
            if iscuda:
                data_adv = data_adv.cuda()
            with torch.no_grad():
                output_adv,ta = single_predict(model,data_adv)
                time_adv += ta
                if comb_test:
                    out_comb_test = combined_predict(model,benign_model,data_adv,indx_target)
                    correct_comb += out_comb_test
            correct_adv += acc_call(output_adv,indx_target)
            loss_adv += criterion(output_adv,target).data.item()
            l2dist += torch.norm((data - data_adv).view(data.size(0), -1), p=2, dim=-1).sum().item()
            linfdist_channels = torch.max((data-data_adv).view(data.size(0), data.size(1), -1).abs(), dim=-1)[0] * std
            linfdist_temp = torch.max(linfdist_channels, dim=-1)[0]
            linfdist += linfdist_temp.sum().item()

        if speed_test_batches != 0 :
            if niter >= (speed_test_batches * stb_rounds):
                stb_times += time.time() - stb_time
                stb_rounds += 1
                stb_time = time.time()
        if niter >= iter_cap:
            break
    if speed_test_batches != 0:
        print("DONE, average time for {} samples = {}".format(speed_test_batches,stb_times / stb_rounds))

    correct_comb /= niter
    loss_benign /= niter
    loss_adv /= niter
    l2dist /= niter
    linfdist /= niter 
    time_benign /= niter
    time_adv /= niter
    acc_benign = correct_benign/niter
    acc_adv = correct_adv/niter
    atk_time/= niter
    if atk_algo != 'None':
        return "Benign acc: {}\nBenign avg loss: {}\nBenign avg time: {}\nAdv acc: {}\nAdv avg loss: {}\nAdv avg time: {}\nCombined Acc: {}\nAvg l2 dist: {}\nAvg linf dist: {}\nAttack avg time: {}".format(\
                acc_benign,loss_benign,time_benign,acc_adv,loss_adv,time_adv,correct_comb,l2dist,linfdist,atk_time)
    return "Benign acc: {}\nBenign avg loss: {}\nBenign avg time: {}\n".format(acc_benign,loss_benign,time_benign) 


'''
def confidence_distance(vec,idx):
    if vec[idx] == max(vec):
        return min([vec[idx]-x for x in vec if x!=vec[idx]])
    else:
        return vec[idx]-max(vec)

def get_worst_example_pgd(model,data,label,eps,iscuda=False):
    for i in range(4096):
        example = pgd(model,data,label,attack_radius=eps,steps=i+1,iscuda=iscuda)
        if iscuda:
            example, label = example.cuda(), label.cuda()
        v,_ = single_predict(model,example)
        conf = confidence_distance(v.tolist()[0],int(label))
        if i==0:
            prev=conf
        elif (round(prev,2)-round(conf,2))<=0:
            print("Found worst example at i={}".format(i))
            return example
        prev=conf

def predict(model, x, y, criterion=F.cross_entropy, ret='binary', verbose=False):
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        o1 = model(x)
        t2 = time.time()
        output = o1.tolist()[0]
    if verbose:
        print("===Class Predictions===")
        print(*[CIFAR_CLASSES[idx]+': '+str(round(val,2)) for idx,val in enumerate(output)],sep='\n')
        print("Predicted Class: {}\nTime: {} seconds".format(CIFAR_CLASSES[output.index(max(output))],round(t2-t1,4)))
    if ret=='binary':
        return int(y.tolist()[0]==output.index(max(output)))
    elif ret=='timed':
        return t2-t1
    elif ret=='training':
        return int(output.index(max(output))), criterion(o1, y).data.item()
    else:
        return int(output.index(max(output)))

def alt_atk(name,model,data,target,rad,it,criterion=F.cross_entropy,iscuda=False):
    if name=='fgsm':
        return fgsm(model,data,target,attack_radius=rad,criterion=criterion,iscuda=iscuda)
    else:
        return None
'''
