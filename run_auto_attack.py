import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from src.resnet_model.resnet_dense import ResNet34 as ResNet
from src.effnet_models.mobilenetv2 import MobileNetV2 as MobNet
from src.effnet_models.mobilenetv2_cifar100 import MobileNetV2 as MobnetCifar100
import sys
sys.path.insert(0,'..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/cifar10')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--cheap', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./AAttack_results')
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--plus', action='store_true')
    parser.add_argument('--log_path', type=str, default='./AALog.txt')

    args = parser.parse_args()

    # load model
    model = ResNet()
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # cheap version
    if args.cheap:
        adversary.cheap()

    # plus version
    if args.plus:
        adversary.plus = True

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.batch_size)

            torch.save({'adv_complete': adv_complete}, '{}/{}_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', adv_complete.shape[0], args.epsilon, args.plus, args.cheap))

        else:
            # individual version, each attack is run on all test points
            # specify attacks to run with
            # adversary.attacks_to_run = ['apgd-ce']
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.batch_size)

            torch.save(adv_complete, '{}/{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
            args.save_dir, 'aa', args.n_ex, args.epsilon, args.plus, args.cheap))