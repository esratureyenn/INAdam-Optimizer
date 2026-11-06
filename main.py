import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from read_data import read_cifar10
from models import *
from INAdam import INAdam
from adam import Adam
import numpy as np
from tqdm import tqdm
from train import train_loop
from test import test_loop
import torchvision.transforms as transforms, torchvision


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-m', '--model', default='resnet18', type=str,
                    dest='model',  help='Which model to use. Options are resnet18(default), resnet50, resnet_flat, customcnn, vit')
parser.add_argument('-lr', '--learning_rate', default=3e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=5, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--norm_t', default=1, type=int, help='norm_t')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--nsgd', type=int, default=3, dest='nsgd',
                    help='Which optimizer to use. 1 for SGD, 2 for Adam, 3 for INAdam, 4 for Adam from Scratch')
parser.add_argument('--device', type=str, default='mps', help='It can be mps(macbook), cuda(windows,colab) and cpu')




def main():
    
    args = parser.parse_args()    

    main_worker(args)


def main_worker(args):

    seed = args.seed
    device = args.device
    save_dir = args.save_dir
    batch_size = args.batch_size
    n_epochs = args.epochs
    weight_decay  = args.weight_decay
    momentum = args.momentum
    model_type = args.model
    nsgd = args.nsgd
    learning_rate = args.lr
    norm_t = args.norm_t

    def my_seed_every(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        random.seed(seed)
        np.random.seed(seed)
        #torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)



    dataset= "cifar10"
    data_dir = 'data'
    transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # CIFAR-10 dataloader generator
    train_generator,test_generator = read_cifar10(batch_size,data_dir, my_seed_every,seed,False) 
    """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)
    my_seed_every(seed=0)
    train_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    """

    # Model type
    my_seed_every(seed=0)
    if model_type == 'resnet18':
        from models.resnet import resnet18
        model = resnet18().to(device)
    elif model_type == 'resnet50':
        from models.resnet import resnet50
        model = resnet50().to(device)
    elif model_type == 'resnet_flat':
        from models.resnet_flat import ResNet20_Flat
        model = ResNet20_Flat().to(device)
    elif model_type == 'customcnn':
        from models.customcnn import SimpleModel_Cifar
        model = SimpleModel_Cifar().to(device)
    elif model_type == 'vit':
        from models.vit import ViT
        model = ViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=4, heads=6, mlp_dim=512).to(device)
    '''
    TO DO (Rest of the Models)
    '''

    #print(model)
    print("Device is: ",device)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    #print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    
    inputs = {}
    
    def get_inputs(name):
        def hook(model, input, output):
            inputs[name] = input[0].detach()
        return hook
      
    for name,module in model.named_modules():
        for param_name, param in module.named_parameters():
            param.input_to_norm = None
        if name != '' and isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_hook(get_inputs(name))
    

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    if nsgd == 1:
        opt = 'sgd'
    elif nsgd == 2:
        opt = 'adam'
    elif nsgd == 3:
        opt = 'inadam'
    else:
        opt= 'classic_adam'
    
    if nsgd == 1:
        print('Optimizer is SGD')
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,
                                     momentum=momentum,
                                     weight_decay=weight_decay)
    
    elif nsgd == 2:
        print('Optimizer is Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        
    elif nsgd == 3:
        print('Optimizer is Input Normalized Adam')
        optimizer = INAdam(model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay, norm_t=norm_t)  
    elif nsgd == 4:
        print('Optimizer is Classic Adam')
        optimizer = Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)  


    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma,last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    idx_best_loss = 0
    idx_best_acc = 0
    
    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []

    for epoch in range(1, n_epochs+1):
        print("===> Epoch {}/{}".format(epoch, n_epochs))
        start_time = time.time()
        train_loss, train_acc = train_loop(train_generator, model, criterion, optimizer, device, inputs)
        test_loss, test_acc = test_loop(test_generator, model, criterion, device)
        time_spent = time.time() - start_time
        print("Time spent:{:f}".format(time_spent))
        print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        print("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))

        #scheduler.step()
        
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss)
        list_test_acc.append(test_acc)
        
        if test_loss <= list_test_loss[idx_best_loss]:
            print("Loss-best model.")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'loss_model.pth'))
            idx_best_loss = epoch - 1
        
        if test_acc >= list_test_acc[idx_best_acc]:
            print("Acc-best model.")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'acc_model.pth'))
            if test_acc > list_test_acc[idx_best_acc]:
                idx_best_acc = epoch - 1    
        #print("")
        
    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(list_train_loss[idx_best_loss], list_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_loss], list_test_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(list_train_loss[idx_best_acc], list_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_acc], list_test_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(list_train_loss[-1], list_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc: {:f}".format(list_test_loss[-1], list_test_acc[-1]))           
    #torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    
    list_train_loss = np.array(list_train_loss)
    list_train_acc = np.array(list_train_acc)
    list_test_loss = np.array(list_test_loss)
    list_test_acc = np.array(list_test_acc)

    print(list_test_acc)
    
    torch.save(list_train_loss, os.path.join(save_dir + '/' + dataset + '/' + opt , 'loss_train.pt'))
    torch.save(list_train_acc , os.path.join(save_dir + '/' + dataset + '/' + opt, 'acc_train.pt'))
    torch.save(list_test_loss, os.path.join(save_dir + '/' + dataset + '/' + opt, 'loss_test.pt'))
    torch.save(list_test_acc, os.path.join(save_dir + '/' + dataset + '/' + opt, 'acc_test.pt'))

    torch.save(model.state_dict(), save_dir + '/' + dataset+'/' + opt + '/model_state_dict.pth')



if __name__ == '__main__':
    main()
