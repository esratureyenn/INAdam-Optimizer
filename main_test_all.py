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
from read_data import read_cifar10, read_cifar100
from models import *
from INAdam import INAdam
from adam import Adam
import numpy as np
from tqdm import tqdm
from train import train_loop
from test import test_loop
import torchvision.transforms as transforms, torchvision
from functions import save_args


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-m', '--model', default='resnet_flat', type=str,
                    dest='model',  help='Which model to use.  Options are resnet18(default), resnet50, resnet_flat, customcnn, vit')
parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--norm_t', default=2, type=int, help='norm_t')
parser.add_argument('--dataset', default=1, type=int, help='0 for cifar10, 1 for cifar100')
parser.add_argument('--main_dir', type=str, default='./Results')
parser.add_argument('--nsgd', type=int, default=1, dest='nsgd',
                    help='Which optimizer to use. 1 for SGD, 2 for Adam, 3 for INAdam, 4 for Adam from Scratch')
parser.add_argument('--device', type=str, default='mps', help='It can be mps(macbook), cuda(windows,colab) and cpu')




def main():
    
    args = parser.parse_args()    
    


    seed_list = 0

    insgd_list = [1, 3, 4]
    norm_t_list = [1, 2, 4]
    lr_list = [0.003, 0.001, 0.0003, 0.0001]
    batch_list = [32, 64, 128]


    args.nsgd = 1
    args.lr = 0.003
    args.norm_t = 1
    args.batch_size = 128

    main_worker(args)




    for insgd_step in range(len(insgd_list)):
        for norm_step in range(len(norm_t_list)):
            for lr_step in range(len(lr_list)):
                for batch_step in range(len(batch_list)):
                    nsgd = insgd_list[insgd_step]
                    norm_t = norm_t_list[norm_step]
                    learning_rate = lr_list[lr_step]
                    batch_size = batch_list[batch_step]

                    seed = seed_list
                    #os.system('python main.py  --nsgd {} --norm_t {} --batch_size {} --learning_rate {}'.format(str(nsgd), str(norm_t), str(batchsize), str(lr)))
                    args.nsgd = nsgd
                    args.lr = learning_rate
                    args.norm_t = norm_t
                    args.batch_size = batch_size
                    main_worker(args)


def main_worker(args):

    seed = args.seed
    device = args.device
    main_dir = args.main_dir
    batch_size = args.batch_size
    n_epochs = args.epochs
    weight_decay  = args.weight_decay
    momentum = args.momentum
    model_type = args.model
    nsgd = args.nsgd
    learning_rate = args.lr
    norm_t = args.norm_t
    dataset = args.dataset



    def my_seed_every(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        random.seed(seed)
        np.random.seed(seed)
        #torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() 
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)


    if nsgd == 1:
        main_dir_alt = "sgd"
    elif nsgd == 2:
        main_dir_alt = "adam"
    elif nsgd == 3:
        main_dir_alt = "INAdam"
    elif nsgd == 4:
        main_dir_alt = "adam_scratch"

    save_dir = "norm_t_%d_learning_rate_%.4f_batch_size_%d" % (norm_t, learning_rate, batch_size)

    # Directory creation
    #main_alt_dir_symbol_size = "Symbol_%d"  % (symbol_bit)
    main_dir2 = os.path.join(main_dir, main_dir_alt)
    directory = os.path.join(main_dir2, save_dir)

    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(main_dir2):
        os.mkdir(main_dir2)
    if not os.path.exists(directory):
        os.mkdir(directory)



    dataset= "cifar10"
    data_dir = 'data'
    transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 0:
        train_generator,test_generator = read_cifar10(batch_size,data_dir, my_seed_every, seed, seed_worker, g, False)
    elif dataset == 1:
        train_generator,test_generator = read_cifar100(batch_size,data_dir, my_seed_every, seed, seed_worker, g, False)


    # Model type
    my_seed_every(seed=seed)
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

    all_results = []


    training_file = os.path.join(directory, 'training.txt')

    for epoch in range(1, n_epochs+1):
        with open(training_file, 'a') as f:
            f.write("\n")
            f.write("===> Epoch {}/{}".format(epoch, n_epochs))
        #print("===> Epoch {}/{}".format(epoch, n_epochs))
        start_time = time.time()
        train_loss, train_acc = train_loop(train_generator, model, criterion, optimizer, device, inputs)
        test_loss, test_acc = test_loop(test_generator, model, criterion, device)
        time_spent = time.time() - start_time

        with open(training_file, 'a') as f:
            f.write("\n")
            f.write("Time spent:{:f}".format(time_spent))
            f.write("\n")
            f.write("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
            f.write("\n")
            f.write("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))

        #print("Time spent:{:f}".format(time_spent))
        #print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        #print("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))

        #scheduler.step()
        
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss)
        list_test_acc.append(test_acc)
        
        if test_loss <= list_test_loss[idx_best_loss]:
            #print("Loss-best model.")
            torch.save(model.state_dict(), os.path.join(directory, 'loss_model.pth'))
            idx_best_loss = epoch - 1
        
        if test_acc >= list_test_acc[idx_best_acc]:
            #print("Acc-best model.")
            torch.save(model.state_dict(), os.path.join(directory, 'acc_model.pth'))
            if test_acc > list_test_acc[idx_best_acc]:
                idx_best_acc = epoch - 1    
        #print("")

    with open(training_file, 'a') as f:
        f.write("\n")
        f.write("========================")
        f.write("\n")
        f.write("Loss-best model training loss: {:f}, acc: {:f}".format(list_train_loss[idx_best_loss], list_train_acc[idx_best_loss]))
        f.write("\n")
        f.write("Loss-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_loss], list_test_acc[idx_best_loss]))             
        f.write("\n")   
        f.write("Acc-best model training loss: {:4f}, acc: {:f}".format(list_train_loss[idx_best_acc], list_train_acc[idx_best_acc]))  
        f.write("\n")
        f.write("Acc-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_acc], list_test_acc[idx_best_acc]))    
        f.write("\n")          
        f.write("Final model training loss: {:f}, acc: {:f}".format(list_train_loss[-1], list_train_acc[-1]))     
        f.write("\n")            
        f.write("Final model test loss: {:f}, acc: {:f}".format(list_test_loss[-1], list_test_acc[-1]))  
        f.write("\n")    
        f.write("Test accuracy list:" + str(np.array(list_test_acc)))

    """
    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(list_train_loss[idx_best_loss], list_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_loss], list_test_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(list_train_loss[idx_best_acc], list_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc: {:f}".format(list_test_loss[idx_best_acc], list_test_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(list_train_loss[-1], list_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc: {:f}".format(list_test_loss[-1], list_test_acc[-1]))           
    #torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    """
    
    list_train_loss = np.array(list_train_loss)
    list_train_acc = np.array(list_train_acc)
    list_test_loss = np.array(list_test_loss)
    list_test_acc = np.array(list_test_acc)

    all_results.append((list_train_loss, list_train_acc, list_test_loss, list_test_acc))

    """
    torch.save(list_train_loss, os.path.join(directory , 'loss_train.pt'))
    torch.save(list_train_acc , os.path.join(directory , 'acc_train.pt'))
    torch.save(list_test_loss, os.path.join(directory , 'loss_test.pt'))
    torch.save(list_test_acc, os.path.join(directory , 'acc_test.pt'))
    """
    torch.save(all_results, os.path.join(directory , 'all_results.pt'))
    torch.save(model.state_dict(), os.path.join(directory, 'model_state_dict.pth'))




    save_args(args, directory)




if __name__ == '__main__':
    main()
