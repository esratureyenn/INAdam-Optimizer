from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch

def read_cifar10(batchsize,data_dir, my_seed_every,seed,normalization_enabled=True):

    if normalization_enabled:
        transform_train = transforms.Compose([
                                        # transforms.RandomRotation(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        # transforms.ColorJitter(brightness=1),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
                                        ])

        transform_test = transforms.Compose([
                                        # transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
                                        ])
    else:
        transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()
                                        ])

        transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        ])

    data_train = datasets.CIFAR10(root=data_dir,
                                  train=True,
                                  transform=transform_train, 
                                  #target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                  download=True)

    data_test = datasets.CIFAR10(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 #target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                 download=True
                                 )
    my_seed_every(seed)
    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    my_seed_every(seed)
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    return data_loader_train,data_loader_test

def read_cifar100(batchsize,data_dir,my_seed_every,seed,seed_worker,g,normalization_enabled=True):

    if normalization_enabled:
        transform_train = transforms.Compose([
                                        # transforms.RandomRotation(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        # transforms.ColorJitter(brightness=1),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
                                        ])

        transform_test = transforms.Compose([
                                        # transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
                                        ])
    else:
        transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()
                                        ])

        transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        ])

    data_train = datasets.CIFAR100(root=data_dir,
                                  train=True,
                                  transform=transform_train, 
                                  #target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                  download=True)

    data_test = datasets.CIFAR100(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 #target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                 download=True
                                 )
    my_seed_every(seed)
    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True, 
                                   worker_init_fn=seed_worker, 
                                   generator=g
                                   )
    my_seed_every(seed)
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    return data_loader_train,data_loader_test



