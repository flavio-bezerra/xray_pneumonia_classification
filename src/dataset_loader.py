import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    """
    Define as transformações para treino e validação/teste.
    Normalização baseada no ImageNet.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Cria os DataLoaders para treino, validação e teste.
    Espera a estrutura: data_dir/train, data_dir/val, data_dir/test
    """
    transforms_dict = get_transforms()
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    
    for x in ['train', 'val', 'test']:
        path = os.path.join(data_dir, x)
        if not os.path.exists(path):
            print(f"Aviso: Diretório {path} não encontrado.")
            continue
            
        image_datasets[x] = datasets.ImageFolder(path, transforms_dict[x])
        
        shuffle = True if x == 'train' else False
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers)
        dataset_sizes[x] = len(image_datasets[x])
        
    class_names = image_datasets['train'].classes if 'train' in image_datasets else []
    
    return dataloaders, dataset_sizes, class_names
