import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import os

def plot_class_distribution(train_dir, test_dir, val_dir=None):
    """Plota a distribuição das classes nos conjuntos de dados."""
    splits = ['Treino', 'Teste']
    dirs = [train_dir, test_dir]
    if val_dir:
        splits.append('Validação')
        dirs.append(val_dir)
    
    counts = {}
    for split, path in zip(splits, dirs):
        counts[split] = {
            'Normal': len(os.listdir(os.path.join(path, 'NORMAL'))),
            'Pneumonia': len(os.listdir(os.path.join(path, 'PNEUMONIA')))
        }
        
    # Plot
    fig, ax = plt.subplots(1, len(splits), figsize=(15, 5))
    if len(splits) == 1: ax = [ax]
    
    for i, split in enumerate(splits):
        data = counts[split]
        sns.barplot(x=list(data.keys()), y=list(data.values()), ax=ax[i], palette='viridis')
        ax[i].set_title(f'Distribuição - {split}')
        ax[i].set_ylabel('Quantidade de Imagens')
        
        # Adicionar contagem nas barras
        for p in ax[i].patches:
            ax[i].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    return fig

def plot_samples(generator, num_samples=5):
    """Plota amostras de imagens do gerador de dados."""
    images, labels = next(generator)
    class_names = ['Normal', 'Pneumonia'] # Assumindo 0=Normal, 1=Pneumonia (verificar class_indices)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    return plt

def plot_pixel_intensity(normal_images, pneumonia_images):
    """Plota histogramas de intensidade de pixel média."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(normal_images.ravel(), bins=256, color='blue', label='Normal', ax=ax[0], kde=True)
    ax[0].set_title('Histograma de Intensidade de Pixel - Normal')
    
    sns.histplot(pneumonia_images.ravel(), bins=256, color='red', label='Pneumonia', ax=ax[1], kde=True)
    ax[1].set_title('Histograma de Intensidade de Pixel - Pneumonia')
    
    plt.legend()
    return fig
