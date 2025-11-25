import torch
import copy
import time
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    """
    Função de treinamento com validação e salvamento do melhor modelo.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Cada época tem uma fase de treino e uma de validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de treino
            else:
                model.eval()   # Modo de avaliação

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre os dados
            # Usar tqdm apenas no treino para não poluir demais o log
            iterator = tqdm(dataloaders[phase], desc=phase) if phase == 'train' else dataloaders[phase]
            
            for inputs, labels in iterator:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) # Ajuste para BCEWithLogitsLoss

                # Zerar os gradientes
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Predições (Sigmoid para probabilidade > 0.5)
                    preds = torch.sigmoid(outputs) > 0.5

                    # Backward + Optimize apenas no treino
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Deep copy do modelo se for o melhor na validação
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), '../models/best_model_pytorch.pth')
                print("Melhor modelo salvo!")

        print()

    time_elapsed = time.time() - since
    print(f'Treinamento completo em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Melhor Val Loss: {best_loss:.4f} Acc: {best_acc:.4f}')

    # Carregar melhores pesos
    model.load_state_dict(best_model_wts)
    return model, history
