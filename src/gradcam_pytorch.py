import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output é uma tupla, pegamos o primeiro elemento
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Resetar estado
        self.gradients = None
        self.activations = None
        
        # Garantir que a entrada requer gradiente para que o backprop flua até ela
        # Isso é necessário se o backbone estiver congelado (pesos com requires_grad=False)
        if not x.requires_grad:
            x.requires_grad = True
            
        # Forward pass
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zerar gradientes anteriores
        self.model.zero_grad()
        
        # Backward pass para a classe alvo
        if output.shape[1] == 1:
            score = output[:, 0]
        else:
            score = output[:, class_idx]
            
        score.backward(retain_graph=True)
        
        # Verificar se capturamos gradientes e ativações
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Não foi possível capturar gradientes ou ativações. Verifique se o hook foi registrado corretamente e se os gradientes estão fluindo.")
        
        # Gerar mapa de calor
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling dos gradientes
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Combinação linear das ativações ponderadas pelos gradientes
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalização min-max
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()

def show_cam_on_image(img_tensor, cam_mask):
    """
    Superpõe o Grad-CAM na imagem original.
    img_tensor: Tensor (C, H, W) normalizado
    cam_mask: Array numpy (1, 1, H, W) ou (H, W)
    """
    # Desnormalizar imagem para visualização (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Redimensionar máscara para o tamanho da imagem
    if len(cam_mask.shape) == 4:
        cam_mask = cam_mask[0, 0]
        
    cam_mask = cv2.resize(cam_mask, (img.shape[1], img.shape[0]))
    
    # Colorir mapa de calor
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    # Superposição
    cam_img = heatmap * 0.4 + img
    cam_img = cam_img / np.max(cam_img)
    
    return np.uint8(255 * cam_img)
