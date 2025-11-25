import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
import os
from torchvision import transforms

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_pytorch import PneumoniaClassifier
import importlib
import src.gradcam_pytorch
importlib.reload(src.gradcam_pytorch)
from src.gradcam_pytorch import GradCAM, show_cam_on_image

# Configuração da Página
st.set_page_config(
    page_title="Detecção de Pneumonia (PyTorch)",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Detecção de Pneumonia em Raios-X")
st.markdown("""
Esta aplicação utiliza Inteligência Artificial (EfficientNetB0 - PyTorch) para analisar radiografias de tórax e identificar sinais de pneumonia.
Além do diagnóstico, fornecemos visualização **Grad-CAM** para explicar onde o modelo focou sua atenção.
""")

# Sidebar
st.sidebar.header("Configurações")
model_path = st.sidebar.text_input("Caminho do Modelo", "notebooks/models/best_model_pytorch.pth")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PneumoniaClassifier(num_classes=1)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None

model_info = load_model(model_path)
if model_info:
    model, device = model_info
else:
    model, device = None, None

if model is None:
    st.warning(f"Modelo não encontrado em `{model_path}`. Por favor, treine o modelo primeiro executando o notebook `treinamento_pytorch.ipynb`.")
else:
    st.sidebar.success("Modelo PyTorch carregado com sucesso!")

# Upload de Imagem
uploaded_file = st.file_uploader("Escolha uma imagem de Raio-X (JPG/PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

if uploaded_file is not None:
    # Carregar imagem
    image = Image.open(uploaded_file).convert('RGB')
    
    # Botão de Análise
    if st.button("Analisar Imagem", type="primary"):
        if model is None:
            st.error("Não é possível analisar sem o modelo carregado.")
        else:
            with st.spinner("Analisando radiografia..."):
                # Pré-processamento
                img_tensor = preprocess_image(image).to(device)
                
                # Predição
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.sigmoid(output).item()
                
                # Grad-CAM
                try:
                    target_layer = model.backbone.features[-1]
                    grad_cam = GradCAM(model, target_layer)
                    cam_mask = grad_cam(img_tensor)
                    cam_image = show_cam_on_image(img_tensor.squeeze(0), cam_mask)
                except Exception as e:
                    st.error(f"Erro ao gerar Grad-CAM: {e}")
                    cam_image = None

            # --- Exibição dos Resultados ---
            st.divider()
            
            # Métricas no topo
            if prob > 0.5:
                st.error(f"🚨 **PNEUMONIA DETECTADA**")
                st.metric("Nível de Confiança do Modelo", f"{prob:.2%}")
            else:
                st.success(f"✅ **NORMAL**")
                st.metric("Nível de Confiança do Modelo", f"{(1-prob):.2%}")
            
            # Imagens lado a lado com tamanhos menores e guia à direita
            st.markdown("### 🔍 Comparação Visual")
            
            # Layout: [Imagem Original] [Grad-CAM] [Guia]
            # Ajuste de proporção: Imagens com destaque equilibrado e guia mais compacto lateralmente
            col1, col2, col3 = st.columns([1.2, 1.2, 1])
            
            with col1:
                st.info("**Raio-X Original**")
                # Redimensionar para 224x224 para garantir alinhamento perfeito com o Grad-CAM
                resized_image = image.resize((224, 224))
                st.image(resized_image, use_container_width=True)
                
            with col2:
                st.info("**Mapa de Calor (IA)**")
                if cam_image is not None:
                    st.image(cam_image, use_container_width=True)
                else:
                    st.warning("Mapa de calor não disponível.")
            
            with col3:
                st.markdown("### 📘 Guia de Interpretação")
                st.markdown("""
                **1. Entendendo o Raio-X:**
                * ⬛ **Preto:** Ar (Pulmões saudáveis e cheios de ar).
                * ⬜ **Branco:** Ossos (costelas, coluna) e tecidos densos (coração).
                * 🌫️ **Cinza/Opaco:** Pode indicar líquido, inflamação ou infecção (**Pneumonia**).
                
                **2. O que a IA viu (Grad-CAM)?**
                * O **Mapa de Calor** revela onde o modelo "olhou".
                * 🔥 **Cores Quentes (Vermelho/Amarelo):** Áreas que *mais influenciaram* a decisão da IA.
                * Se o calor estiver sobre áreas opacas (esbranquiçadas) nos pulmões, isso reforça a suspeita de pneumonia.
                
                **3. Nível de Confiança:**
                * Representa a certeza matemática do modelo, **não** a gravidade da doença.
                """)

            # Disclaimer
            st.markdown("---")
            st.error("⚠️ **AVISO IMPORTANTE:** Esta solução tem fins meramente **acadêmicos** e de demonstração técnica. Ela **NÃO** substitui um diagnóstico médico profissional. Consulte sempre um médico ou radiologista.")
            
    else:
        # Estado inicial (antes de clicar no botão)
        st.subheader("Pré-visualização da Imagem")
        # Centralizar a imagem inicial ou mostrar em tamanho razoável
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            st.image(image, caption="Imagem carregada", use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ e PyTorch")
