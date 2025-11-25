import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_activation", pred_index=None):
    """
    Gera o mapa de calor Grad-CAM para uma imagem dada.
    
    Args:
        img_array: Array numpy da imagem (1, 224, 224, 3)
        model: Modelo Keras
        last_conv_layer_name: Nome da última camada convolucional do base_model.
                              Para EfficientNetB0, geralmente é 'top_activation'.
                              Nota: Como o base_model está aninhado, precisamos acessar o output dele.
        pred_index: Índice da classe prevista (None para a classe com maior score)
    """
    
    # Primeiro, precisamos acessar o modelo base (EfficientNet) dentro do nosso modelo
    # Nosso model.py encapsula o EfficientNet.
    # Vamos criar um modelo que mapeia a entrada para (ativações da conv, saída final)
    
    # Encontrar o base_model
    base_model = None
    for layer in model.layers:
        if 'efficientnet' in layer.name:
            base_model = layer
            break
    
    if not base_model:
        raise ValueError("Base model EfficientNet não encontrado.")

    # Criar um modelo grad_model que retorna a saída da última conv e a saída final do modelo completo
    # Isso é complexo porque o modelo completo tem camadas DEPOIS do base_model.
    # Abordagem:
    # 1. Obter output da conv layer do base_model
    # 2. Criar um novo modelo que vai da entrada -> conv_output
    # 3. Criar um novo modelo que vai da conv_output -> output final
    
    grad_model = tf.keras.models.Model(
        [base_model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    
    # No entanto, nosso modelo principal tem camadas APÓS o base_model (GlobalAvg, Dense, etc).
    # O gradiente precisa fluir por essas camadas finais também.
    # A maneira mais fácil com modelos aninhados é usar tf.GradientTape no fluxo completo.
    
    with tf.GradientTape() as tape:
        # Forward pass no base model para pegar conv outputs
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        iterate = tf.keras.models.Model([base_model.inputs], [base_model.output, last_conv_layer.output])
        base_model_out, conv_outputs = iterate(img_array)
        
        # Forward pass no restante do modelo
        # Precisamos aplicar as camadas restantes manualmente ou criar um sub-modelo
        # Vamos tentar uma abordagem mais robusta: usar o modelo completo e acessar a camada interna via nome
        # Mas 'get_layer' em modelo aninhado não funciona direto.
        
        # Simplificação: Assumindo que queremos explicar a saída do EfficientNet (features) 
        # que mais contribuem para a classificação.
        # Mas o Grad-CAM clássico precisa do gradiente da CLASSE em relação aos MAPAS DE CARACTERÍSTICAS.
        
        # Workaround para modelos aninhados:
        # Construir um modelo funcional que expõe a camada interna
        pass
        
    # Vamos re-implementar de forma mais simples para EfficientNet
    # O EfficientNetB0 tem a camada 'top_activation' como última conv.
    
    # Criar um modelo que recebe a entrada e retorna [conv_output, predictions]
    # Para isso, precisamos reconstruir o grafo ou usar o modelo existente se ele não fosse aninhado.
    # Como ele É aninhado (base_model dentro de model), é chato.
    
    # Solução: Acessar o output da camada conv através do base_model e conectar com o resto.
    # Mas o 'model' já está compilado.
    
    # Vamos usar uma técnica padrão:
    # 1. Obter as ativações da última conv layer
    # 2. Obter os gradientes da classe alvo em relação a essas ativações
    
    # Para fazer isso com modelo aninhado:
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    # Modelo que vai da entrada até a última conv e a saída do base model
    base_model_extractor = tf.keras.Model(base_model.inputs, [last_conv_layer.output, base_model.output])
    
    # Modelo que pega a saída do base model e passa pelo "classifier head" (camadas densas do nosso model)
    # Precisamos recriar a parte densa ou pegar as camadas.
    classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = classifier_input
    # Aplicar as camadas do topo do nosso modelo sequencialmente
    # Pular a primeira camada (Input) e a segunda (EfficientNet)
    for layer in model.layers[2:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        conv_outputs, base_model_out = base_model_extractor(img_array)
        tape.watch(conv_outputs)
        preds = classifier_model(base_model_out)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    
    # Average pooling dos gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada canal pelo "peso" (gradiente médio)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    """
    Sobrepõe o heatmap na imagem original.
    """
    # Carregar imagem original
    img = tf.keras.utils.img_to_array(img)
    
    # Rescale heatmap para o tamanho da imagem
    heatmap = np.uint8(255 * heatmap)
    
    # Usar jet colormap
    jet = cm.get_cmap("jet")
    
    # Cores do heatmap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Criar imagem RGB do heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    # Superpor
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    
    return superimposed_img
