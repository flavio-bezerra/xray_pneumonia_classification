import tensorflow as tf
import os

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42

def load_data(data_dir, subset=None, validation_split=None):
    """
    Carrega o dataset usando image_dataset_from_directory.
    """
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary', # 0 para Normal, 1 para Pneumonia
        class_names=['NORMAL', 'PNEUMONIA'],
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=validation_split,
        subset=subset
    )

def get_data_augmentation():
    """
    Retorna um modelo sequencial de camadas de augmentation.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1), # Rotação leve (+/- 10%)
        tf.keras.layers.RandomZoom(0.1),     # Zoom leve (+/- 10%)
        tf.keras.layers.RandomContrast(0.1), # Contraste leve
        # RandomBrightness pode ser arriscado se alterar muito a "densidade" do raio-x
    ])
    return data_augmentation

def prepare_dataset(ds, augment=False):
    """
    Prepara o dataset para performance (cache, prefetch) e aplica augmentation se solicitado.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if augment:
        data_augmentation = get_data_augmentation()
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)

    # Normalização (EfficientNetB0 espera 0-255, mas é bom garantir ou usar preprocess_input)
    # Na verdade, EfficientNetB0 tem rescaling embutido se usar weights='imagenet' e include_top=False?
    # Não, o preprocess_input do efficientnet faz o scaling correto.
    # Vamos usar o preprocess_input na modelagem, então aqui mantemos uint8 ou cast para float.
    
    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds
