import kagglehub
import shutil
import os

def download_and_organize_data():
    print("Iniciando download do dataset...")
    # Download latest version
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Dataset baixado em:", path)

    target_dir = os.path.join("data", "raw", "chest_xray")
    
    # Se o diretório já existe, remove para evitar conflitos ou dados antigos
    if os.path.exists(target_dir):
        print(f"Removendo diretório existente: {target_dir}")
        shutil.rmtree(target_dir)
    
    print(f"Copiando dados para: {target_dir}")
    # A estrutura do dataset baixado geralmente já contém a pasta 'chest_xray' dentro.
    # Vamos verificar o conteúdo para copiar corretamente.
    
    # Copia todo o conteúdo baixado para o diretório alvo
    shutil.copytree(path, target_dir)
    print("Dados organizados com sucesso!")

    # Verificação da estrutura
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(target_dir, "chest_xray", split)
        if not os.path.exists(split_path):
             # Tenta verificar se está na raiz (alguns datasets variam a estrutura)
             split_path = os.path.join(target_dir, split)
        
        if os.path.exists(split_path):
            n_normal = len(os.listdir(os.path.join(split_path, "NORMAL")))
            n_pneumonia = len(os.listdir(os.path.join(split_path, "PNEUMONIA")))
            print(f"{split}: NORMAL={n_normal}, PNEUMONIA={n_pneumonia}")
        else:
            print(f"Aviso: Split '{split}' não encontrado na estrutura esperada.")

if __name__ == "__main__":
    download_and_organize_data()
