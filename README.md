# 🫁 Detecção de Pneumonia com Inteligência Artificial

![Streamlit App Preview](data/streamlit_page.png)

Bem-vindo! Este projeto utiliza Inteligência Artificial para auxiliar na identificação de pneumonia em radiografias de tórax.

O objetivo é demonstrar como a tecnologia pode ser uma ferramenta de apoio para profissionais de saúde, oferecendo uma "segunda opinião" rápida e visual.

---

## 💡 A Estratégia: Como funciona?

Para construir esta solução, não começamos do zero. Utilizamos uma técnica chamada **Transfer Learning** (Aprendizado por Transferência).

### 1. O "Cérebro" Pré-treinado (EfficientNet)

Imagine um estudante que já leu milhares de livros e sabe identificar formas, bordas e texturas complexas em imagens gerais (carros, animais, objetos). Esse "estudante" é a nossa IA base, chamada **EfficientNet-B0**. Ela já foi treinada em milhões de imagens do mundo real.

### 2. A Especialização (Fine-Tuning)

Nós pegamos esse "estudante" experiente e demos a ele um novo livro para estudar: **milhares de Raio-X de pulmões**, alguns saudáveis e outros com pneumonia.

- Mantivemos o conhecimento visual básico dele.
- Ensinamos especificamente a diferenciar um pulmão limpo de um pulmão com infecção.

### 3. Explicabilidade (Grad-CAM)

Uma IA não deve ser uma "caixa preta". Precisamos saber **por que** ela tomou uma decisão.
Para isso, implementamos o **Grad-CAM (Mapas de Calor)**.

- Quando a IA diz "Pneumonia", ela também pinta de **vermelho/amarelo** as áreas da imagem que a fizeram pensar isso.
- Geralmente, essas áreas correspondem às manchas brancas (opacidades) típicas da doença, permitindo que um humano valide se a IA está olhando para o lugar certo.

---

## 🛠️ Como usar a Aplicação (Passo a Passo)

Criamos uma interface visual simples para que qualquer pessoa possa testar o modelo.

### Pré-requisitos

Você precisará ter o **Python** instalado no seu computador.

### 1. Instalação

Abra o seu terminal (ou prompt de comando) na pasta do projeto e execute o comando abaixo para instalar as "ferramentas" necessárias:

```bash
pip install -r requirements.txt
```

_(Caso não tenha o arquivo requirements.txt, instale manualmente: `pip install torch torchvision streamlit matplotlib pandas opencv-python`)_

### 2. Treinando o Modelo (Opcional)

Se você ainda não tem o "cérebro" treinado (o arquivo `.pth` na pasta `models`), precisará executar o treinamento primeiro.

- Abra a pasta `notebooks`.
- Execute o arquivo `treinamento_pytorch.ipynb` (você pode usar o Jupyter Notebook ou VS Code).
- Isso criará o arquivo `best_model_pytorch.pth`.

### 3. Rodando o App

Com tudo pronto, digite o seguinte comando no terminal para abrir o sistema:

```bash
streamlit run app/app.py
```

O seu navegador abrirá automaticamente com a aplicação.

1.  Clique em **"Browse files"** e selecione uma imagem de Raio-X (formato .jpeg ou .png).
2.  Clique no botão **"Analisar Imagem"**.
3.  Veja o resultado e compare a imagem original com o mapa de calor gerado pela IA.

---

## 📂 Estrutura do Projeto (Para Curiosos)

- `app/`: Onde fica o código da interface visual (o site que você vê).
- `notebooks/`: Os "cadernos de estudo" onde fizemos as análises e o treinamento da IA.
- `src/`: O "motor" do projeto. Contém os códigos pesados de processamento de imagem e inteligência artificial.
- `models/`: A "memória" da IA. Onde o arquivo do modelo treinado é salvo.
- `data/`: Onde as imagens de Raio-X são armazenadas.

---

## ⚠️ Aviso Legal

**Este projeto tem fins estritamente acadêmicos e educacionais.**

A inteligência artificial, embora poderosa, pode cometer erros. Esta ferramenta **NÃO** substitui o diagnóstico de um médico ou radiologista profissional. Nunca utilize este software para tomadas de decisão clínica reais.
