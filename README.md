
# Gold Guard — Reconhecimento Facial Local

## Alunos
Márcio Gastaldi - RM98811

Arthur Bessa Pian - RM99215

Davi Desenzi - RM550849

João Victor - RM551410


## Objetivo
O **Gold Guard** é um sistema de reconhecimento facial local, desenvolvido em Python com **OpenCV** (Haar Cascade para detecção e LBPH para identificação).  
O objetivo é identificar usuários cadastrados previamente em pastas de imagens e, durante a execução via webcam, exibir um retângulo em torno do rosto detectado e o **nome do usuário** logo abaixo.

> Projeto orientado a uso acadêmico/POC, sem necessidade de conexão com a internet.

---

## Estrutura do Projeto
GoldGuard/  
│  
├── faces/                  # Pastas de cada usuário com imagens de treino  
│   ├── Joao/  
│   └── Maria/  
│  
├── model/                  # Gerado após o treino  
│   ├── lbph_model.xml  
│   └── labels.json  
│  
├── train_recognizer.py     # Treinamento (gera modelo e labels)  
├── recognize.py            # Reconhecimento via webcam  
├── requirements.txt        # Dependências do projeto  
└── README.md               # Documentação  

---

## Execução

### 1) Preparar ambiente
python -m venv .venv  
# Windows  
.venv\Scripts\activate  
# Linux/macOS  
source .venv/bin/activate  

pip install -r requirements.txt  

### 2) Inserir imagens
Crie a pasta `faces/` com **uma subpasta por pessoa** (o nome da pasta será o rótulo exibido):  
faces/Joao/  
faces/Maria/  

É possível trabalhar com 2–3 imagens por pessoa; para melhor robustez, use 10–20 fotos variando ângulos e iluminação.

### 3) Treinar o modelo
python train_recognizer.py  

Arquivos gerados:  
model/lbph_model.xml  
model/labels.json  

### 4) Executar reconhecimento
python recognize.py  

Durante a execução:  
- Rostos detectados recebem um retângulo.  
- Se identificado, o **nome do usuário** é exibido abaixo do retângulo.  
- Pressione `q` para encerrar.  

---

## Dependências
- Python 3.8+  
- OpenCV (contrib)  
- NumPy  

Instalação direta:  
pip install opencv-contrib-python numpy  

---

## Parâmetros Relevantes

### Detecção (Haar Cascade)
- `scaleFactor` (padrão 1.1): escala entre verificações; menor detecta mais, porém mais lento.  
- `minNeighbors` (padrão 5–6): maior reduz falsos positivos.  
- `minSize` (ex.: 80×80 px): tamanho mínimo da face detectada.  

### Identificação (LBPH)
- `THRESHOLD` (ex.: 70–120): limiar do erro/score; **menor** é mais restritivo.  
  - Se não reconhecer rostos legítimos, aumente o valor.  
  - Se reconhecer incorretamente, diminua e/ou amplie o dataset.  

---

## Organização do Código

### train_recognizer.py
- Lê `faces/<Pessoa>/*.jpg|png`.  
- Opcionalmente detecta e recorta a maior face por imagem.  
- Redimensiona as amostras para 200×200 (tons de cinza).  
- Treina `LBPHFaceRecognizer` e salva `model/lbph_model.xml`.  
- Cria `labels.json` mapeando índices → nomes de pessoas.  

### recognize.py
- Carrega `haarcascade_frontalface_default.xml` (do OpenCV).  
- Detecta rostos na webcam com Haar Cascade.  
- Para cada rosto, executa `recognizer.predict(roi)`.  
- Exibe retângulo e o nome do usuário abaixo do retângulo quando a pontuação for menor que `THRESHOLD`.  

---

## Nota Ética sobre Uso de Dados Faciais
- Reconhecimento facial envolve **dados pessoais sensíveis**. Utilize apenas com **consentimento explícito** dos indivíduos.  
- Armazene imagens e modelos em repositórios **seguros** e com **controle de acesso**.  
- Limite o uso à finalidade informada e elimine dados quando não forem mais necessários.  
- Este projeto é uma **prova de conceito**. Para uso em produção, conduza avaliações de segurança, privacidade e conformidade (por exemplo, LGPD), além de testes de viés e desempenho.  

---

## Solução de Problemas

- `AttributeError: module 'cv2' has no attribute 'face'`  
  Instale a variante contrib:  
  pip uninstall -y opencv-python  
  pip install opencv-contrib-python  

- Câmera não abre  
  Ajuste o índice da câmera no código (`0`, `1`, `2`).  

- Muitas falhas de identificação  
  Aumente o número/variedade de fotos por pessoa, melhore a iluminação, ajuste `THRESHOLD` e `minNeighbors`.  

---

## Licença
Uso educacional e de demonstração. Avalie requisitos legais e de privacidade antes de qualquer uso real.


