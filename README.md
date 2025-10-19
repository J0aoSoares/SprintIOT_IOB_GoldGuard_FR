# Gold Guard — Reconhecimento Facial Local

## Alunos
Márcio Gastaldi - RM98811  
Arthur Bessa Pian - RM99215  
Davi Desenzi - RM550849  
João Victor - RM551410

---

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
├── model/                  # Gerado após o treino automático  
│   ├── lbph_model.xml  
│   └── labels.json  
│  
├── integration/            # Integrações opcionais (ex.: MQTT/HTTP/log)  
│   └── simple_integration.py  
│  
├── haarcascade_frontalface_default.xml  # Cascade de detecção (fallback para o do OpenCV)  
├── app.py                  # Aplicação principal (treina se necessário e reconhece)  
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

> Instalação manual (alternativa):  
> pip uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless  
> pip install numpy==1.26.4  
> pip install opencv-contrib-python==4.9.0.80

### 2) Inserir imagens
Crie a pasta `faces/` com **uma subpasta por pessoa** (o nome da pasta será o rótulo exibido):  
faces/Joao/  
faces/Maria/  

É possível trabalhar com 2–3 imagens por pessoa; para maior robustez, use 10–20 fotos variando ângulos e iluminação.

### 3) Treinar o modelo
O **treino é automático** na primeira execução do `app.py`.  
Arquivos gerados após o treino:  
model/lbph_model.xml  
model/labels.json

### 4) Executar reconhecimento
python app.py  
# ou, em ambientes sem interface gráfica (headless):  
python app.py --no-gui

Durante a execução:  
- Rostos detectados recebem um retângulo.  
- Se identificado, o **nome do usuário** é exibido abaixo do retângulo (com a confiança).  
- Pressione `q` para encerrar (quando GUI estiver ativa).

---

## Dependências
- Python 3.10+  
- OpenCV (contrib)  
- NumPy

Instalação direta (alternativa ao requirements):  
pip install opencv-contrib-python==4.9.0.80 numpy==1.26.4

---

## Parâmetros Relevantes

### Detecção (Haar Cascade)
- scaleFactor (padrão 1.2): escala entre verificações; menor detecta mais, porém mais lento.  
- minNeighbors (padrão 5): maior reduz falsos positivos.  
- minSize (ex.: 80×80 px): tamanho mínimo da face detectada.

### Identificação (LBPH)
- --threshold (ex.: 70–120): limiar do erro/score; **menor** é mais restritivo.  
  - Se não reconhecer rostos legítimos, aumente o valor.  
  - Se reconhecer incorretamente, diminua e/ou amplie o dataset.

### Opções de execução
- --cam Índice da câmera (padrão 0).  
- --stability Quadros consecutivos para confirmar reconhecimento (padrão 5).  
- --min-size Tamanho mínimo do rosto em pixels (padrão 80).  
- --no-gui Desativa janela de visualização (para servidores/headless).

---

## Organização do Código

### app.py
- Verifica instalação do OpenCV (módulo `cv2.face` e `CascadeClassifier`).  
- Treina automaticamente o modelo LBPH a partir de `faces/<Pessoa>/*.jpg|png` caso ainda não exista.  
- Carrega `labels.json` e `lbph_model.xml`.  
- Captura vídeo, detecta rostos, executa `predict(roi)` e desenha retângulo + nome.  
- Dispara `integration.simple_integration.notify_event(...)` quando o reconhecimento estiver estável.

### integration/simple_integration.py (opcional)
- Ponto de extensão para acionar logs, HTTP, MQTT etc. durante um evento de reconhecimento.

---

## Nota Ética sobre Uso de Dados Faciais
- Reconhecimento facial envolve **dados pessoais sensíveis**. Utilize apenas com **consentimento explícito**.  
- Armazene imagens e modelos em repositórios **seguros** e com **controle de acesso**.  
- Limite o uso à finalidade informada e elimine dados quando não forem mais necessários.  
- Este projeto é uma **prova de conceito**. Para produção, realize avaliações de segurança, privacidade (LGPD), viés e desempenho.

---

## Solução de Problemas

- `AttributeError: module 'cv2' has no attribute 'face'`  
  Instale a variante contrib:  
  pip uninstall -y opencv-python  
  pip install opencv-contrib-python==4.9.0.80

- `ImportError: numpy.core.multiarray failed to import` / `_ARRAY_API not found`  
  Ambiente com versões incompatíveis. Em um venv novo:  
  pip uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless  
  pip install numpy==1.26.4  
  pip install opencv-contrib-python==4.9.0.80

- Câmera não abre  
  Ajuste o índice da câmera (`--cam 0`, `--cam 1`, …). Verifique permissões do sistema.

- Muitas falhas de identificação  
  Aumente o número/variedade de fotos por pessoa, melhore a iluminação, ajuste `--threshold` e `minNeighbors`.

- Cascade não encontrado  
  Garanta que o arquivo `haarcascade_frontalface_default.xml` exista na raiz do projeto ou que o OpenCV possua o caminho interno (`cv2.data.haarcascades`).

---

## Licença
Uso educacional e de demonstração. Avalie requisitos legais e de privacidade antes de qualquer uso real.
