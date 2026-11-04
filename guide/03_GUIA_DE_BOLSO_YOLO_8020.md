# 📋 Guia de Bolso: YOLO 80/20

> **Objetivo:** O resumo de consulta rápida mais objetivo para manter aberto ao lado do seu editor de código durante o desenvolvimento de modelos de detecção de objetos.

---

## 1. Estrutura de Dados & Formato de Anotação

| Cenário / Objetivo | Formato / Snippet | O que resolve |
| :--- | :--- | :--- |
| **Linha de anotação YOLO TXT** | `classe_id x_centro y_centro largura altura` | Formato padrão com coordenadas normalizadas em $[0, 1]$. |
| **Normalizar coordenada X** | `x_norm = x_pixel / largura_imagem` | Torna a coordenada independente da resolução final da imagem. |
| **Normalizar coordenada Y** | `y_norm = y_pixel / altura_imagem` | Garante que o modelo opere em qualquer aspecto de proporção. |
| **Configuração mínima de dataset** | Arquivo `data.yaml` com caminhos `path`, `train`, `val` e lista `names` | Centraliza metadados e mapeamento numérico de classes. |
| **Imagem de Fundo (Sem objetos)** | Arquivo `.txt` vazio com mesmo nome da foto | Ensina o modelo a não gerar falsos positivos em áreas vazias. |

---

## 2. Treinamento & Transfer Learning

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Carregar rede pré-treinada** | `model = YOLO("yolov8n.pt")` | Baixa os filtros visuais refinados no dataset COCO (80 classes). |
| **Treino básico balanceado** | `model.train(data="data.yaml", epochs=50, imgsz=640)` | Dispara o fine-tuning adaptando a cabeça às suas novas classes. |
| **Evitar congelamento de GPU** | `model.train(..., batch=8, workers=2)` | Reduz pressão na VRAM para placas de vídeo com 4GB a 6GB. |
| **Early Stopping anti-overfitting** | `model.train(..., patience=15)` | Interrompe o treino se o `mAP50-95` estagnar por 15 épocas. |
| **Treinar na CPU (Sem GPU)** | `model.train(..., device="cpu")` | Permite testes locais sem dependência de drivers CUDA. |
| **Retomar treino interrompido** | `model = YOLO("runs/.../last.pt"); model.train(resume=True)` | Continua o processo exatamente da época e pesos onde parou. |

---

## 3. Inferência & Manipulação de Resultados

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Carregar pesos otimizados** | `model = YOLO("runs/.../weights/best.pt")` | Garante o uso dos pesos que atingiram o maior mAP na validação. |
| **Prever e salvar caixas** | `results = model.predict("foto.jpg", conf=0.35, save=True)` | Desenha as caixas detectadas e grava em disco na pasta `runs/detect/predict`. |
| **Stream contínuo sem estouro de RAM** | `for res in model("video.mp4", stream=True):` | Gerador Python $O(1)$ RAM: impede crash do sistema por esgotamento de memória. |
| **Câmera IP ou Webcam** | `model.predict(source=0)` ou `source="rtsp://..."` | Ingestão universal de vídeo local, webcam USB ou streaming de segurança. |
| **Extrair caixas em pixels** | `x1, y1, x2, y2 = box.xyxy[0].int().tolist()` | Coordenadas inteiras absolutas prontas para slicing no OpenCV. |
| **Auto-Crop de evidência** | `crop = frame[y1:y2, x1:x2]; cv2.imwrite("fogo.jpg", crop)` | Recorta apenas a chama/fumaça para auditoria forense ou alerta via webhook. |
| **Cerca Virtual / Zonas de Risco** | `cv2.pointPolygonTest(poligono, (xc, yc), False) >= 0` | Ignora fumaça esperada em caldeiras e monitora apenas áreas inflamáveis. |
| **Persistência Anti-Alarme Falso**| `buffer = deque(maxlen=30); if sum(buffer) > 20: sirene()` | Impede que a sirene física de 12V dispare por causa de um único frame de ruído. |
| **Rastrear expansão de fogo** | `model.track(source="video.mp4", tracker="bytetrack.yaml")` | Atribui ID persistente a cada foco e permite calcular a taxa de crescimento da área. |

---

## 4. Auditoria de Métricas & Diagnósticos

| Métrica / Artefato | Arquivo Gerado | O que indica |
| :--- | :--- | :--- |
| **mAP50 alto (> 0.85)** | `results.png` | O modelo sabe localizar e identificar os objetos com clareza. |
| **mAP50-95 em subida** | `results.png` | As bordas das caixas estão se tornando milimetricamente precisas. |
| **Auditoria de Falsos Positivos**| `confusion_matrix.png` | Avalia se o modelo está confundindo classes entre si ou com o fundo. |
| **Inspeção Visual Imediata** | `val_batch0_pred.jpg` | Permite enxergar diretamente na imagem se as caixas previstas fazem sentido. |

---

## 5. Exportação & Deploy em Produção

| Cenário / Objetivo | Snippet de Código | O que resolve |
| :--- | :--- | :--- |
| **Exportar para ONNX Universal** | `model.export(format="onnx", dynamic=True, simplify=True)` | Gera grafo estático compatível com C++, Go, Web e `onnxruntime`. |
| **Exportar em Meia Precisão (FP16)** | `model.export(format="onnx", half=True)` | Reduz o arquivo pela metade e dobra a velocidade em GPUs modernas. |
| **Inferência ultraleve com ONNX** | `onnx_model = YOLO("best.onnx"); onnx_model.predict("foto.jpg")` | Executa inferência preservando exatamente a mesma API do PyTorch. |
