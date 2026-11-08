# 🚀 Guia 03: Treinamento, Inferência e Casos de Uso com Ultralytics

> **Prioridade:** 🔧 **Alta (Do Treino à Manipulação Geométrica de Máscaras)**
> **Objetivo:** Dominar os modelos de segmentação da biblioteca **`ultralytics`** (`-seg`), configurar rotinas de **Transfer Learning** robustas, interpretar os relatórios de perda de máscara (`seg_loss`), e dominar os principais casos de uso industriais: **cálculo de área em pixels**, **remoção de fundo (alpha channel)** e **exportação ONNX**.

---

## 1. Escolha da Variante do Modelo: A Regra 80/20 de Recursos

Modelos de segmentação possuem uma tarefa a mais do que simples detectores: além de prever caixas e classes, eles geram **máscaras protótipo (ProtoNet)** e calculam coeficientes de interpolação linear para cada instância.

Mesmo com essa complexidade adicional, a família **YOLOv8-seg** e **YOLO11-seg** foi projetada para ser extremamente rápida:

| Variante                     | Exemplo de Peso    | Parâmetros | Tamanho | Latência T4 (ms) | Uso Ideal / Cenário 80/20                                                                          |
| :--------------------------- | :----------------- | :---------- | :------ | :---------------- | :-------------------------------------------------------------------------------------------------- |
| **Nano (`n-seg`)**   | `yolov8n-seg.pt` | ~3.4M       | ~7 MB   | ~2.0 ms           | **Dispositivos de borda, CPUs, prototipagem ágil e validação de dataset.**                 |
| **Small (`s-seg`)**  | `yolov8s-seg.pt` | ~11.8M      | ~24 MB  | ~3.5 ms           | **O "Sweet Spot" 80/20: Equilíbrio perfeito entre velocidade e fidelidade de bordas finas.** |
| **Medium (`m-seg`)** | `yolov8m-seg.pt` | ~27.3M      | ~54 MB  | ~6.5 ms           | Objetos complexos com textura intrincada e concavidades difíceis.                                  |
| **Large (`l-seg`)**  | `yolov8l-seg.pt` | ~46.0M      | ~92 MB  | ~10.5 ms          | Cenários onde precisão sub-pixel é mandatória e há GPU dedicada potente.                       |
| **XLarge (`x-seg`)** | `yolov8x-seg.pt` | ~71.8M      | ~144 MB | ~15.0 ms          | Benchmarks acadêmicos e competições Kaggle.                                                      |

> **Dica 80/20:** Comece sempre com **`yolov8n-seg.pt`**. O treino é quase $4\times$ mais rápido que os modelos grandes e já permite verificar se o formato das anotações e o `data.yaml` estão perfeitos. Para a versão final de produção, evolua para **`yolov8s-seg.pt`** caso precise de bordas mais suaves.

---

## 2. Pipeline de Treinamento com Transfer Learning

A chamada de treino de segmentação no Python usa exatamente a mesma API elegante da Ultralytics, bastando carregar um peso base que contenha o sufixo `-seg`:

```python
from ultralytics import YOLO
import torch

# 1. Checagem de Hardware (Usa sua GPU se disponível, senão CPU)
device = 0 if torch.cuda.is_available() else "cpu"

# 2. Carrega arquitetura com pesos pré-treinados no COCO (segmentação de 80 classes)
modelo = YOLO("yolov8n-seg.pt")

# 3. Dispara o fine-tuning adaptando as cabeças às suas classes específicas
resultados = modelo.train(
    data="data.yaml",           # Caminho para o arquivo de configuração
    epochs=50,                  # 50 épocas são suficientes para a regra 80/20
    imgsz=640,                  # Resolução padrão da grade espacial
    batch=16 if device == 0 else 4,  # Ajuste conforme memória de GPU/RAM
    patience=15,                # Early stopping: para se o Mask mAP estagnar por 15 épocas
    device=device,              # GPU (0) ou CPU ('cpu')
    workers=4,                  # Processamento paralelo de carregamento de imagens
    project="runs/segment",     # Diretório raiz dos experimentos
    name="meu_segmentador_v1"   # Subpasta do experimento atual
)
```

### O que o YOLO-seg faz nos bastidores?

* **Backbone & Neck**: Extraem pirâmide de características espaciais da imagem.
* **ProtoNet**: Gera 32 máscaras protótipo genéricas em resolução $160 \times 160$ (1/4 da entrada de 640).
* **Segment Head**: Para cada objeto detectado, a rede prediz a caixa delimitadora $[xc, yc, w, h]$, a confiança, a classe e **32 coeficientes escalares**.
* A máscara final da instância é o produto escalar dos 32 coeficientes pelos 32 protótipos, recortada rigorosamente dentro da bounding box prevista.

---

## 3. Interpretando os Artefatos e Métricas de Validação

Ao concluir o treino, os arquivos de diagnóstico estarão em `runs/segment/meu_segmentador_v1/`:

```text
runs/segment/meu_segmentador_v1/
├── weights/
│   ├── best.pt              # Melhores pesos (maior Mask mAP na validação)
│   └── last.pt              # Pesos da última época (para retomar treino se cair)
├── results.png              # Gráficos de perda e evolução do mAP
├── confusion_matrix.png     # Matriz de confusão de classificação
├── val_batch0_labels.jpg    # Amostra visual do Ground Truth (anotações reais)
└── val_batch0_pred.jpg      # Amostra visual com as máscaras previstas pelo modelo
```

### O que observar no `results.png`?

1. **`train/seg_loss` e `val/seg_loss`**: Curva de perda das máscaras. Deve cair continuamente. Se `val/seg_loss` começar a subir enquanto `train/seg_loss` continua descendo, o modelo está memorizando (overfitting).
2. **`metrics/mAP50(B)` vs `metrics/mAP50(M)`**:
   * O sufixo `(B)` significa **Box** (retângulo).
   * O sufixo `(M)` significa **Mask** (polígono de pixels).
   * Verifique se `mAP50(M)` atingiu um patamar satisfatório (geralmente acima de $0.75$ a $0.85$).

---

## 4. Pipeline Profissional de Inferência & Anatomia de `result.masks`

O grande diferencial prático da segmentação está no pós-processamento dos resultados.

```python
from ultralytics import YOLO

modelo = YOLO("runs/segment/meu_segmentador_v1/weights/best.pt")

# Inferência com limiares 80/20
resultados = modelo.predict(
    source="foto_teste.jpg",
    conf=0.35,      # Descarta detecções de baixa certeza
    iou=0.60,       # Suprime sobreposições espaciais redundantes
    save=False,     # Não grava em disco automaticamente para processarmos em memória
    verbose=False
)

resultado = resultados[0]  # Primeiro resultado do lote
```

### O Objeto `resultado.masks` (O Coração da Segmentação)

Se objetos forem detectados, `resultado.masks` conterá as informações geométricas:

```python
if resultado.masks is not None:
    # 1. result.masks.data -> Tensor PyTorch binário (N, H, W) com valores 0.0 e 1.0
    # Ideal para operações tensoriais e máscaras matriciais
    mascaras_tensor = resultado.masks.data.cpu().numpy()

    # 2. result.masks.xy -> Lista com N arrays NumPy de vértices [[x1, y1], [x2, y2], ...]
    # Já convertidos para pixels inteiros originais da imagem de entrada!
    poligonos_pixels = resultado.masks.xy

    # 3. result.masks.xyn -> Lista com N arrays de vértices normalizados em [0.0, 1.0]
    poligonos_norm = resultado.masks.xyn
```

---

## 5. Casos de Uso Reais: O Poder Prático das Máscaras (80/20)

### Caso 1: Cálculo de Área Real e Porcentagem de Ocupação

Imagine que você quer medir o tamanho de uma folha, a área de uma lesão biológica ou a área de uma mancha de óleo:

```python
import cv2
import numpy as np

# Pega a máscara binária do primeiro objeto redimensionada para a foto original
h_orig, w_orig = resultado.orig_shape
mascara_original = resultado.masks.data[0].cpu().numpy()
mascara_binaria = cv2.resize(mascara_original, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

# 1. Total de pixels ocupados pelo objeto
pixels_objeto = int(np.count_nonzero(mascara_binaria > 0.5))
pixels_totais = h_orig * w_orig

# 2. Porcentagem da cena ocupada pelo objeto
porcentagem_cena = (pixels_objeto / pixels_totais) * 100

print(f"Área do objeto: {pixels_objeto:,} pixels ({porcentagem_cena:.2f}% da imagem)")

# Se você souber a calibração física da câmera (ex: 1 pixel = 0.05 cm):
escala_cm_por_pixel = 0.05
area_cm2 = pixels_objeto * (escala_cm_por_pixel ** 2)
print(f"Área física estimada: {area_cm2:.2f} cm²")
```

---

### Caso 2: Remoção Perfeita de Fundo & PNG com Canal Alfa

Na detecção, recortar uma caixa retangular traz todo o fundo junto. Na segmentação, podemos isolar o objeto com precisão cirúrgica:

```python
import cv2
import numpy as np

imagem_bgr = resultado.orig_img  # Imagem original carregada pelo Ultralytics
h, w = imagem_bgr.shape[:2]

# Cria uma máscara vazia de 1 canal
mascara_total = np.zeros((h, w), dtype=np.uint8)

# Preenche os polígonos de todos os objetos detectados
for poligono in resultado.masks.xy:
    pts = np.int32([poligono])
    cv2.fillPoly(mascara_total, pts, 255)

# Opção A: Isolar o objeto com fundo preto (3 canais BGR)
objeto_isolado = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mascara_total)
cv2.imwrite("objeto_fundo_preto.jpg", objeto_isolado)

# Opção B: Gerar PNG transparente com Canal Alfa (4 canais BGRA)
b, g, r = cv2.split(imagem_bgr)
rgba = cv2.merge([b, g, r, mascara_total])
cv2.imwrite("objeto_transparente.png", rgba)
print("✓ Imagem com fundo transparente salva em 'objeto_transparente.png'!")
```

---

### Caso 3: Extração de Centróide (Centro de Massa) para Robótica e Geometria

Se você precisa guiar o braço de um robô ou uma garra para agarrar um objeto, o centro de uma bounding box falha se o objeto for em formato de "L" ou tiver peso assimétrico. O centróide da máscara de pixels entrega o centro de massa exato:

```python
# Calcula momentos estatísticos da máscara binária
momentos = cv2.moments(mascara_total)

if momentos["m00"] != 0:
    centro_x = int(momentos["m10"] / momentos["m00"])
    centro_y = int(momentos["m01"] / momentos["m00"])
    print(f"Centro de massa real (X, Y): ({centro_x}, {centro_y})")
```

---

## 6. Exportação para Produção & Edge Computing (ONNX)

Assim como os modelos de detecção, os modelos `-seg` podem ser compilados diretamente para **ONNX**, permitindo rodar em hardware embarcado, servidores em C++ ou dispositivos com **ONNX Runtime**:

```python
from ultralytics import YOLO

modelo = YOLO("runs/segment/meu_segmentador_v1/weights/best.pt")

caminho_onnx = modelo.export(
    format="onnx",
    imgsz=640,
    dynamic=True,   # Suporta tamanhos dinâmicos de batch
    simplify=True   # Otimiza o grafo fundindo nós redundantes
)

print(f"✓ Modelo de segmentação exportado com sucesso: {caminho_onnx}")
```

### O que muda na saída do grafo ONNX de segmentação?

* Em detecção, o grafo retornava um único tensor com `[xc, yc, w, h, classes]`.
* Na segmentação, o grafo ONNX exporta **duas saídas**:
  1. `output0`: Vetores de predição com caixas, pontuações de classe e os **32 coeficientes de máscara**.
  2. `output1`: As **32 máscaras protótipo** geradas pela ProtoNet em resolução reduzida.

> 💡 **Atalho 80/20 em Produção:** Se for rodar a inferência com Python, você não precisa fazer a matemática de reconstrução dos protótipos manualmente! Basta carregar o `.onnx` diretamente pelo Ultralytics:
> `onnx_seg = YOLO("best.onnx")`
> `res = onnx_seg.predict("foto.jpg")`
> O Ultralytics decodifica os tensores ONNX e devolve a mesma API elegante com `res.masks.xy` pronta para uso!
