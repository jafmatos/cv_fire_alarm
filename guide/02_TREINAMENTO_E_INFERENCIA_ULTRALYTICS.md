# 🚀 Guia 02: Treinamento, Inferência e Deploy com Ultralytics

> **Prioridade:** 🔧 **Alta (Acelerando o Ciclo de Desenvolvimento & Produção)**
> **Objetivo:** Dominar a biblioteca padrão da indústria **`ultralytics`**, escolher o tamanho ideal de modelo (**Nano** vs **Small**), configurar rotinas de **Transfer Learning** robustas, interpretar os relatórios de validação e construir pipelines de inferência e exportação (**ONNX**).

---

## 1. Escolha da Variante do Modelo: A Regra 80/20 de Recursos

Ao invés de tentar treinar arquiteturas imensas que exigem GPUs industriais, 80% das aplicações de mercado alcançam resultados comerciais viáveis usando apenas as variantes **Nano (`n`)** ou **Small (`s`)**:

| Variante                 | Exemplo de Peso                 | Parâmetros | Tamanho do Arquivo | Latência T4 (ms) | Uso Ideal / Cenário                                                                    |
| :----------------------- | :------------------------------ | :---------- | :----------------- | :---------------- | :-------------------------------------------------------------------------------------- |
| **Nano (`n`)**   | `yolov8n.pt` / `yolo11n.pt` | ~3.2M       | ~6 MB              | ~1.5 ms           | **Dispositivos de Borda (Raspberry Pi, celulares), CPUs e prototipagem rápida.** |
| **Small (`s`)**  | `yolov8s.pt` / `yolo11s.pt` | ~11.2M      | ~22 MB             | ~2.5 ms           | **O "Sweet Spot" 80/20: Equilíbrio perfeito entre acurácia e velocidade.**      |
| **Medium (`m`)** | `yolov8m.pt` / `yolo11m.pt` | ~25.9M      | ~50 MB             | ~4.7 ms           | Tarefas complexas com dezenas de classes e sobreposição visual.                       |
| **Large (`l`)**  | `yolov8l.pt` / `yolo11l.pt` | ~43.7M      | ~87 MB             | ~7.5 ms           | Quando a GPU de inferência é potente e a precisão é prioritária.                   |
| **XLarge (`x`)** | `yolov8x.pt` / `yolo11x.pt` | ~68.2M      | ~136 MB            | ~11.0 ms          | Competições acadêmicas e benchmarks de pesquisa.                                     |

> **Dica Prática:** Sempre inicie seu experimento com a variante **Nano**. Se o modelo atingir um bom `mAP50` rapidamente, teste a versão **Small** para refinar as bordas das caixas. Dificilmente você precisará de algo maior que `s` para datasets com menos de 5.000 imagens.

---

## 2. Pipeline de Treinamento com Transfer Learning

A biblioteca `ultralytics` automatiza o download dos pesos pré-treinados no COCO (80 classes gerais) e ajusta as camadas finais para as classes definidas no seu `data.yaml`.

### Script de Treino Python Modular (`train.py`)

```python
from pathlib import Path
from ultralytics import YOLO

def treinar_detector():
    # 1. Carrega os pesos pré-treinados oficiais
    # O YOLO reconhece que o data.yaml possui um número diferente de classes
    # e inicializa uma nova cabeça de detecção automaticamente.
    model = YOLO("yolov8n.pt")

    # 2. Configura e inicia o treinamento
    results = model.train(
        data="data.yaml",       # Caminho para o descritor de dados
        epochs=60,              # 50-80 épocas costumam ser o suficiente
        imgsz=640,              # Resolução padrão de redimensionamento
        batch=16,               # Ajuste para 8 se tiver restrição de VRAM
        patience=15,            # Early Stopping se o mAP50-95 não subir em 15 épocas
        workers=4,              # Subprocessos para carregamento de imagens na CPU
        device=0,               # 0 para GPU CUDA, 'cpu' para processador
        save=True,              # Salva automaticamente pesos periódicos
        save_period=10,         # Salva checkpoints a cada 10 épocas
        project="experimentos", # Nome da pasta raiz dos resultados
        name="detector_v1",     # Subpasta do experimento atual
        optimizer="auto",       # Ajusta automaticamente entre SGD e AdamW
        verbose=True
    )
  
    print("Treinamento finalizado com sucesso!")
    print(f"Melhores pesos salvos em: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    treinar_detector()
```

---

## 3. Por Dentro do Motor: O que o YOLO Faz nos Bastidores Durante o Treino?

Quando você executa `model.train(data="data.yaml", epochs=60)`, parece que nada de mais está acontecendo além de uma barra de progresso. Na realidade, o YOLO executa um dos pipelines matemáticos e arquiteturais mais sofisticados da inteligência artificial moderna, dividido em **6 fases contínuas**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      O CICLO DE TREINAMENTO DO YOLO NOS BASTIDORES                     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   ▼                                                                                 ▼
[ FASE 0: Mutação Arquitetural ]                                  [ FASE 1: Augmentation Dinâmico ]
• Detecta 2 classes no data.yaml                                  • Mosaic 4-Image (mistura escalas)
• Transplanta a cabeça final (Decoupled Head)                     • Recálculo geométrico das caixas
• Congela/ajusta bias com probabilidade a priori                  • Desliga Mosaic nas últimas 10 épocas
   │                                                                                 │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   ▼                                                                                 ▼
[ FASE 2: Geração das 8.400 Caixas ]                              [ FASE 3: Atribuição de Alvos (TAL) ]
• 3 escalas de grid: P3 (80x80), P4 (40x40), P5 (20x20)           • Métrica de alinhamento: t = s^α × IoU^β
• Prevê 8.400 caixas candidatas por imagem                        • Seleciona as melhores caixas positivas
   │                                                                                 │
   └────────────────────────────────────────┬────────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────┴────────────────────────────────────────┐
   ▼                                                                                 ▼
[ FASE 4: A Tríplice Perda (Loss Balanceada) ]                    [ FASE 5: Otimização & Modelo Sombra ]
• Box Loss (CIoU: centro, área e aspect ratio)                    • Warmup inicial nas épocas 1 a 3
• Class Loss (BCE: Fumaça vs Fogo vs Fundo)                       • Cosine Annealing da taxa de aprendizado
• DFL Loss (Bordas difusas via probabilidade integral)            • EMA (Exponential Moving Average) ──► best.pt
```

---

### Fase 0: A Cirurgia Arquitetural Inicial (Model Mutation)

O modelo base (`yolov8s.pt`) veio pré-treinado nas **80 classes do COCO** (pessoa, carro, cachorro, etc.). O seu `data.yaml` tem apenas **2 classes** (`0: Fumaça`, `1: Fogo`).

Antes de disparar a primeira época, o YOLO realiza uma cirurgia na rede:

1. **Preservação do Backbone e Neck:** Todos os extratores convolucionais profundos (blocos C2f e pirâmide PANet) são mantidos 100% intactos com os filtros do COCO.
2. **Substituição da Cabeça Desacoplada (*Decoupled Head*):** A última camada convolucional de classificação (que tinha 80 canais de saída) é extirpada e substituída por uma nova camada com **2 canais**.
3. **Calibração de Bias Inicial:** Se os novos pesos fossem inicializados de forma puramente aleatória, a perda (*loss*) explodiria na época 1. O YOLO inicializa o bias da nova cabeça com base na frequência esperada:

   $$
   b = -\log\left(\frac{1 - \pi}{\pi}\right)
   $$

   onde $\pi \approx 0.01$. Isso força a rede a começar prevendo probabilidades baixas para o primeiro plano, evitando que gradientes instáveis destruam os filtros pré-treinados nos primeiros lotes.

---

### Fase 1: Data Augmentation com Recálculo Geométrico

Em problemas de classificação, aplicar data augmentation é fácil (basta girar ou cortar o tensor da imagem). Na detecção, **se você rotaciona, estica ou corta a imagem, as Bounding Boxes $[x, y, w, h]$ precisam ser recalculadas trigonometricamente**:

* **Mosaic 4-Image:** O YOLO sorteia 4 fotos aleatórias do seu dataset, redimensiona cada uma em escalas distintas e as "costura" em uma única imagem composta de $640 \times 640$. Isso ensina o modelo a identificar fogo e fumaça em diferentes distâncias (longe vs perto) e em posições não centralizadas.
* **HSV Jitter:** Distorce matiz (*hue*), saturação e brilho para que a rede não memorize uma cor exata de chama ou fumaça, aprendendo a reconhecer fogo tanto à noite quanto sob sol forte.
* **O Segredo do `close_mosaic=10`:** Nas últimas 10 épocas do treino, o YOLO **desativa o Mosaic** e passa a treinar apenas com fotos reais e limpas. Isso reduz o ruído de corte artificial e permite que a rede ajuste as bordas das caixas com precisão milimétrica na reta final.

---

### Fase 2: A Grade Multi-Escala e as 8.400 Predições

O YOLOv8 é um detector **Anchor-Free** (livre de âncoras fixas). Ele divide a imagem de entrada ($640 \times 640$) em 3 níveis espaciais da pirâmide de características (*Feature Pyramid*):

| Nível          | Stride | Resolução da Grade | Total de Células      | Especialidade de Detecção                                                     |
| :-------------- | :----- | :------------------- | :--------------------- | :------------------------------------------------------------------------------ |
| **P3**    | 8      | $80 \times 80$     | **6.400 caixas** | **Objetos Minúsculos** (pequenos focos de fogo ou fumaça no horizonte). |
| **P4**    | 16     | $40 \times 40$     | **1.600 caixas** | **Objetos Médios** (queimadas a meia distância, fogueiras).             |
| **P5**    | 32     | $20 \times 20$     | **400 caixas**   | **Objetos Grandes** (incêndios em primeiro plano ocupando a tela).       |
| **Total** | —     | —                   | **8.400 caixas** | **Candidatas avaliadas simultaneamente a cada foto!**                     |

---

### Fase 3: O Algoritmo de Atribuição de Alvos (Task-Aligned Assigner)

Se a rede gera 8.400 caixas candidatas para cada imagem, **como ela sabe qual dessas caixas deve aprender a caixa de fogo anotada por você?**

Antigamente, usava-se apenas a sobreposição espacial geométrica (IoU). O YOLOv8 usa o **Task-Aligned Assigner (TAL)**, que calcula uma métrica de alinhamento $t$ para cada uma das 8.400 caixas candidatas:

$$
t = s^\alpha \times \text{IoU}^\beta
$$

* $s$ é a probabilidade da classe correta prevista pela caixa;
* $\text{IoU}$ é a sobreposição geométrica com a caixa verdadeira;
* $\alpha = 0.5$ e $\beta = 6.0$ são pesos que priorizam caixas que acertam a geometria **e** a classe simultaneamente.

As $top\text{-}k$ caixas com maior valor de $t$ são selecionadas como **exemplos positivos** e recebem os gradientes para se aproximarem ainda mais da caixa real. As outras ~8.390 caixas restantes são marcadas como **fundo (*background*)** e são treinadas para prever probabilidade zero de classe.

---

### Fase 4: A Tríplice Função de Perda (Loss Composta)

Durante a retropropagação (*backpropagation*), o otimizador calcula o erro da rede combinando 3 funções de perda simultâneas:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}
$$

1. **`box_loss` (Complete IoU - CIoU):**Não mede apenas a distância entre pixels. Ela penaliza:
   * A área de sobreposição (IoU);
   * A distância euclidiana normalizada entre os centros das duas caixas;
   * A discrepância da proporção de aspecto (*aspect ratio* entre largura e altura).
2. **`cls_loss` (Binary Cross-Entropy - BCE):**Avalia a classificação probabilística entre `Fumaça`, `Fogo` e `Fundo`.
3. **`dfl_loss` (Distribution Focal Loss - DFL):**
   O grande diferencial para detectar fumaça! Como fumaça é semitransparente e não tem margem rígida, o modelo não tenta adivinhar uma coordenada absoluta única. Ele prevê uma **distribuição de probabilidade contínua** sobre os limites das bordas e calcula uma integral discreta. Isso dá ao modelo flexibilidade para aprender bordas suaves e esfumaçadas.

---

### Fase 5: Otimizador, Warm-up e o "Modelo Sombra" (EMA)

* **Warm-up Inicial (Épocas 1 a 3):**Nas 3 primeiras épocas, a taxa de aprendizado começa minúscula e sobe progressivamente. Isso amortece os gradientes iniciais e protege os filtros pré-treinados de serem descalibrados por lotes atípicos.
* **Cosine Annealing:**A taxa de aprendizado decai seguindo uma curva cosseno suave, reduzindo a velocidade dos passos perto da época 60 para que a rede assente os pesos nos vales mais estáveis da função de perda.
* **EMA (Exponential Moving Average):**
  A cada época, o YOLO mantém duas versões do modelo na memória: os pesos do treino ativo e uma cópia "sombra" suavizada por média móvel exponencial:

  $$
  \theta_{\text{EMA}} = \beta \cdot \theta_{\text{EMA}} + (1 - \beta) \cdot \theta_{\text{atual}}
  $$

  O arquivo final **`best.pt` é gerado a partir do modelo EMA**, e não dos pesos instantâneos. Isso imuniza o seu detector contra oscilações bruscas e garante uma generalização muito superior no mundo real.

---

## 4. Interpretação dos Artefatos de Validação

Ao final do treino, o YOLO gera uma pasta completa com diagnósticos em `experimentos/detector_v1/`. Três arquivos respondem 80% das suas perguntas:

```
experimentos/detector_v1/
├── weights/
│   ├── best.pt                   # Pesos com maior mAP na validação (USE ESTE!)
│   └── last.pt                   # Última época executada (para retomar treino)
├── results.png                   # Gráficos de evolução das perdas e mAP por época
├── confusion_matrix.png          # Matriz de acertos e confusão entre classes
└── val_batch0_pred.jpg           # Predições visuais com caixas desenhadas na validação
```

### Como Diagnosticar Problemas:

1. **`results.png`**:
   * A curva `val/box_loss` começou a subir enquanto `train/box_loss` continuou caindo? $\rightarrow$ **Overfitting**. O *patience* (Early Stopping) interromperá o processo automaticamente.
2. **`confusion_matrix.png`**:
   * Observe a coluna e linha de **Background (Fundo)**.
   * Se o modelo prevê muitas classes como fundo: o modelo está sofrendo de **falsos negativos** (objetos não detectados).
   * Se o modelo prevê fundo como objeto: o modelo tem **falsos positivos** (detectando ruídos aleatórios).
3. **`val_batch0_pred.jpg`**:
   * Inspeção visual direta. Verifique se as caixas estão alinhadas com as margens reais dos objetos.

---

## 4. Pipeline de Inferência: A Jornada Gradual

Depois do treinamento concluído e do arquivo de pesos `best.pt` gerado, entramos na fase de **inferência** (colocar o modelo para prever amostras inéditas).

Para dominar a inferência sem se perder na complexidade, a evolução deve ser **estritamente gradual**: partindo de uma simples imagem estática de 3 linhas até chegar, passo a passo, em automações, vídeos e atuadores físicos.

---

### Nível 1: O "Hello World" da Inferência (Imagem Única)

No nível mais simples possível, você só precisa carregar o modelo e passar o caminho de uma foto:

```python
from ultralytics import YOLO

# 1. Carrega o modelo com os pesos treinados
modelo = YOLO("runs/detect/experimentos/detector_v1/weights/best.pt")

# 2. Executa a predição sobre uma única imagem
resultados = modelo.predict(source="input/foto_teste.jpg", save=True)
```

#### O que o Ultralytics faz automaticamente por baixo dos panos?

1. **Lê a imagem** do disco e a redimensiona para $640 \times 640$ mantendo a proporção (adiciona barras cinzas nas bordas via *letterboxing*).
2. **Executa o forward pass** na rede neural, calculando as 8.400 predições candidatas.
3. **Aplica o NMS (Non-Maximum Suppression)** para descartar caixas sobrepostas no mesmo objeto.
4. **Salva a foto anotada**: se `save=True`, cria a pasta `runs/detect/predict/` e grava a foto com as caixas coloridas e as porcentagens desenhadas.
5. **Devolve a variável `resultados`**: uma lista Python contendo um objeto `Results` para cada imagem avaliada.

---

### Nível 2: Desvendando o Objeto `Results` (Lendo os Dados Via Código)

Em vez de apenas abrir a imagem salva na pasta para olhar as caixinhas, você vai querer inspecionar o que o modelo detectou via código para tomar decisões:

```python
# Como passamos 1 imagem, pegamos o primeiro item da lista:
resultado = resultados[0]

# 1. Quantos objetos foram vistos na foto?
total_objetos = len(resultado.boxes)
print(f"Total de objetos encontrados: {total_objetos}")

# 2. Se total_objetos == 0, a imagem está limpa (cenário seguro / sem fogo nem fumaça)
if total_objetos == 0:
    print("Nenhum foco detectado. Ambiente normal.")

# 3. Se houver detecções, inspecionamos cada caixa encontrada:
for box in resultado.boxes:
    # Classe: devolve um tensor com o ID numérico (0 ou 1)
    classe_id = int(box.cls[0])
    nome_classe = resultado.names[classe_id]   # Converte 0 -> "Fumaça", 1 -> "Fogo"
  
    # Confiança: certeza probabilística do modelo (de 0.0 a 1.0)
    confianca = float(box.conf[0])
  
    # Coordenadas da caixa: pixels inteiros na imagem original [x1, y1, x2, y2]
    # (x1, y1) = canto superior-esquerdo | (x2, y2) = canto inferior-direito
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
  
    print(f"-> {nome_classe:<8} | Certeza: {confianca:.1%} | Posição: [{x1}, {y1}, {x2}, {y2}]")
```

---

### Nível 3: Predição em Lote (Processando uma Pasta Inteira de Imagens)

Depois de entender o fluxo de uma imagem, o próximo passo natural é passar um diretório inteiro contendo dezenas ou centenas de fotos (como a pasta `input/`):

```python
from pathlib import Path
from ultralytics import YOLO

modelo = YOLO("runs/detect/experimentos/detector_v1/weights/best.pt")

# Passando o diretório como source:
resultados = modelo.predict(
    source="input/",       # Processa todas as imagens da pasta
    conf=0.35,             # Limiar ótimo indicado pela curva F1 do nosso treino
    save=True,             # Salva as fotos desenhadas
    project="output",      # Pasta de destino personalizada
    name="predicoes",      # Subpasta (output/predicoes/)
    exist_ok=True,         # Evita criar pastas duplicadas (predicoes2, predicoes3...)
    verbose=False          # Reduz poluição no terminal
)

# Agora a lista 'resultados' contém um item para cada foto processada:
for resultado in resultados:
    nome_arquivo = Path(resultado.path).name
    qtd = len(resultado.boxes)
    print(f"Arquivo: {nome_arquivo:<25} | Objetos: {qtd}")
```

---

### Nível 4: O Que Fazer com as Imagens? (Aplicações com os Resultados)

Com as coordenadas e as classes em mãos, você pode implementar regras de negócio reais ainda no ambiente de fotos:

#### 1. Auto-Crop de Evidências (Salvar recorte do foco)

O objeto `resultado.orig_img` contém a imagem original como matriz NumPy (formato BGR do OpenCV). Como temos as coordenadas exatas `[x1, y1, x2, y2]`, podemos fazer um *slicing* simples para recortar apenas a região da chama ou fumaça:

```python
import cv2

imagem_original = resultado.orig_img

for i, box in enumerate(resultado.boxes):
    classe_id = int(box.cls[0])
    nome = resultado.names[classe_id]
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
  
    # Slicing NumPy: [linhas, colunas] -> [y1:y2, x1:x2]
    recorte = imagem_original[y1:y2, x1:x2]
  
    # Salva apenas a evidência do foco:
    cv2.imwrite(f"output/evidencia_{nome.lower()}_{i}.jpg", recorte)
```

#### 2. Cerca Virtual (Áreas de Interesse / ROI)

Se a imagem for de uma fábrica com uma chaminé que produz fumaça normal de operação, podemos definir um polígono de segurança e ignorar qualquer detecção cujo centro esteja na chaminé, alarmando apenas se houver foco dentro da área de armazenamento crítico:

```python
import numpy as np

# Polígono da zona perigosa (ex: galpão de madeira):
zona_critica = np.array([[100, 200], [500, 200], [500, 600], [100, 600]], dtype=np.int32)

# Ponto central da caixa detectada:
xc, yc, w, h = box.xywh[0].int().tolist()

# Checa se o centro do fogo está dentro da zona crítica (>= 0 significa dentro ou na borda):
dentro_da_zona = cv2.pointPolygonTest(zona_critica, (xc, yc), False) >= 0
```

---

### Nível 5: Avançando para Vídeos, Câmeras e Tempo Real

Somente após dominar as predições estáticas em imagens, damos o salto para **fontes contínuas de vídeo**:

| Fonte                              | Sintaxe no `source`                  | Exemplo Prático                                        |
| :--------------------------------- | :------------------------------------- | :------------------------------------------------------ |
| **Arquivo de Vídeo**        | Caminho do vídeo (`.mp4`, `.avi`) | `source="videos/incidente.mp4"`                       |
| **Webcam USB / Integrada**   | Índice inteiro                        | `source=0` (webcam principal)                         |
| **Câmera IP de Segurança** | URL RTSP / RTMP                        | `source="rtsp://admin:123@192.168.1.100:554/stream1"` |

#### ⚠️ O Conceito Vital do `stream=True`

Em fotos, o Ultralytics pode acumular todas as respostas em uma lista na memória RAM.
Porém, em um vídeo de 15 minutos a 30 FPS (27.000 quadros), guardar 27.000 objetos `Results` na memória esgota toda a RAM e derruba o processo (*Out Of Memory*).

Por isso, em fluxos de vídeo contínuos usamos **sempre** `stream=True`, que transforma a chamada em um gerador Python com consumo fixo de memória ($O(1)$ de RAM):

```python
# O gerador entrega 1 quadro por vez e descarta o anterior da memória:
for resultado in modelo.predict(source="video.mp4", stream=True):
    # Processa o quadro atual
    pass
```

#### O Filtro de Persistência Temporal com `collections.deque` (Anti-Alarme Falso)

> [!IMPORTANT]
> **A Regra de Ouro da Visão Computacional para Atuadores Físicos:**
> Em um sistema conectado a uma **sirene física de 12V**, um relé ou uma válvula de gás, **nunca dispare uma ação física com base em um único quadro isolado**.
> Um reflexo de farol, uma folha amarela caindo, uma luz de solda ou um glitch na câmera pode gerar uma detecção espúria por exatamente 1 frame (33 milissegundos). Se a sirene tocar a cada fração de segundo por engano, ninguém confiará no sistema.

Para resolver isso de forma elegante, usamos uma **Fila Circular de Tamanho Fixo** com o `collections.deque` da biblioteca padrão do Python.

---

##### 1. O que é e como funciona o `collections.deque(maxlen=N)`?

Uma lista comum do Python cresce indefinidamente até a memória acabar. Já o `deque` com o parâmetro `maxlen` cria um buffer com capacidade máxima fixa.

Quando o buffer está cheio e você adiciona um novo elemento, **o elemento mais antigo é automaticamente descartado pela outra ponta** com complexidade de tempo instantânea $O(1)$:

```
Novo frame entra (Direita) ──> [ F, F, F, F, V, V, V, V, V ] ──> Frame mais antigo cai fora (Esquerda)
```

Veja esse teste simples no terminal Python:

```python
from collections import deque

# Cria um buffer que guarda no máximo 5 itens:
buffer = deque(maxlen=5)

buffer.append(False)
buffer.append(False)
buffer.append(True)
print(buffer)  # deque([False, False, True], maxlen=5)

buffer.append(True)
buffer.append(True)
print(buffer)  # deque([False, False, True, True, True], maxlen=5) - Encheu!

# Ao adicionar o próximo, o primeiro 'False' é expulso automaticamente:
buffer.append(True)
print(buffer)  # deque([False, True, True, True, True], maxlen=5)
```

---

##### 2. A Lógica da Janela Deslizante para a Sirene

Em uma câmera a 30 FPS, definimos um `deque(maxlen=30)`. Isso significa que o buffer sempre representa a **memória do último 1 segundo de vídeo**:

* A cada quadro analisado, perguntamos: *"Houve fogo com confiança $\ge 35\%$ neste quadro?"*

  * Se sim: `buffer.append(True)`
  * Se não: `buffer.append(False)`
* Como em Python `True == 1` e `False == 0`, para saber a porcentagem de quadros com fogo, basta fazer:

  $$
  \text{taxa} = \frac{\sum(\text{buffer})}{\text{len}(\text{buffer})}
  $$

##### Comparação de Cenários:

| Cenário Real                             | Conteúdo do Buffer (30 frames)                        | Taxa                | Decisão do Sistema                                            |
| :---------------------------------------- | :----------------------------------------------------- | :------------------ | :------------------------------------------------------------- |
| **Reflexo Momentâneo (Sol/Farol)** | 29 `False` e 1 `True`                              | $1/30 = 3.3\%$    | 🟢 **Ignora.** Sirene quieta (falso positivo filtrado).  |
| **Incêndio Real em Propagação**  | 5 `False` e 25 `True`                              | $25/30 = 83.3\%$  | 🚨 **DISPARA SIRENE!** Presença contínua e confirmada. |
| **Fogo Extinto / Normalizado**      | Conforme novos `False` entram, os `True` caem fora | Taxa cai $< 30\%$ | 🟢 **Silencia a sirene** automaticamente.                |

---

##### 3. Implementação Prática Pronta para Uso:

Veja como essa estrutura se encaixa de forma limpa no laço de inferência da câmera:

```python
from collections import deque
from ultralytics import YOLO

modelo = YOLO("runs/detect/experimentos/detector_v1/weights/best.pt")

# Buffer com histórico dos últimos 30 frames (~1 segundo a 30 FPS)
historico_fogo = deque(maxlen=30)
sirene_ativa = False

for resultado in modelo.predict(source=0, stream=True, show=True, verbose=False):
    # 1. Verifica se há alguma detecção de Fogo com confiança >= 35% no frame atual
    fogo_no_frame = any(
        resultado.names[int(box.cls[0])] == "Fogo" and float(box.conf[0]) >= 0.35
        for box in resultado.boxes
    )
  
    # 2. Registra no buffer (True se viu fogo, False se não viu)
    historico_fogo.append(fogo_no_frame)
  
    # 3. Calcula a consistência temporal da detecção
    # Só tomamos decisão quando o buffer já acumulou pelo menos 15 quadros de histórico:
    if len(historico_fogo) >= 15:
        taxa_presenca = sum(historico_fogo) / len(historico_fogo)
  
        # Se o fogo estiver presente em mais de 70% dos quadros recentes:
        if taxa_presenca >= 0.70:
            if not sirene_ativa:
                print(f"🚨🚨 [ALARME ATIVADO] Fogo confirmado com {taxa_presenca:.0%} de consistência!")
                # acionar_rele_sirene(True)
                sirene_ativa = True
  
        # Se a presença cair para menos de 20%, o fogo foi apagado/dissipado:
        elif taxa_presenca <= 0.20:
            if sirene_ativa:
                print("🟢 [ALARME DESATIVADO] Área normalizada.")
                # acionar_rele_sirene(False)
                sirene_ativa = False
```

---

### Nível 6: Exportação e Deploy Otimizado (ONNX & Edge Computing)

Quando o detector estiver validado e você for instalá-lo no computador embarcado (como um Raspberry Pi 5 ou Mini-PC antigo) conectado fisicamente à sirene via GPIO ou relé USB:

Você **não precisa** instalar o PyTorch completo (que consome gigabytes de disco e muita RAM). O formato **ONNX (Open Neural Network Exchange)** converte o modelo em um grafo puramente numérico e otimizado:

```mermaid
graph LR
    A["Treinamento PyTorch<br/>(best.pt - GPU)"] -->|"model.export(format='onnx')"| B["Grafo Otimizado<br/>(best.onnx - 12 MB)"]
    B --> C["Raspberry Pi / Mini-PC<br/>(ONNX Runtime)"]
    C -->|"Confirmação Temporal"| D["Relé / GPIO"]
    D --> E["Sirene Física 12V"]
```

#### 1. Exportando os pesos treinados para ONNX:

```python
from ultralytics import YOLO

modelo = YOLO("runs/detect/experimentos/detector_v1/weights/best.pt")

caminho_onnx = modelo.export(
    format="onnx",
    imgsz=640,
    dynamic=True,       # Permite batch size flexível
    simplify=True       # Otimiza operadores redundantes do grafo
)

print(f"Modelo otimizado pronto para produção em: {caminho_onnx}")
```

#### 2. Executando em produção com `onnxruntime` (Sem PyTorch nem Ultralytics):

No dispositivo final (Raspberry Pi ou Mini-PC), o objetivo é economizar memória RAM e disco. Você **não precisa instalar o PyTorch nem a biblioteca Ultralytics**.

Basta instalar o runtime da Microsoft e o OpenCV:

```bash
pip install onnxruntime opencv-python-headless numpy
```

> O ambiente inteiro ocupa menos de **50 MB** (contra mais de 3 GB a 5 GB do PyTorch!).

Aqui está o script de inferência 100% puro com `onnxruntime`:

```python
"""
producao_onnx.py
Inferência leve em produção no Raspberry Pi / Mini-PC com ONNX Runtime.
Sem dependências pesadas de PyTorch ou Ultralytics.
"""
import cv2
import numpy as np
import onnxruntime as ort

# 1. Carrega o grafo estático no ONNX Runtime (usando CPUExecutionProvider)
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
in_name = session.get_inputs()[0].name    # 'images'
out_name = session.get_outputs()[0].name  # 'output0'

# Mapeamento das classes do modelo
NOMES = {0: "Fumaça", 1: "Fogo"}

def inferir_frame(frame, conf_min=0.35, iou_nms=0.50):
    h_orig, w_orig = frame.shape[:2]

    # 2. Pré-processamento do quadro:
    # Resize para 640x640, converte BGR->RGB, normaliza para [0, 1] e muda formato para (1, 3, 640, 640)
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

    # 3. Inferência pura com ONNX Runtime:
    saida = session.run([out_name], {in_name: img_tensor})[0]  # Shape: (1, 6, 8400)
  
    # Transpõe para (8400, 6) -> Cada linha: [xc, yc, w, h, score_fumaca, score_fogo]
    predicoes = np.transpose(saida[0])

    caixas = []
    confiancas = []
    classe_ids = []

    # 4. Extração de candidatos com confiança acima do limiar
    for linha in predicoes:
        scores_classes = linha[4:]
        melhor_conf = float(np.max(scores_classes))

        if melhor_conf >= conf_min:
            cls_id = int(np.argmax(scores_classes))
            xc, yc, w, h = linha[:4]

            # Converte de 640x640 de volta para as dimensões originais da câmera:
            x1 = int((xc - w / 2) * (w_orig / 640))
            y1 = int((yc - h / 2) * (h_orig / 640))
            largura = int(w * (w_orig / 640))
            altura = int(h * (h_orig / 640))

            caixas.append([x1, y1, largura, altura])
            confiancas.append(melhor_conf)
            classe_ids.append(cls_id)

    # 5. Non-Maximum Suppression (NMS) nativo do OpenCV (elimina caixas redundantes)
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, conf_min, iou_nms)

    deteccoes = []
    if len(indices) > 0:
        for i in indices.flatten():
            deteccoes.append({
                "classe": NOMES[classe_ids[i]],
                "confianca": confiancas[i],
                "caixa": caixas[i]  # [x1, y1, w, h]
            })

    return deteccoes


# Exemplo de teste simples:
if __name__ == "__main__":
    frame_teste = cv2.imread("input/foto_teste.jpg")
    if frame_teste is not None:
        resultados = inferir_frame(frame_teste)
        print(f"Detecções encontradas: {len(resultados)}")
        for det in resultados:
            print(f" -> {det['classe']} ({det['confianca']:.1%}) em {det['caixa']}")
```
