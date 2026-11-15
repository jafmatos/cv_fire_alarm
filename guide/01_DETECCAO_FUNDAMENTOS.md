# 📦 Guia 01: Fundamentos da Detecção & O Padrão YOLO

> **Prioridade:** ⭐ **Máxima (O Núcleo da Visão Computacional Espacial)**
> **Objetivo:** Compreender a transição matemática e conceitual entre a classificação de imagens e a detecção de objetos, dominar o formato de anotação normalizado do **YOLO**, estruturar os diretórios e o arquivo **`data.yaml`**, e interpretar com clareza as métricas espaciais (**IoU** e **mAP**).

---

## 1. De Classificação para Detecção: O que Muda na Prática?

Em modelos de classificação como ResNet ou MobileNet, a imagem inteira é compactada por camadas convolucionais até um vetor 1D, gerando uma distribuição de probabilidade para uma única etiqueta:

```
[ Classificação ]
Imagem (3 x 224 x 224) ──► Convoluções ──► Pooling ──► Linear(num_classes)
                                                          │
                                                          ▼
                                                  "Cachorro: 0.92"
```

Na **Detecção de Objetos com YOLO (You Only Look Once)**, a rede divide a imagem em uma grade (*grid*) e faz predições espaciais contínuas. Para cada região de interesse, ela prevê um vetor contendo tanto o local exato quanto a probabilidade de classe:

```
[ Detecção de Objetos ]
Imagem (3 x 640 x 640) ──► Backbone (CSP/C2f) ──► Neck (PANet) ──► Detection Head
                                                                         │
                                                                         ▼
                                                     ┌────────────────────────────────────────┐
                                                     │ Vetor de Predição por Objeto:          │
                                                     │ [x_center, y_center, w, h, conf, cls]  │
                                                     └────────────────────────────────────────┘
```

A rede não responde apenas *o que* está na imagem, mas *quantos são*, *onde estão* e o *tamanho exato* de cada um.

---

## 2. Anatomia da Bounding Box no Formato YOLO

A maior causa de falha em modelos de detecção de iniciantes está em discrepâncias na anotação dos dados. O formato YOLO utiliza uma convenção rigorosa e elegante: **todos os valores são relativos ao tamanho da imagem e normalizados no intervalo $[0.0, 1.0]$**.

### Formato do arquivo `.txt`:

Para cada imagem `foto_001.jpg`, deve haver um arquivo correspondente `foto_001.txt` no diretório de rótulos:

```text
<classe_id> <x_centro> <y_centro> <largura> <altura>
```

Se uma imagem contiver três objetos, o arquivo `.txt` terá três linhas:

```text
0 0.485123 0.612456 0.231450 0.450120
1 0.781250 0.354167 0.125000 0.208333
0 0.156250 0.729167 0.187500 0.312500
```

### Conversão Matemática: De Pixels Absolutos para YOLO Normalizado

Se você tiver as coordenadas de uma caixa em pixels absolutos $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ em uma imagem de largura $W$ e altura $H$:

$$
x_{\text{centro}} = \frac{x_{\min} + x_{\max}}{2 \times W}
$$

$$
y_{\text{centro}} = \frac{y_{\min} + y_{\max}}{2 \times H}
$$

$$
\text{largura} = \frac{x_{\max} - x_{\min}}{W}
$$

$$
\text{altura} = \frac{y_{\max} - y_{\min}}{H}
$$

#### Exemplo Prático com Números Reais:

* **Dimensões da Imagem:** $W = 1920\text{ px}$, $H = 1080\text{ px}$
* **Coordenadas do Objeto (gato):**
  * $x_{\min} = 480$, $y_{\min} = 270$
  * $x_{\max} = 1440$, $y_{\max} = 810$
* **Cálculo da Caixa:**
  * $x_{\text{centro}} = \frac{480 + 1440}{2 \times 1920} = \frac{1920}{3840} = 0.5000$
  * $y_{\text{centro}} = \frac{270 + 810}{2 \times 1080} = \frac{1080}{2160} = 0.5000$
  * $\text{largura} = \frac{1440 - 480}{1920} = \frac{960}{1920} = 0.5000$
  * $\text{altura} = \frac{810 - 270}{1080} = \frac{540}{1080} = 0.5000$
* **Linha salva no arquivo de anotação:**
  ```text
  0 0.500000 0.500000 0.500000 0.500000
  ```

### Comparativo: Padrões de Mercado para Bounding Boxes

| Formato                    | Estrutura                                    | Unidade                | Onde é usado                                       |
| :------------------------- | :------------------------------------------- | :--------------------- | :-------------------------------------------------- |
| **YOLO**             | `[cls, x_center, y_center, width, height]` | Normalizado $[0, 1]$ | YOLOv5, YOLOv8, YOLO11, Darknet                     |
| **COCO JSON**        | `[x_min, y_min, width, height]`            | Pixels absolutos       | TorchVision detection, MMDetection, Detectron2      |
| **Pascal VOC (XML)** | `[x_min, y_min, x_max, y_max]`             | Pixels absolutos       | Datasets acadêmicos clássicos, ImageNet detection |

---

## 3. A Estrutura de Diretórios e o Arquivo `data.yaml`

Ao contrário do `ImageFolder` da classificação (onde as classes eram nomes de pastas), no YOLO as imagens e anotações ficam organizadas em pastas irmãs com o **mesmo nome base de arquivo**:

```text
projeto_deteccao/
│
├── data.yaml                     # Arquivo de configuração de caminhos e classes
└── dataset/
    ├── images/
    │   ├── train/                # Ex: img01.jpg, img02.jpg
    │   └── val/                  # Ex: img50.jpg, img51.jpg
    └── labels/
        ├── train/                # Ex: img01.txt, img02.txt (mesmo nome das fotos!)
        └── val/                  # Ex: img50.txt, img51.txt
```

> **Regra Fundamental:** Se uma imagem de treino não tiver nenhum objeto de interesse (imagem de fundo / falso positivo negativo), você deve manter o arquivo `.txt` vazio ou simplesmente não criar o `.txt`. O YOLO interpretará automaticamente como imagem negativa (sem objetos).

### O Arquivo `data.yaml`

Este arquivo centraliza as informações do dataset para o compilador do YOLO:

```yaml
# Caminho base absoluto ou relativo ao local onde o treino é executado
path: ./dataset

# Subpastas relativas ao 'path' acima
train: images/train
val: images/val
test: images/test   # opcional

# Dicionário mapeando os IDs numéricos aos nomes legíveis
names:
  0: pessoa
  1: capacete
  2: colete
```

---

## 4. Métricas de Avaliação Espacial: IoU e mAP

Na classificação, medimos apenas se a classe prevista bate com a classe real. Na detecção, temos dois critérios simultâneos: **localização correta** e **classificação correta**.

### IoU (Intersection over Union)

O IoU quantifica a precisão geométrica da caixa prevista em relação à caixa verdadeira (*Ground Truth*):

```
       Caixa Prevista (A)
         ┌───────────────┐
         │               │
    ┌────┼──────────┐    │
    │    │ //////// │    │   Área com //////// = Interseção (A ∩ B)
    │    │ //////// │    │
    │    └──────────┼────┘
    │               │
    └───────────────┘
     Caixa Real (B)

                 Área da Interseção (A ∩ B)
     IoU = ──────────────────────────────────────
                    Área da União (A ∪ B)
```

* **IoU = 1.0**: Sobreposição perfeita.
* **IoU $\ge$ 0.50**: Padrão histórico aceito para considerar uma detecção "correta".
* **IoU $\ge$ 0.75**: Detecção de alta precisão métrica.

### mAP (Mean Average Precision)

O **mAP** resume a curva de Precisão vs. Revocação para todas as classes em diferentes limiares de IoU:

* **mAP@50 (ou mAP_0.5)**: Avalia a precisão considerando acerto qualquer predição com $\text{IoU} \ge 0.50$. É a métrica mais amigável e ótima para verificar se o modelo aprendeu a encontrar os objetos.
* **mAP@50-95**: Média do mAP calculado em 10 limiares progressivos de IoU ($0.50, 0.55, 0.60, \dots, 0.95$). É a métrica padrão e rigorosa da competição COCO. Valores acima de **$0.40$ (40%)** nessa métrica já indicam modelos de excelente nível industrial.

---

## 5. O Atalho 80/20: Ferramentas de Anotação

Não perca tempo convertendo arquivos JSON manualmente com scripts Python. Utilize ferramentas especializadas que exportam diretamente no padrão YOLO:

1. **Roboflow (Recomendado para início imediato)**:
   * Interface web intuitiva para arrastar imagens e desenhar as caixas.
   * Suporte a download no formato **YOLOv8 PyTorch** (já cria as pastas `images`, `labels` e o `data.yaml` compactados em `.zip`).
   * Contém o **Roboflow Universe**, repositório público com milhares de datasets gratuitos já anotados (de células microscópicas a veículos e plantas).
2. **CVAT (Computer Vision Annotation Tool)** ou **Label Studio**:
   * Alternativas open-source gratuitas para rodar localmente via Docker quando há restrições de privacidade de dados.
