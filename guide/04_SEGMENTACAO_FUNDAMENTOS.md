# 📦 Guia 01: Fundamentos da Segmentação & O Padrão YOLO-seg

> **Prioridade:** ⭐ **Máxima (O Núcleo da Visão Computacional Geométrica)**
> **Objetivo:** Compreender a evolução da detecção para a **segmentação de instâncias**, dominar a estrutura matemática dos **polígonos normalizados no formato YOLO-seg**, configurar os diretórios e o arquivo **`data.yaml`**, e interpretar as métricas de **Mask mAP** vs **Box mAP**.

---

## 1. De Detecção para Segmentação: A Evolução Espacial

Na visão computacional clássica e na detecção de objetos tradicional, a saída espacial máxima era uma **Bounding Box** retangular:

```
[ Classificação ]             [ Detecção de Objetos ]             [ Segmentação de Instâncias ]
"O que tem na cena?"          "O que tem e onde está?"            "O que tem, onde está e qual é
                                                                   o contorno exato de cada um?"

┌───────────────────┐         ┌───────────────────┐               ┌───────────────────┐
│                   │         │   ┌───────────┐   │               │       /████\      │
│     Gato          │         │   │   Gato    │   │               │      /██████\     │
│    (0.95)         │         │   │  (0.92)   │   │               │     |████████|    │
│                   │         │   └───────────┘   │               │      \██/  \██/   │
└───────────────────┘         └───────────────────┘               └───────────────────┘
 1 Rótulo Global               Caixa [xc, yc, w, h]                Polígono [x1, y1, ..., xn, yn]
```

### O Problema da Bounding Box no Mundo Real

A caixa delimitadora é uma aproximação grosseira. Ela sofre de limitações críticas:

1. **Poluição por Fundo (Background Noise):** Se um objeto for diagonal (ex: uma folha caída, um braço levantado, uma rachadura ou uma ferramenta), mais de **50% a 70% dos pixels** dentro da caixa retangular pertencem ao fundo, e não ao objeto.
2. **Sobreposição Complexa:** Se dois objetos estiverem entrelaçados (ex: duas pessoas se abraçando, engrenagens acopladas, frutos em um cacho), suas bounding boxes colidem fortemente, confundindo o modelo e o pós-processamento.
3. **Incapacidade de Medição Físico-Métrica:** É impossível medir a área real, o perímetro, a curvatura ou a orientação de um objeto usando apenas largura e altura de um retângulo.

> 💡 **A Solução da Segmentação de Instâncias:** Cada objeto detectado ganha sua própria **máscara binária independente** que isola pixel a pixel a sua silhueta precisa, separando instâncias vizinhas da mesma classe.

---

## 2. Anatomia do Formato de Anotação YOLO-seg

Na detecção YOLO convencional, cada linha do arquivo `.txt` continha exatamente 5 números:

```text
<classe_id> <x_centro> <y_centro> <largura> <altura>
```

Na **Segmentação de Instâncias YOLO (`-seg`)**, a linha passa a conter o identificador da classe seguido de uma sequência de **pares de coordenadas $[x, y]$ dos vértices do polígono** que contorna o objeto:

```text
<classe_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> ... <xn> <yn>
```

### Regras Fundamentais do Formato:

1. **Normalização Total no Intervalo $[0.0, 1.0]$:**Assim como na detecção, todos os pontos $x$ são divididos pela largura total da imagem ($W$) e todos os pontos $y$ são divididos pela altura total ($H$):
   $$
   x_i = \frac{\text{pixel\_x}_i}{W}, \quad y_i = \frac{\text{pixel\_y}_i}{H}
   $$
2. **Quantidade de Vértices Flexível:**Diferente da caixa fixa de 4 números, um polígono pode ter quantos vértices forem necessários para contornar a forma com fidelidade.
   - **Regra 80/20 de Vértices:** Em média, de **15 a 35 pontos** são suficientes para contornar perfeitamente 95% dos objetos sem sobrecarregar o tamanho dos arquivos nem a memória de treino.
3. **Múltiplos Objetos:**Cada linha do arquivo `.txt` representa uma instância distinta. Se a foto tiver 2 copos e 1 prato, haverá 3 linhas no arquivo.
4. **E a Bounding Box, preciso anotar?**
   **NÃO!** O compilador do Ultralytics extrai automaticamente a bounding box calculando os valores extremos do polígono:
   $$
   x_{\min} = \min(x_1, \dots, x_n), \quad x_{\max} = \max(x_1, \dots, x_n)
   $$

   Portanto, um dataset de segmentação **serve automaticamente tanto para segmentação quanto para detecção**.

---

### Exemplo Numérico Passo a Passo:

Imagine uma imagem de resolução $1000 \times 1000\text{ pixels}$.Temos um objeto da classe `0` (ex: uma maçã) cujos vértices do contorno foram marcados nos seguintes pixels:

- Ponto 1: $(500, 200)$
- Ponto 2: $(700, 400)$
- Ponto 3: $(650, 700)$
- Ponto 4: $(500, 800)$
- Ponto 5: $(350, 700)$
- Ponto 6: $(300, 400)$

#### 1. Conversão para valores normalizados $[0.0, 1.0]$:

- Ponto 1: $500/1000 = 0.5000$, $200/1000 = 0.2000$
- Ponto 2: $700/1000 = 0.7000$, $400/1000 = 0.4000$
- Ponto 3: $650/1000 = 0.6500$, $700/1000 = 0.7000$
- Ponto 4: $500/1000 = 0.5000$, $800/1000 = 0.8000$
- Ponto 5: $350/1000 = 0.3500$, $700/1000 = 0.7000$
- Ponto 6: $300/1000 = 0.3000$, $400/1000 = 0.4000$

#### 2. Linha exata salva no arquivo `labels/train/maca_01.txt`:

```text
0 0.5000 0.2000 0.7000 0.4000 0.6500 0.7000 0.5000 0.8000 0.3500 0.7000 0.3000 0.4000
```

---

### Comparativo: Padrões de Mercado para Segmentação

| Formato                                  | Representação                                                                      | Onde é usado                             | Vantagens / Desvantagens                                                              |
| :--------------------------------------- | :----------------------------------------------------------------------------------- | :---------------------------------------- | :------------------------------------------------------------------------------------ |
| **YOLO-seg TXT**                   | `classe x1 y1 x2 y2 ...` em $[0, 1]$                                             | YOLOv8-seg, YOLO11-seg, Ultralytics       | 🟢 Ultra legível, leve, 1 arquivo por imagem, fácil inspeção.                     |
| **COCO JSON (Polygons)**           | Arquivo único JSON com chaves `segmentation: [[x1, y1, ...]]` em pixels absolutos | Detectron2, MMDetection, TorchVision      | 🟡 Padrão acadêmico global, mas um único JSON corrompido quebra o dataset inteiro. |
| **COCO RLE (Run Length Encoding)** | Strings codificadas comprimindo a máscara binária de pixels                        | Datasets densos de multidão (CrowdHuman) | 🟡 Eficiente para objetos cheios de buracos, mas ilegível para humanos.              |
| **Máscara PNG (Color Map)**       | Imagens PNG 2D onde cada pixel guarda o ID da classe                                 | Segmentação Semântica (UNet, DeepLab)  | 🔴 Não separa instâncias que se tocam da mesma classe.                              |

---

## 3. A Estrutura de Diretórios e o Arquivo `data.yaml`

A organização de diretórios para segmentação segue rigorosamente a mesma convenção do ecossistema YOLO:

```text
projeto_segmentacao/
│
├── data.yaml                     # Configuração central de caminhos e nomes
└── dataset/
    ├── images/
    │   ├── train/                # Ex: folha01.jpg, folha02.jpg
    │   └── val/                  # Ex: folha50.jpg, folha51.jpg
    └── labels/
        ├── train/                # Ex: folha01.txt, folha02.txt (formato poligonal!)
        └── val/                  # Ex: folha50.txt, folha51.txt
```

> **Atenção:** O nome do arquivo `.txt` deve ser exatamente idêntico ao nome da imagem (mudando apenas a extensão). Imagens sem nenhum objeto da classe alvo devem ter um arquivo `.txt` vazio para ensinar o modelo a não gerar falsos positivos.

### Exemplo de `data.yaml`:

```yaml
# Caminho raiz absoluto ou relativo ao diretório onde o comando é executado
path: ./dataset

# Subpastas relativas ao 'path' acima
train: images/train
val: images/val
test: images/test # opcional

# Dicionário de classes
names:
  0: folha_saudavel
  1: lesao_ferrugem
  2: queima_borda
```

---

## 4. Métricas de Avaliação Espacial: Box mAP vs Mask mAP

Ao treinar ou validar um modelo de segmentação, você verá no terminal e nos relatórios de validação duas famílias de métricas simultâneas:

1. **`metrics.box` (Box mAP):** Avalia a precisão e revocação da bounding box externa calculada pelo modelo.
2. **`metrics.seg` (Mask mAP):** Avalia a precisão e revocação da **máscara de pixels real**.

### O que é o Mask IoU?

Na detecção, o IoU media a sobreposição de retângulos. Na segmentação, o **Mask IoU** mede a sobreposição real pixel a pixel:

$$
\text{Mask IoU} = \frac{\text{Pixels da Máscara Prevista} \cap \text{Pixels da Máscara Real}}{\text{Pixels da Máscara Prevista} \cup \text{Pixels da Máscara Real}}
$$

```
   Máscara Prevista (A)        Máscara Real (B)                Interseção (A ∩ B)
         /████\                    /████\                           /████\
        /██████\                  /██████\                         /██████\
       |████████|                |████████|                       |████████|
        \██/  \██/                \██████/                         \██/  \██/
```

### Por que o Mask mAP costuma ser menor que o Box mAP?

É comum ver em relatórios:

- `Box mAP50`: **0.91**
- `Mask mAP50`: **0.86**

Isso é perfeitamente normal e esperado! Prever um retângulo que englobe grosseiramente o objeto é muito mais fácil do que delinear cada curva, concavidade e detalhe fino da silhueta dos pixels.

- **Mask mAP@50 (mAP_0.5):** O modelo é considerado certeiro se sua máscara sobrepuser pelo menos $50\%$ da máscara real. Excelente para checar se a segmentação está funcional.
- **Mask mAP@50-95:** A média calculada em múltiplos patamares rigorosos de sobreposição de silhueta ($50\%$ até $95\%$). Valores acima de **$0.35$ a $0.45$** já representam um modelo de segmentação de nível comercial avançado.

---

## 5. O Atalho 80/20: Ferramentas de Anotação Assistidas por IA

Anotar caixas de detecção leva 3 segundos por objeto (dois cliques). Anotar polígonos ponto a ponto manualmente pode levar de **30 a 60 segundos por objeto**.Para não perder semanas nessa etapa, adote ferramentas modernas com **IA assistida**:

1. **Roboflow com Smart Polygon (Recomendação Principal):**
   - Você sobe as fotos e usa a ferramenta **Smart Polygon**. Basta clicar uma vez no centro do objeto e o modelo foundation detecta as bordas e gera o polígono automaticamente.
   - Exportação direta com 1 clique no formato **"YOLOv8 Segmentation"** (já entrega as pastas `images`, `labels` em polígonos normalizados e o `data.yaml` pronto).
2. **X-AnyLabeling (Open-Source Desktop):**
   - Ferramenta local e gratuita instalável via `pip install anylabeling`.
   - Integra modelos como **Segment Anything (SAM)** rodando na sua própria máquina: basta passar o mouse sobre o objeto para gerar o polígono YOLO-seg instantaneamente.
3. **Roboflow Universe:**
   - Milhares de datasets de segmentação prontos e públicos para testar pipelines antes mesmo de coletar suas próprias imagens.
