# 👁️ Guia 02: OpenCV Essencial para YOLO & Segmentação (80/20)

> **Prioridade:** ⭐ **Máxima (As Mãos da Visão Computacional)**  
> **Objetivo:** Dominar o núcleo prático da biblioteca **OpenCV (`cv2`)** necessário para manipular imagens, vídeos, câmeras e as máscaras geométricas devolvidas pelo YOLO. Em vez de estudar centenas de filtros legados, este guia foca nas **10 funções vitais que resolvem 95% dos problemas de visão computacional**.

---

## 🗺️ A Divisão de Trabalho: YOLO vs. OpenCV

```
┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
│             O YOLO                   │          │             O OPENCV                 │
│          (O CÉREBRO)                 │          │              (AS MÃOS)               │
├──────────────────────────────────────┤          ├──────────────────────────────────────┤
│ • Encontra onde estão os objetos.    │ ───────► │ • Abre câmeras, webcams e vídeos.    │
│ • Prevê classes e confianças.        │  Dados   │ • Desenha contornos, caixas e HUDs.  │
│ • Devolve polígonos [x1, y1... xn, yn│ Brutos   │ • Recorta silhuetas e apaga o fundo. │
│   e matrizes binárias de pixels.     │          │ • Calcula áreas e cercas virtuais.   │
└──────────────────────────────────────┘          └──────────────────────────────────────┘
```

---

## 1. Como o OpenCV Enxerga o Mundo: Matrizes NumPy

Para o OpenCV, **não existe "arquivo de imagem" na memória**. Uma imagem é estritamente uma matriz numérica multidimensional do **NumPy** (`uint8`, com números de 0 a 255):

```python
import cv2

imagem = cv2.imread("input/bus.jpg")

print(type(imagem))        # <class 'numpy.ndarray'>
print(imagem.shape)        # (1080, 1920, 3) -> (Altura, Largura, 3 Canais)
print(imagem.dtype)        # uint8 (valores inteiros de 0 a 255)
```

### ⚠️ A Pegadinha Histórica: BGR em vez de RGB

No mundo digital e na web, as cores são ordenadas como **RGB** (Vermelho, Verde, Azul).  
Por motivos históricos de quando as câmeras analógicas eram conectadas a computadores nos anos 90, o **OpenCV lê as cores em ordem inversa: BGR (Azul, Verde, Vermelho)**!

```python
# Se você abrir no OpenCV e plotar direto no Matplotlib, fica tudo azulado:
# Para converter para RGB correto:
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Para converter em Tons de Cinza (1 único canal):
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Shape vira (1080, 1920)
```

---

## 2. Ingestão, Vídeo e Câmera ao Vivo

O ciclo básico de leitura e gravação que sustenta qualquer sistema industrial:

```python
import cv2

# 1. Leitura e Gravação básica
img = cv2.imread("input/foto.jpg")
cv2.imwrite("output/resultado.jpg", img)

# 2. Redimensionamento
# ATENÇÃO: No resize, a ordem é (Largura, Altura), o inverso do shape do numpy!
img_pequena = cv2.resize(img, (640, 480))

# 3. Captura de Câmera ou Vídeo
# Use 0 para webcam USB, ou caminho do arquivo, ou "rtsp://..." para câmera de segurança IP
cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    # Aqui você joga o frame no seu modelo YOLO!
    # resultado = model.predict(frame)[0]

    cv2.imshow("Monitoramento ao Vivo", frame)
    
    # Sai do loop se pressionar a tecla 'q' (espera 1 milissegundo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 3. Desenho e Anotação Visual (Para Bounding Boxes)

Quando você quer criar suas próprias caixas personalizadas na imagem em vez de usar o visual padrão do YOLO:

```python
import cv2

img = cv2.imread("input/bus.jpg")

# 1. Desenhar Bounding Box retangular
# pt1 = (x1, y1) [canto superior esquerdo]
# pt2 = (x2, y2) [canto inferior direito]
# cor = (B, G, R) -> (0, 255, 0) é Verde Puro
# espessura = 2 pixels
cv2.rectangle(img, (100, 150), (400, 500), (0, 255, 0), 2)

# 2. Escrever Texto / Rótulo
cv2.putText(
    img, 
    "Onibus: 95%", 
    (100, 140),                 # Posição (x, y) do texto
    cv2.FONT_HERSHEY_SIMPLEX,   # Tipo de fonte
    0.6,                        # Escala do tamanho
    (0, 255, 0),                # Cor BGR
    2                           # Espessura do traço
)

# 3. Desenhar um Ponto / Círculo (Útil para marcar centróides ou rodas)
# -1 significa círculo preenchido
cv2.circle(img, (250, 325), 5, (0, 0, 255), -1)  # Ponto vermelho
```

---

## 4. O Arsenal Específico da Segmentação (Máscaras de Pixels)

Aqui está o verdadeiro tesouro do OpenCV para quem trabalha com **YOLO-seg**:

### A. Criando a Máscara Binária a partir do Polígono (`cv2.fillPoly`)

O YOLO-seg devolve uma lista de pontos $[x, y]$ da silhueta em `result.masks.xy`.  
Para transformar essa lista de pontos em uma máscara de pixels pretos e brancos:

```python
import numpy as np
import cv2

h, w = img.shape[:2]

# Cria uma tela 100% preta (zeros) de 1 canal (escala de cinza)
mascara_binaria = np.zeros((h, w), dtype=np.uint8)

# Pega o polígono do primeiro objeto detectado pelo YOLO
poligono = resultado.masks.xy[0]  # Array numpy com shape (N, 2)

# Converte para inteiros e preenche o interior de branco (255)
pontos = np.int32([poligono])
cv2.fillPoly(mascara_binaria, pontos, 255)

# Agora:
# Pixels = 255 -> Pertencem ao objeto (branco)
# Pixels = 0   -> Pertencem ao fundo (preto)
```

---

### B. Recorte Cirúrgico do Objeto (`cv2.bitwise_and`)

Como apagar o fundo da foto e manter **apenas o objeto** intacto com suas cores originais:

```python
# O bitwise_and multiplica a imagem pela máscara:
# Onde a máscara for 255 (branco), mantém a cor original.
# Onde a máscara for 0 (preto), transforma em preto total.
objeto_isolado = cv2.bitwise_and(img, img, mask=mascara_binaria)

cv2.imwrite("output/objeto_isolado.jpg", objeto_isolado)
```

---

### C. Máscara Translúcida Estilo HUD / Videogame (`cv2.addWeighted`)

Se você quer pintar o objeto com uma cor semi-transparente (como o YOLO faz no `save=True`), mas com as suas próprias cores personalizadas:

```python
overlay = img.copy()

# Pinta o objeto de ciano luminoso (B=255, G=255, R=0) no overlay
cv2.fillPoly(overlay, pontos, (255, 255, 0))

# Mistura a imagem original (70%) com o overlay pintado (30%)
# alpha = 0.7, beta = 0.3, gamma = 0
resultado_translucido = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
```

---

### D. Exportar PNG com Fundo Transparente Real (Canal Alfa)

Para criar imagens prontas para design ou e-commerce sem nenhum fundo:

```python
# 1. Separa os 3 canais BGR da foto original
b, g, r = cv2.split(img)

# 2. Junta os 3 canais com a máscara como 4º canal (Alpha/Transparência)
bgra = cv2.merge([b, g, r, mascara_binaria])

# 3. Salva em formato PNG (JPG não suporta transparência!)
cv2.imwrite("output/objeto_transparente.png", bgra)
```

---

## 5. Geometria Espacial & Inteligência de Negócio

### A. Centro de Massa Real (Centróide com `cv2.moments`)

Em peças curvas ou pessoas em posições inclinadas, o centro da Bounding Box cai no ar vazio. A máscara entrega o centróide ponderado exato:

```python
# Calcula os momentos estatísticos da área branca
momentos = cv2.moments(mascara_binaria)

if momentos["m00"] != 0:
    centro_x = int(momentos["m10"] / momentos["m00"])
    centro_y = int(momentos["m01"] / momentos["m00"])
    print(f"Centro de massa real: X={centro_x}, Y={centro_y}")
```

---

### B. Cerca Virtual Poligonal (`cv2.pointPolygonTest`)

Imagine que você desenhou uma área proibida (ex: faixa de pedestre, trilho de trem, caldeira de fábrica):

```python
# Polígono da zona restrita definido manualmente (ex: 4 cantos da calçada)
zona_proibida = np.array([[200, 400], [600, 400], [550, 700], [150, 700]], dtype=np.int32)

# cv2.pointPolygonTest(poligono, ponto_xy, measureDist=False):
# Retorna:
#   +1 -> Ponto está DENTRO do polígono
#    0 -> Ponto está EXATAMENTE na borda
#   -1 -> Ponto está FORA do polígono

invadiu = cv2.pointPolygonTest(zona_proibida, (centro_x, centro_y), False) >= 0

if invadiu:
    print("🚨 ALERTA: Objeto invadiu a zona proibida!")
```

---

## 📋 Tabela Resumo: O Canivete Suíço 80/20 do OpenCV

| Objetivo no Projeto | Função do OpenCV | Parâmetros Chave |
| :--- | :--- | :--- |
| **Carregar foto** | `cv2.imread(caminho)` | Devolve matriz NumPy `uint8` em BGR. |
| **Salvar imagem** | `cv2.imwrite(caminho, img)` | Salva PNG ou JPG no disco. |
| **Redimensionar** | `cv2.resize(img, (W, H))` | Atenção: largura vem antes da altura! |
| **Criar máscara sólida** | `cv2.fillPoly(mascara, [pontos], 255)` | Preenche o interior do contorno com branco. |
| **Traçar contorno fino** | `cv2.polylines(img, [pontos], True, cor, 2)` | Desenha apenas a linha da silhueta. |
| **Recortar sem fundo** | `cv2.bitwise_and(img, img, mask=masc)` | Apaga todo o fundo da imagem. |
| **Efeito translúcido** | `cv2.addWeighted(img, 0.7, over, 0.3, 0)` | Mistura cores com transparência elegante. |
| **Transparência PNG** | `cv2.merge([b, g, r, mascara])` | Adiciona o 4º canal Alfa para salvar `.png`. |
| **Centro de Massa** | `cv2.moments(mascara)` | Extrai o baricentro geométrico exato. |
| **Cerca Virtual** | `cv2.pointPolygonTest(zona, ponto, False)` | Checa invasão espacial de perímetros. |
