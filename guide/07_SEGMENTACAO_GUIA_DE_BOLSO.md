# 📋 Guia de Bolso: YOLO Segmentação 80/20

> **Objetivo:** O resumo de consulta rápida mais objetivo para manter aberto ao lado do seu editor de código durante o desenvolvimento de modelos de **segmentação de instâncias com YOLO**.

---

## 1. Estrutura de Dados & Formato Poligonal

| Cenário / Objetivo                 | Formato / Snippet                                        | O que resolve                                                                                     |
| :--------------------------------- | :------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| **Linha de anotação YOLO-seg**     | `classe_id x1 y1 x2 y2 ... xn yn`                        | Formato padrão com vértices poligonais normalizados em $[0, 1]$.                                  |
| **Normalizar vértice individual**  | `xn = px / W; yn = py / H`                               | Torna o polígono independente da resolução nativa da foto.                                        |
| **Configuração mínima do dataset** | Arquivo `data.yaml` com `path`, `train`, `val` e `names` | Centraliza metadados e mapeamento numérico das classes.                                           |
| **Imagem de Fundo (Sem objetos)**  | Arquivo `.txt` vazio com mesmo nome da foto              | Ensina o modelo a suprimir falsos positivos em áreas vazias.                                      |
| **Bounding Box automática**        | Sem anotação extra necessária                            | O compilador deduz os limites $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ diretamente do polígono. |

---

## 2. Treinamento & Pesos Pré-Treinados (`-seg`)

| Cenário / Objetivo                   | Snippet de Código                                            | O que resolve                                                      |
| :----------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------------- |
| **Carregar rede pré-treinada**       | `model = YOLO("yolov8n-seg.pt")`                             | Baixa filtros convolucionais e a ProtoNet treinados no COCO.       |
| **Treino 80/20 balanceado**          | `model.train(data="data.yaml", epochs=50, imgsz=640)`        | Dispara o fine-tuning adaptando as máscaras às suas novas classes. |
| **Evitar estouro de memória (VRAM)** | `model.train(..., batch=8, workers=2)`                       | Reduz pressão em GPUs com 4GB a 6GB ou em máquinas modestas.       |
| **Early Stopping anti-overfitting**  | `model.train(..., patience=15)`                              | Interrompe o treino se o `Mask mAP` não melhorar por 15 épocas.    |
| **Treinar na CPU (Sem GPU)**         | `model.train(..., device="cpu", batch=4)`                    | Permite validar o pipeline localmente sem placas CUDA.             |
| **Retomar treino interrompido**      | `model = YOLO("runs/.../last.pt"); model.train(resume=True)` | Continua o treinamento da época exata onde houve interrupção.      |

---

## 3. Inferência & Acesso aos Dados das Máscaras

| Cenário / Objetivo                     | Snippet de Código                                           | O que resolve                                                            |
| :------------------------------------- | :---------------------------------------------------------- | :----------------------------------------------------------------------- |
| **Carregar pesos otimizados**          | `model = YOLO("runs/.../weights/best.pt")`                  | Carrega o checkpoint com o maior Mask mAP obtido na validação.           |
| **Inferência básica com visualização** | `results = model.predict("foto.jpg", conf=0.35, save=True)` | Gera imagem com contornos translúcidos salvos em `runs/segment/predict`. |
| **Polígonos em pixels absolutos**      | `poligonos = result.masks.xy`                               | Retorna lista de arrays NumPy com coordenadas inteiras $[x, y]$ da foto. |
| **Polígonos normalizados $[0, 1]$**    | `poligonos_norm = result.masks.xyn`                         | Coordenadas float relativas ideais para reexportação ou banco de dados.  |
| **Tensor binário de máscaras**         | `mascaras = result.masks.data.cpu().numpy()`                | Matriz 3D $(N, H, W)$ com valores $0.0$ ou $1.0$ de cada objeto.         |
| **Stream contínuo sem estouro de RAM** | `for res in model("video.mp4", stream=True):`               | Gerador Python $O(1)$ RAM: processa horas de vídeo sem consumir memória. |

---

## 4. Operações de Visão Computacional com Máscaras (OpenCV & NumPy)

| Cenário / Objetivo                       | Snippet de Código                                                                   | O que resolve                                                                |
| :--------------------------------------- | :---------------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **Calcular área do objeto em pixels**    | `area_px = int(np.count_nonzero(mascara > 0.5))`                                    | Mede a área exata da silhueta sem contar pixels de fundo.                    |
| **Porcentagem de cobertura da cena**     | `porcentagem = (area_px / (H * W)) * 100`                                           | Quantifica cobertura vegetal, avanço de manchas ou densidade de ocupação.    |
| **Isolar objeto com fundo preto**        | `cv2.bitwise_and(img, img, mask=mascara_uint8)`                                     | Recorta apenas a região contornada, zerando todo o restante da imagem.       |
| **Gerar PNG transparente (Canal Alfa)**  | `rgba = cv2.merge([b, g, r, mascara_uint8])`                                        | Cria recorte comercial profissional sem depender de softwares manuais.       |
| **Calcular centro de massa (Centróide)** | `M = cv2.moments(mascara); cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])` | Localização geométrica precisa para garras robóticas e esteiras industriais. |
| **Desenhar contorno vetorial fino**      | `cv2.polylines(img, [np.int32(poligono)], True, (0, 255, 0), 2)`                    | Renderiza bordas limpas sem o sombreamento opaco padrão.                     |
| **Overlay Translúcido HUD**              | `cv2.addWeighted(img, 0.7, overlay, 0.3, 0)`                                       | Mescla a cor do objeto com transparência elegante de videogame.              |
| **Cerca Virtual Poligonal**              | `cv2.pointPolygonTest(zona, (cx, cy), False) >= 0`                                 | Retorna True se o centróide do objeto invadiu a zona proibida.               |

---

## 5. Auditoria de Métricas & Diagnósticos

| Métrica / Artefato                    | Onde verificar                    | O que indica                                                                |
| :------------------------------------ | :-------------------------------- | :-------------------------------------------------------------------------- |
| **`metrics/mAP50(M)` alto (> 0.80)**  | `results.png` ou log de validação | As máscaras de pixels estão contornando os objetos com alta fidelidade.     |
| **`metrics/mAP50-95(M)` em evolução** | `results.png`                     | O modelo está refinando as micro-curvaturas e detalhes de borda fina.       |
| **Queda contínua de `seg_loss`**      | `results.png`                     | A perda da ProtoNet está convergindo sem divergência matemática.            |
| **Inspeção Visual Imediata**          | `val_batch0_pred.jpg`             | Permite enxergar com os próprios olhos onde o modelo acerta ou vaza pixels. |
| **Matriz de Confusão**                | `confusion_matrix.png`            | Checa se há ambiguidade entre classes irmãs ou detecções espúrias no fundo. |

---

## 6. Exportação & Deploy em Produção

| Cenário / Objetivo                   | Snippet de Código                                                | O que resolve                                                                 |
| :----------------------------------- | :--------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **Exportar para ONNX Universal**     | `model.export(format="onnx", dynamic=True, simplify=True)`       | Gera arquivo estático para execução em C++, C#, Python ou Web.                |
| **Exportar em Meia Precisão (FP16)** | `model.export(format="onnx", half=True)`                         | Reduz o arquivo para metade do tamanho (~12 MB) e acelera em GPUs modernas.   |
| **Inferência ultraleve com ONNX**    | `onnx_model = YOLO("best.onnx"); onnx_model.predict("foto.jpg")` | Executa inferência otimizada mantendo a mesma sintaxe de máscaras do PyTorch. |
