# 🎯 Trilha de Especialização: Detecção de Objetos com YOLO (80/20)

> **Objetivo:** Dominar a detecção de objetos moderna com **YOLO** aplicando a regra 80/20. Em vez de se perder na complexidade matemática das centenas de variações de arquiteturas de redes neurais, esta trilha foca nos **20% de conceitos, padrões de dados e ferramentas práticas que entregam 80% do resultado real** em projetos industriais.

---

## 🗺️ Mapa da Transição: De Classificação para Detecção

| Aspecto | Como fazíamos na Classificação (TorchVision) | O Novo Desafio na Detecção de Objetos | Como o Ecossistema YOLO Resolve |
| :--- | :--- | :--- | :--- |
| **Objetivo do Modelo** | Prever um único rótulo probabilístico para a imagem inteira. | Localizar múltiplos objetos em posições espaciais livres e classificá-los simultaneamente. | **Single-Shot Detector:** Prevê caixas delimitadoras (*Bounding Boxes*) e probabilidades em uma única passada de inferência. |
| **Anotação de Dados** | Organização simples em pastas por nome de classe (`ImageFolder`). | Cada imagem necessita de coordenadas precisas $[x, y, w, h]$ para cada instância presente. | **Formato YOLO TXT:** Arquivos `.txt` individuais contendo coordenadas normalizadas no intervalo $[0, 1]$. |
| **Pipeline de Ingestão** | `DataLoader` padrão com `transforms` aplicados apenas ao tensor da imagem. | Transformações geométricas (rotação, corte, flip) devem deformar a imagem **e** recalcular as coordenadas das caixas sincronizadamente. | **Augmentation Integrado:** O YOLO aplica *Mosaic*, *MixUp* e distorções afins recalculando os vértices das caixas em tempo real. |
| **Métrica de Sucesso** | Acurácia Top-1 / Top-5, Matriz de Confusão e F1-Score clássico. | Uma previsão só é válida se acertar a classe **e** tiver alta sobreposição espacial com a caixa real. | **mAP (Mean Average Precision):** Avalia precisão e revocação combinadas a limiares de sobreposição espacial (**IoU**). |
| **Deploy e Inferência** | Extrair `logits`, passar por `torch.softmax` e obter a classe vencedora. | Processar centenas de caixas candidatas, suprimir redundâncias (**NMS**) e desenhar retângulos rotulados. | **NMS Nativo:** A biblioteca já entrega tensores limpos com coordenadas desnormalizadas prontas para renderização. |

---

## 📚 Índice da Trilha 80/20

```
┌────────────────────────────────────────────────────────────────────────┐
│  ⭐ ETAPA 1: O Núcleo dos Dados & Métricas                              │
│     • Guia 01: [Fundamentos da Detecção & O Padrão YOLO](01_FUNDAMENTOS_E_FORMATO_YOLO.md)
│       Entendendo Bounding Boxes, normalização [0, 1], data.yaml e mAP  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  🚀 ETAPA 2: Treinamento e Inferência Profissional                     │
│     • Guia 02: [Treinamento, Inferência e Deploy com Ultralytics](02_TREINAMENTO_E_INFERENCIA_ULTRALYTICS.md)
│       Transfer learning, métricas, rastreamento MOT (ByteTrack) e exportação  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  📋 ETAPA 3: O Guia de Consulta Rápida                                 │
│     • Guia 03: [Guia de Bolso: YOLO 80/20](03_GUIA_DE_BOLSO_YOLO_8020.md)
│       Tabelas com snippets diretos para manter ao lado do editor       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Scripts Práticos Prontos para Uso

| Script | O que demonstra no contexto do YOLO |
| :--- | :--- |
| [`exemplo_treino_yolo.py`](exemplo_treino_yolo.py) | Checagem de dispositivo (GPU/CPU), carregamento de rede pré-treinada e configuração 80/20 de treino. |
| [`exemplo_inferencia_e_export.py`](exemplo_inferencia_e_export.py) | Inferência com limiares de confiança e NMS, extração estruturada de caixas e exportação otimizada para ONNX. |

---

## 💡 Como Navegar por Este Material

1. Se você acabou de vir de um problema de classificação e precisa entender **como estruturar suas pastas e arquivos de anotação**, comece pelo [Guia 01](01_FUNDAMENTOS_E_FORMATO_YOLO.md).
2. Se você já tem imagens anotadas e quer rodar o **treino, validar com gráficos de métricas e fazer predições em fotos/vídeos**, siga o [Guia 02](02_TREINAMENTO_E_INFERENCIA_ULTRALYTICS.md).
3. Se estiver no meio do código e só precisar de um **snippet rápido** de inferência, carregamento de pesos ou exportação ONNX, consulte diretamente o [Guia 03 (Guia de Bolso)](03_GUIA_DE_BOLSO_YOLO_8020.md).
