"""
exemplo_inferencia_e_export.py
Demonstração prática dos conceitos de Inferência e Exportação de Modelos YOLO:
1. Carregamento de modelo treinado (ou pré-treinado para teste imediato)
2. Inferência com ajuste fino de limiares (confiança e NMS IoU)
3. Extração e manipulação programática de Bounding Boxes [x1, y1, x2, y2]
4. Exportação do modelo para formato universal ONNX para produção
"""

from pathlib import Path

def demonstrar_inferencia():
    print("=" * 65)
    print("INFERÊNCIA E EXTRAÇÃO PROGRAMÁTICA DE BOUNDING BOXES")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ A biblioteca 'ultralytics' não está instalada.")
        print("Instale com: pip install ultralytics")
        return

    # Usando o modelo nano padrão com 80 classes do COCO para demonstração imediata
    modelo = YOLO("yolov8n.pt")
    print("✓ Modelo carregado com sucesso.")

    # URL ou caminho local de imagem de teste
    imagem_teste = "https://ultralytics.com/images/bus.jpg"
    print(f"\n✓ Executando inferência sobre a imagem de teste: {imagem_teste}...")

    resultados = modelo.predict(
        source=imagem_teste,
        conf=0.35,        # Descarta detecções abaixo de 35% de confiança
        iou=0.60,         # Suprime caixas redundantes sobre o mesmo objeto
        save=False,
        verbose=False
    )

    resultado = resultados[0]
    total_objetos = len(resultado.boxes)
    print(f"✓ Total de objetos detectados: {total_objetos}")

    for i, box in enumerate(resultado.boxes):
        # Coordenadas em pixels na imagem original
        coords = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        nome_classe = resultado.names[cls_id]

        print(f"  [{i+1}] Classe: {nome_classe:<12} | Conf: {conf:>6.1%} | Caixa: {list(coords)}")


def demonstrar_exportacao_onnx():
    print("\n" + "=" * 65)
    print("EXPORTAÇÃO DE MODELO PARA ONNX (PRODUÇÃO & DEPLOY)")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        return

    modelo = YOLO("yolov8n.pt")
    print("✓ Exportando 'yolov8n.pt' para formato ONNX...")
    
    caminho = modelo.export(
        format="onnx",
        dynamic=True,   # Suporta tamanhos de batch dinâmicos
        simplify=True   # Otimiza operadores do grafo computacional
    )
    print(f"✓ Exportação concluída com sucesso! Arquivo gerado em: {caminho}")


if __name__ == "__main__":
    demonstrar_inferencia()
    demonstrar_exportacao_onnx()
