"""
exemplo_inferencia_seg.py
Demonstração prática dos conceitos de Inferência e Manipulação de Máscaras com YOLO:
1. Carregamento de modelo de segmentação (yolov8n-seg.pt)
2. Inferência com controle de limiares de confiança e NMS
3. Extração programática de Bounding Boxes e Polígonos de Máscaras (result.masks.xy)
4. Cálculo de área em pixels e porcentagem de ocupação
5. Isolamento do objeto com fundo preto e exportação ONNX
"""

from pathlib import Path
import cv2
import numpy as np

def demonstrar_inferencia_segmentacao():
    print("=" * 70)
    print("INFERÊNCIA DE SEGMENTAÇÃO & EXTRAÇÃO DE MÁSCARAS POLIGONAIS")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ A biblioteca 'ultralytics' não está instalada.")
        print("Instale com: pip install ultralytics")
        return

    # Usando o modelo pré-treinado nano de segmentação (80 classes do COCO)
    modelo = YOLO("yolov8n-seg.pt")
    print("✓ Modelo 'yolov8n-seg.pt' carregado com sucesso.")

    # Busca imagem na pasta input/ ou utiliza imagem pública de teste
    imagens_locais = list(Path("input").glob("*.*"))
    if imagens_locais:
        imagem_teste = str(imagens_locais[0])
    else:
        imagem_teste = "https://ultralytics.com/images/bus.jpg"

    print(f"\n✓ Executando inferência sobre a imagem: {imagem_teste}...")

    resultados = modelo.predict(
        source=imagem_teste,
        conf=0.35,      # Descarta detecções com confiança inferior a 35%
        iou=0.60,       # Suprime sobreposições espaciais redundantes
        save=False,
        device=0,       # Aceleração nativa na GPU
        verbose=False
    )

    resultado = resultados[0]
    total_objetos = len(resultado.boxes)
    print(f"✓ Total de instâncias detectadas e segmentadas: {total_objetos}")

    if total_objetos == 0 or resultado.masks is None:
        print("Nenhuma máscara encontrada com os limiares atuais.")
        return

    h_img, w_img = resultado.orig_shape
    total_pixels_imagem = h_img * w_img

    print("\n--- DETALHAMENTO DAS INSTÂNCIAS ---")
    for i, (box, poligono) in enumerate(zip(resultado.boxes, resultado.masks.xy)):
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        nome_classe = resultado.names[cls_id]
        caixa = box.xyxy[0].cpu().numpy().astype(int).tolist()
        num_vertices = len(poligono)

        mascara_instancia = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mascara_instancia, [np.int32(poligono)], 255)
        area_pixels = int(np.count_nonzero(mascara_instancia))
        porcentagem = (area_pixels / total_pixels_imagem) * 100

        print(f"[{i+1}] {nome_classe.upper():<12} | Conf: {conf:>6.1%} | Vértices: {num_vertices:>3} pts | "
              f"Área: {area_pixels:>7,} px ({porcentagem:>5.2f}%) | Caixa: {caixa}")

    # Salva uma amostra recortada do primeiro objeto
    if len(resultado.masks.xy) > 0 and resultado.orig_img is not None:
        img_original = resultado.orig_img
        mascara_primeiro = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mascara_primeiro, [np.int32(resultado.masks.xy[0])], 255)
        isolado = cv2.bitwise_and(img_original, img_original, mask=mascara_primeiro)

        pasta_saida = Path("runs/segment/exemplos")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_saida = pasta_saida / "primeiro_objeto_isolado.jpg"
        cv2.imwrite(str(caminho_saida), isolado)
        print(f"\n✓ Imagem com o 1º objeto isolado (fundo preto) salva em: {caminho_saida}")


if __name__ == "__main__":
    demonstrar_inferencia_segmentacao()
