"""
exemplo_opencv_8020.py
Laboratório prático dos 10 mandamentos do OpenCV para YOLO & Segmentação:
1. Leitura de imagem e canais BGR
2. Conversão de polígono do YOLO em máscara binária (cv2.fillPoly)
3. Desenho de contornos vetoriais finos (cv2.polylines)
4. Recorte cirúrgico sem fundo (cv2.bitwise_and)
5. Overlay translúcido estilo HUD de videogame (cv2.addWeighted)
6. Exportação para PNG com Canal Alfa (cv2.split e cv2.merge)
7. Centro de massa geométrico (cv2.moments)
8. Cerca Virtual de segurança (cv2.pointPolygonTest)
"""

from pathlib import Path
import cv2
import numpy as np


def executar_laboratorio_opencv():
    print("=" * 70)
    print("LABORATÓRIO PRÁTICO: O NÚCLEO 80/20 DO OPENCV PARA SEGMENTAÇÃO")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics não instalado.")
        return

    # 1. Carrega modelo leve e imagem de teste
    modelo = YOLO("yolov8n-seg.pt")
    
    # Busca imagem em input/ ou usa a padrão
    img_path = Path("input/bus.jpg")
    if not img_path.exists():
        img_path = "https://ultralytics.com/images/bus.jpg"
    
    print(f"\n1. Executando YOLO na imagem: {img_path}...")
    resultado = modelo.predict(source=str(img_path), conf=0.35, device=0, verbose=False)[0]

    img = resultado.orig_img  # Matriz NumPy BGR original
    h, w = img.shape[:2]
    print(f"✓ Imagem carregada: {w}x{h} pixels, 3 canais BGR.")

    if resultado.masks is None or len(resultado.masks.xy) == 0:
        print("Nenhuma máscara encontrada para demonstrar.")
        return

    # Pega o primeiro objeto segmentado (ex: o ônibus ou uma pessoa)
    poligono = resultado.masks.xy[0]  # Array numpy (N, 2)
    cls_id = int(resultado.boxes.cls[0].cpu().numpy())
    nome_objeto = resultado.names[cls_id]
    pontos = np.int32([poligono])
    print(f"✓ Objeto selecionado: {nome_objeto.upper()} com {len(poligono)} vértices.")

    pasta_saida = Path("runs/segment/exemplos_opencv")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # 2. Criando Máscara Binária com cv2.fillPoly
    print("\n2. Criando máscara binária com cv2.fillPoly...")
    mascara_binaria = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mascara_binaria, pontos, 255)
    cv2.imwrite(str(pasta_saida / "01_mascara_binaria.png"), mascara_binaria)
    print(f"  -> Salvo: {pasta_saida / '01_mascara_binaria.png'}")

    # 3. Traçando Contorno Fino com cv2.polylines
    print("\n3. Desenhando contorno vetorial com cv2.polylines...")
    img_contorno = img.copy()
    cv2.polylines(img_contorno, pontos, isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(str(pasta_saida / "02_contorno_vetorial.jpg"), img_contorno)
    print(f"  -> Salvo: {pasta_saida / '02_contorno_vetorial.jpg'}")

    # 4. Recorte Cirúrgico com cv2.bitwise_and
    print("\n4. Isolando o objeto (apagando o fundo) com cv2.bitwise_and...")
    objeto_isolado = cv2.bitwise_and(img, img, mask=mascara_binaria)
    cv2.imwrite(str(pasta_saida / "03_fundo_preto.jpg"), objeto_isolado)
    print(f"  -> Salvo: {pasta_saida / '03_fundo_preto.jpg'}")

    # 5. Overlay Translúcido com cv2.addWeighted
    print("\n5. Criando efeito translúcido (HUD) com cv2.addWeighted...")
    overlay = img.copy()
    cv2.fillPoly(overlay, pontos, color=(255, 120, 0))  # Azul luminoso em BGR
    img_hud = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(pasta_saida / "04_overlay_translucido.jpg"), img_hud)
    print(f"  -> Salvo: {pasta_saida / '04_overlay_translucido.jpg'}")

    # 6. PNG Transparente com Canal Alfa (cv2.split e cv2.merge)
    print("\n6. Gerando PNG transparente com canal Alfa (BGRA)...")
    b, g, r = cv2.split(img)
    bgra = cv2.merge([b, g, r, mascara_binaria])
    cv2.imwrite(str(pasta_saida / "05_recorte_transparente.png"), bgra)
    print(f"  -> Salvo: {pasta_saida / '05_recorte_transparente.png'}")

    # 7. Centro de Massa (Centróide) com cv2.moments
    print("\n7. Calculando centro de massa com cv2.moments...")
    M = cv2.moments(mascara_binaria)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"  -> Baricentro exato: X={cx}, Y={cy}")
        img_centroide = img.copy()
        cv2.circle(img_centroide, (cx, cy), 8, (0, 0, 255), -1)  # Ponto vermelho
        cv2.putText(img_centroide, f"Centro ({cx},{cy})", (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(str(pasta_saida / "06_centro_de_massa.jpg"), img_centroide)

    # 8. Cerca Virtual com cv2.pointPolygonTest
    print("\n8. Testando Cerca Virtual com cv2.pointPolygonTest...")
    # Cria uma zona arbitrária na metade inferior da tela
    zona_perigo = np.array([[50, 400], [w - 50, 400], [w - 50, h - 50], [50, h - 50]], dtype=np.int32)
    invadiu = cv2.pointPolygonTest(zona_perigo, (cx, cy), False) >= 0
    print(f"  -> O centroide do objeto invadiu a zona de perigo? {'🚨 SIM!' if invadiu else '🟢 NÃO'}")

    print("\n" + "=" * 70)
    print(f"✓ Laboratório concluído com sucesso! Veja as saídas em: {pasta_saida}/")
    print("=" * 70)


if __name__ == "__main__":
    executar_laboratorio_opencv()
