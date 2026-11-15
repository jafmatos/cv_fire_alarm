"""
exemplo_treino_seg.py
Demonstração prática dos conceitos de Treinamento e Fine-Tuning para Segmentação com Ultralytics YOLO:
1. Verificação de hardware (CUDA/ROCm GPU vs CPU)
2. Carregamento de arquitetura pré-treinada para segmentação (YOLOv8 Nano Segmentation)
3. Configuração de hiperparâmetros essenciais (Regra 80/20)
4. Diagnóstico de métricas (Mask mAP vs Box mAP)
"""

import torch
from pathlib import Path

def executar_treinamento_segmentacao():
    print("=" * 70)
    print("TREINAMENTO DE SEGMENTAÇÃO DE INSTÂNCIAS COM YOLOV8 NANO (80/20)")
    print("=" * 70)

    # 1. Checagem de Hardware
    tem_gpu = torch.cuda.is_available()
    device = 0 if tem_gpu else "cpu"
    print(f"✓ Dispositivo de execução detectado: {device} ({'GPU AMD/CUDA' if tem_gpu else 'CPU'})")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ A biblioteca 'ultralytics' não está instalada.")
        print("Instale com: pip install ultralytics")
        return

    # 2. Carrega modelo pré-treinado no COCO (segmentação de 80 classes)
    print("\n✓ Carregando pesos pré-treinados 'yolov8n-seg.pt'...")
    model = YOLO("yolov8n-seg.pt")

    # 3. Exemplo de chamada de treino (apontando para data.yaml local ou demonstrativo)
    data_yaml = Path("data.yaml")
    if not data_yaml.exists():
        print(f"\n[Aviso] Arquivo '{data_yaml}' não encontrado no diretório atual.")
        print("Para treinar com seus próprios dados, crie o arquivo data.yaml conforme o Guia 01.")
        print("\nDemonstração da sintaxe padrão recomendada 80/20 para Segmentação:")
        print("""
        results = model.train(
            data="data.yaml",             # Caminho para configuração do dataset
            epochs=50,                    # Suficiente para convergência inicial de máscaras
            imgsz=640,                    # Resolução espacial balanceada
            batch=16 if tem_gpu else 4,   # Tamanho do lote
            patience=15,                  # Early Stopping após 15 épocas sem ganho no Mask mAP
            device=device,                # 0 para GPU ou 'cpu'
            workers=4,                    # Paralelismo no carregamento de dados
            project="runs/segment",       # Pasta raiz dos experimentos
            name="experimento_seg_v1"     # Nome da pasta do experimento atual
        )
        """)
        return

    print("\nIniciando treinamento de segmentação...")
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16 if tem_gpu else 4,
        patience=15,
        device=device,
        workers=4 if tem_gpu else 2,
        project="runs/segment",
        name="experimento_seg_v1"
    )

    print("\n✓ Treino finalizado com sucesso!")
    print(f"✓ Melhores pesos salvos em: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    executar_treinamento_segmentacao()
