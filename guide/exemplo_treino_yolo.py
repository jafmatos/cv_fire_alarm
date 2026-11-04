"""
exemplo_treino_yolo.py
Demonstração prática dos conceitos de Treinamento e Fine-Tuning com Ultralytics YOLO:
1. Verificação de hardware (CUDA GPU vs CPU)
2. Carregamento de arquitetura pré-treinada (YOLOv8 Nano)
3. Configuração de hiperparâmetros essenciais (Regra 80/20)
4. Inspeção de métricas de validação
"""

import torch
from pathlib import Path

def executar_treinamento_exemplo():
    print("=" * 65)
    print("TREINAMENTO DE DETECTOR COM YOLOV8 NANO (TRANSFER LEARNING)")
    print("=" * 65)

    # 1. Checagem de Hardware
    tem_cuda = torch.cuda.is_available()
    device = 0 if tem_cuda else "cpu"
    print(f"✓ Dispositivo de execução detectado: {device} ({'GPU CUDA' if tem_cuda else 'CPU'})")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ A biblioteca 'ultralytics' não está instalada.")
        print("Instale com: pip install ultralytics")
        return

    # 2. Carrega modelo pré-treinado no COCO
    # Pesos compactos (~6 MB), ideal para prototipagem rápida e iteração 80/20
    print("\n✓ Carregando pesos pré-treinados 'yolov8n.pt'...")
    model = YOLO("yolov8n.pt")

    # 3. Exemplo de chamada de treino (apontando para data.yaml hipotético ou local)
    data_yaml = Path("data.yaml")
    if not data_yaml.exists():
        print(f"\n[Aviso] Arquivo '{data_yaml}' não encontrado no diretório atual.")
        print("Para treinar com seus próprios dados, crie o arquivo data.yaml conforme o Guia 01.")
        print("\nDemonstração da sintaxe padrão recomendada 80/20:")
        print("""
        results = model.train(
            data="data.yaml",       # Caminho para configuração do dataset
            epochs=50,              # Quantidade suficiente para convergência inicial
            imgsz=640,              # Resolução padrão balanceada
            batch=16,               # Tamanho do lote (use 8 se tiver restrição de VRAM)
            patience=15,            # Early Stopping após 15 épocas sem ganho de mAP
            device=device,          # 0 para GPU ou 'cpu'
            workers=4,              # Paralelismo de I/O na CPU
            project="experimentos", # Pasta raiz dos resultados
            name="detector_v1"      # Nome da pasta do experimento
        )
        """)
        return

    print("\nIniciando treinamento...")
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16 if tem_cuda else 4,
        patience=15,
        device=device,
        project="experimentos",
        name="detector_v1"
    )

    print("\n✓ Treino finalizado com sucesso!")
    print(f"✓ Pesos salvos em: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    executar_treinamento_exemplo()
