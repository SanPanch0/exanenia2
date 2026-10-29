"""
run.py — Punto de entrada del proyecto (ejecutar desde la raíz).

    python run.py                 # verifica entorno + shapes y lanza el entrenamiento definitivo
    python run.py --check         # solo verificación de entorno y del flujo de tensores 4D
    python run.py --tests         # ejecuta las pruebas automatizadas (pytest)
    python run.py --quick         # prueba rápida (1 época, pocas imágenes) -> outputs/quick_test/
    python run.py --epochs 20 --batch-size 64 --lr 0.0005   # hiperparámetros personalizados

Cualquier argumento no reconocido aquí se reenvía a src/train.py.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_environment() -> None:
    import matplotlib
    import numpy
    import sklearn
    import torch
    import torchvision

    from src import config
    from src.model import verify_shapes

    print("=" * 60)
    print("VERIFICACIÓN DEL ENTORNO")
    print("=" * 60)
    print(f"Python        : {sys.version.split()[0]} ({platform.system()})")
    print(f"PyTorch       : {torch.__version__}")
    print(f"torchvision   : {torchvision.__version__}")
    print(f"NumPy         : {numpy.__version__}")
    print(f"scikit-learn  : {sklearn.__version__}")
    print(f"matplotlib    : {matplotlib.__version__}")
    print(f"CUDA          : {'sí - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (se usará CPU)'}")
    print(f"SEMILLA       : {config.SEMILLA}")
    print(f"Carpeta datos : {config.DATA_DIR.relative_to(config.ROOT_DIR)}/")
    verify_shapes(batch_size=8, device="cpu")
    if torch.cuda.is_available():
        verify_shapes(batch_size=8, device="cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN CIFAR-10 — SEMILLA = 65", add_help=True)
    parser.add_argument("--check", action="store_true", help="solo verificar entorno y shapes")
    parser.add_argument("--tests", action="store_true", help="ejecutar pytest")
    args, train_args = parser.parse_known_args()

    if args.tests:
        return subprocess.call([sys.executable, "-m", "pytest", "-v", str(ROOT / "tests")], cwd=ROOT)

    check_environment()
    if args.check:
        return 0

    from src import train
    train.main(train_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
