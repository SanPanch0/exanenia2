"""
config.py — Configuración centralizada del experimento.

TODOS los módulos (dataset, model, train, run.py, notebook y pruebas) importan
la semilla y los hiperparámetros desde aquí, de modo que ningún archivo pueda
usar una semilla distinta.

    SEMILLA = 65   (dos últimos dígitos del código de matrícula 021101465)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Datos del estudiante y de la evaluación
# ---------------------------------------------------------------------------
INSTITUCION = "Universidad Andina del Cusco"
ESCUELA = "Ingeniería de Sistemas"
CURSO = "Inteligencia Artificial"
EVALUACION = "Segunda Unidad"
CODIGO_MATRICULA = "021101465"

# ---------------------------------------------------------------------------
# Semilla individual obligatoria
# ---------------------------------------------------------------------------
SEMILLA = 65

# ---------------------------------------------------------------------------
# Rutas (relativas a la raíz del proyecto, nunca absolutas de una máquina)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = len(CLASSES)
IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH = 3, 32, 32

# Media y desviación estándar por canal (R, G, B) del conjunto de ENTRENAMIENTO
# oficial de CIFAR-10 (50 000 imágenes, píxeles escalados a [0, 1]).
# Se pueden recalcular con: python -m src.dataset --stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Fracción del conjunto de entrenamiento oficial que se reserva para validación
VAL_FRACTION = 0.10  # 45 000 entrenamiento / 5 000 validación

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto (modificables desde la consola)
# ---------------------------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
# 0 workers es lo más portable (Windows/Jupyter). Se puede subir con --num-workers.
NUM_WORKERS = 0


def set_seed(seed: int = SEMILLA, deterministic: bool = True) -> None:
    """Fija todas las fuentes de aleatoriedad con la semilla indicada (65)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Necesario para que cuBLAS sea determinista en GPU
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Semilla de cada proceso del DataLoader (derivada de la semilla 65)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEMILLA) -> torch.Generator:
    """Generador de PyTorch con semilla 65 para el barajado (shuffle) del DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device() -> torch.device:
    """CUDA si está disponible; en caso contrario, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
