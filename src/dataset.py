"""
dataset.py — PREGUNTA 1: carga, visualización, transformaciones y partición.

Contenido:
    * Descarga / carga del dataset oficial CIFAR-10 con torchvision.datasets.
    * Pipeline de transformaciones (ToTensor, Normalize y Data Augmentation).
    * Partición ESTRATIFICADA entrenamiento/validación con SEMILLA = 65.
    * DataLoaders reproducibles (generador con semilla 65).
    * Cuadrícula de imágenes de ejemplo con su clase real.

Uso desde la raíz del proyecto:
    python -m src.dataset            # resumen de la partición + cuadrícula de ejemplo
    python -m src.dataset --stats    # recalcula la media/desviación de los canales RGB
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src import config

# ---------------------------------------------------------------------------
# 1. Pipeline de transformaciones
# ---------------------------------------------------------------------------


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Devuelve (transformación de entrenamiento, transformación de evaluación).

    Entrenamiento (con aumento de datos, aleatorio en cada época):
        RandomCrop(32, padding=4) -> RandomHorizontalFlip(p=0.5) -> ToTensor -> Normalize
    Validación / prueba (determinista, sin aumento):
        ToTensor -> Normalize
    """
    normalize = transforms.Normalize(mean=config.CIFAR10_MEAN, std=config.CIFAR10_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # desplazamientos de hasta 4 píxeles
        transforms.RandomHorizontalFlip(p=0.5),  # espejo horizontal
        transforms.ToTensor(),                   # PIL (H,W,C) uint8 -> tensor (C,H,W) float en [0,1]
        normalize,                               # (x - media) / desviación, por canal RGB
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# 2. Carga del dataset oficial
# ---------------------------------------------------------------------------


def load_cifar10(root: Path | str = config.DATA_DIR, download: bool = True):
    """Carga CIFAR-10 oficial (torchvision). Solo descarga si no existe en `root`.

    Se cargan sin transformación (imágenes PIL); las transformaciones se aplican
    después según el subconjunto, para no mezclar el aumento de datos con la
    validación ni con la prueba.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # torchvision verifica la integridad (MD5) y omite la descarga si ya existe.
    train_full = datasets.CIFAR10(root=str(root), train=True, download=download, transform=None)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=download, transform=None)
    return train_full, test_set


class TransformedSubset(Dataset):
    """Subconjunto de un dataset al que se le aplica una transformación propia.

    Permite que entrenamiento y validación compartan las mismas imágenes en memoria
    pero con pipelines distintos (con y sin aumento de datos).
    """

    def __init__(self, base, indices: Sequence[int], transform=None):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        targets = np.asarray(base.targets)
        self.targets = targets[self.indices].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.base[int(self.indices[i])]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# 3. Partición estratificada
# ---------------------------------------------------------------------------


def stratified_split(
    targets: Sequence[int],
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEMILLA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Índices (train, val) estratificados por clase. Reproducible con la semilla 65."""
    targets = np.asarray(targets)
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_fraction,
        stratify=targets,
        random_state=seed,
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(val_idx)


def stratified_subsample(targets: Sequence[int], n: int, seed: int = config.SEMILLA) -> np.ndarray:
    """Submuestra estratificada de n índices (solo para pruebas rápidas)."""
    targets = np.asarray(targets)
    if n >= len(targets):
        return np.arange(len(targets))
    idx, _ = train_test_split(
        np.arange(len(targets)), train_size=n, stratify=targets, random_state=seed
    )
    return np.sort(idx)


def class_distribution(targets: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(t) for t in targets)
    return {config.CLASSES[c]: counts.get(c, 0) for c in range(config.NUM_CLASSES)}


# ---------------------------------------------------------------------------
# 4. DataLoaders
# ---------------------------------------------------------------------------


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    val_fraction: float = config.VAL_FRACTION,
    num_workers: int = config.NUM_WORKERS,
    data_dir: Path | str = config.DATA_DIR,
    download: bool = True,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = config.SEMILLA,
    verbose: bool = True,
):
    """Construye los DataLoaders de entrenamiento, validación y prueba.

    * El conjunto de prueba oficial (10 000 imágenes) se mantiene separado.
    * `limit_train` / `limit_test` permiten pruebas rápidas con pocas imágenes
      (submuestras estratificadas); en el entrenamiento definitivo son None.
    """
    train_full, test_set = load_cifar10(data_dir, download=download)
    train_tf, eval_tf = get_transforms()

    base_targets = np.asarray(train_full.targets)
    pool = np.arange(len(base_targets))
    if limit_train is not None:
        pool = stratified_subsample(base_targets, limit_train, seed)

    tr_rel, va_rel = stratified_split(base_targets[pool], val_fraction, seed)
    train_idx, val_idx = pool[tr_rel], pool[va_rel]

    test_idx = np.arange(len(test_set))
    if limit_test is not None:
        test_idx = stratified_subsample(test_set.targets, limit_test, seed)

    train_ds = TransformedSubset(train_full, train_idx, train_tf)  # con aumento
    val_ds = TransformedSubset(train_full, val_idx, eval_tf)       # sin aumento
    test_ds = TransformedSubset(test_set, test_idx, eval_tf)       # sin aumento

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin, worker_init_fn=config.seed_worker)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=config.make_generator(seed), **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    info = {
        "semilla": seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "batch_size": batch_size,
        "dist_train": class_distribution(train_ds.targets),
        "dist_val": class_distribution(val_ds.targets),
        "dist_test": class_distribution(test_ds.targets),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    if verbose:
        print_split_summary(info, train_loader)
    return train_loader, val_loader, test_loader, info


def print_split_summary(info: dict, train_loader: Optional[DataLoader] = None) -> None:
    """Imprime en consola tamaños, distribución por clase y dimensiones de los tensores."""
    print("=" * 74)
    print(f"PARTICIÓN ESTRATIFICADA DE CIFAR-10   (SEMILLA = {info['semilla']})")
    print("=" * 74)
    print(f"Imágenes de entrenamiento : {info['n_train']:>6}")
    print(f"Imágenes de validación    : {info['n_val']:>6}")
    print(f"Imágenes de prueba        : {info['n_test']:>6}  (conjunto oficial, separado)")
    print(f"Imágenes por lote (batch) : {info['batch_size']:>6}")
    print("-" * 74)
    print(f"{'Clase':<12}{'Train':>8}{'%':>8}{'Val':>8}{'%':>8}{'Test':>8}{'%':>8}")
    for c in config.CLASSES:
        tr, va, te = info["dist_train"][c], info["dist_val"][c], info["dist_test"][c]
        print(f"{c:<12}{tr:>8}{100*tr/max(info['n_train'],1):>7.1f}%"
              f"{va:>8}{100*va/max(info['n_val'],1):>7.1f}%"
              f"{te:>8}{100*te/max(info['n_test'],1):>7.1f}%")
    if train_loader is not None:
        images, labels = next(iter(train_loader))
        print("-" * 74)
        print(f"Tensor de imágenes de un lote : {tuple(images.shape)}  -> (B, C, H, W)")
        print(f"Tensor de etiquetas           : {tuple(labels.shape)}  dtype={labels.dtype}")
        print(f"Tipo de dato de las imágenes  : {images.dtype}")
        print(f"Lotes por época (train)       : {len(train_loader)}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# 5. Visualización
# ---------------------------------------------------------------------------


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Deshace Normalize para poder visualizar: x = x_norm * std + media."""
    mean = torch.tensor(config.CIFAR10_MEAN, dtype=t.dtype).view(-1, 1, 1)
    std = torch.tensor(config.CIFAR10_STD, dtype=t.dtype).view(-1, 1, 1)
    return (t * std + mean).clamp(0.0, 1.0)


def show_samples(dataset, n: int = 16, seed: int = config.SEMILLA, save_path=None,
                 title: str | None = None, show: bool = False):
    """Cuadrícula de n (8–16) imágenes con el nombre de su clase real.

    Acepta tanto el dataset sin transformar (imágenes PIL) como uno normalizado
    (tensores), en cuyo caso deshace la normalización antes de dibujar.
    """
    import matplotlib.pyplot as plt

    if not 8 <= n <= 16:
        raise ValueError("n debe estar entre 8 y 16 imágenes")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    cols = 8 if n > 8 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 2.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idxs):
        img, label = dataset[int(i)]
        if isinstance(img, torch.Tensor):
            arr = denormalize(img).permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
        else:
            arr = np.asarray(img)
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(config.CLASSES[label], fontsize=9)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.suptitle(title or f"CIFAR-10: {n} imágenes de ejemplo (SEMILLA = {seed})", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def compute_mean_std(train_dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula media y desviación por canal sobre el dataset de entrenamiento (sin transformar)."""
    data = np.asarray(train_dataset.data, dtype=np.float64) / 255.0  # (N, 32, 32, 3)
    return data.mean(axis=(0, 1, 2)), data.std(axis=(0, 1, 2))


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pregunta 1: dataset CIFAR-10 (SEMILLA = 65)")
    parser.add_argument("--stats", action="store_true", help="recalcula media/desviación RGB")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.set_seed(config.SEMILLA)
    print(f"SEMILLA = {config.SEMILLA}")
    if args.stats:
        train_full, _ = load_cifar10()
        m, s = compute_mean_std(train_full)
        print("Media por canal (R,G,B):", np.round(m, 4))
        print("Desv. por canal (R,G,B):", np.round(s, 4))
    else:
        train_loader, _, _, _ = get_dataloaders(batch_size=args.batch_size)
        train_full, _ = load_cifar10()
        out = config.OUTPUTS_DIR / "samples_grid.png"
        show_samples(train_full, n=16, save_path=out)
        print(f"Cuadrícula de ejemplo guardada en: {out.relative_to(config.ROOT_DIR)}")
