"""Pruebas del dataset (Pregunta 1). Ejecutar desde la raíz: python -m pytest -v

Las pruebas marcadas con `requiere_cifar` se omiten si CIFAR-10 aún no se ha
descargado en data/ (ejecute antes: python -m src.dataset).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dataset import (denormalize, get_transforms, stratified_split,  # noqa: E402
                         stratified_subsample)


def _cifar_disponible() -> bool:
    try:
        from torchvision import datasets
        datasets.CIFAR10(root=str(config.DATA_DIR), train=False, download=False)
        return True
    except Exception:
        return False


requiere_cifar = pytest.mark.skipif(not _cifar_disponible(),
                                    reason="CIFAR-10 no está descargado en data/")


def test_semilla_centralizada():
    assert config.SEMILLA == 65


def test_set_seed_reproduce_numeros():
    config.set_seed(65)
    a = (np.random.rand(3), torch.rand(3))
    config.set_seed(65)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.equal(a[1], b[1])


def test_pipeline_de_transformaciones():
    train_tf, eval_tf = get_transforms()
    tipos_train = [type(t) for t in train_tf.transforms]
    tipos_eval = [type(t) for t in eval_tf.transforms]
    for t in (transforms.ToTensor, transforms.Normalize, transforms.RandomCrop,
              transforms.RandomHorizontalFlip):
        assert t in tipos_train
    # La evaluación NO debe contener transformaciones aleatorias
    assert transforms.RandomCrop not in tipos_eval
    assert transforms.RandomHorizontalFlip not in tipos_eval
    assert tipos_eval == [transforms.ToTensor, transforms.Normalize]


def test_transformacion_produce_tensor_3x32x32():
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    train_tf, eval_tf = get_transforms()
    for tf in (train_tf, eval_tf):
        t = tf(img)
        assert t.shape == (3, 32, 32) and t.dtype == torch.float32


def test_denormalize_invierte_normalize():
    x = torch.rand(3, 32, 32)
    norm = transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD)(x)
    assert torch.allclose(denormalize(norm), x, atol=1e-6)


def test_particion_estratificada_sintetica():
    targets = np.repeat(np.arange(10), 5000)  # misma estructura que el train de CIFAR-10
    tr, va = stratified_split(targets, 0.10, 65)
    assert len(tr) == 45000 and len(va) == 5000
    assert len(np.intersect1d(tr, va)) == 0
    assert np.all(np.bincount(targets[tr]) == 4500)
    assert np.all(np.bincount(targets[va]) == 500)


def test_particion_reproducible_y_depende_de_semilla():
    targets = np.repeat(np.arange(10), 100)
    a, _ = stratified_split(targets, 0.2, 65)
    b, _ = stratified_split(targets, 0.2, 65)
    c, _ = stratified_split(targets, 0.2, 14)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_submuestra_estratificada():
    targets = np.repeat(np.arange(10), 100)
    idx = stratified_subsample(targets, 200, 65)
    assert len(idx) == 200 and np.all(np.bincount(targets[idx]) == 20)


@requiere_cifar
def test_dataloaders_reales():
    from src.dataset import get_dataloaders
    tl, vl, sl, info = get_dataloaders(batch_size=16, download=False, limit_train=1000,
                                       limit_test=200, verbose=False)
    x, y = next(iter(tl))
    assert x.shape == (16, 3, 32, 32) and y.shape == (16,)
    assert info["n_train"] == 900 and info["n_val"] == 100 and info["n_test"] == 200
    assert set(info["dist_val"].values()) == {10}
    assert len(np.intersect1d(info["train_idx"], info["val_idx"])) == 0


@requiere_cifar
def test_dataset_oficial_completo():
    from src.dataset import load_cifar10
    train_full, test_set = load_cifar10(download=False)
    assert len(train_full) == 50000 and len(test_set) == 10000
    assert train_full.data.shape == (50000, 32, 32, 3)
    assert tuple(train_full.classes) == config.CLASSES
    assert np.all(np.bincount(train_full.targets) == 5000)
