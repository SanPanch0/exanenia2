"""
model.py — PREGUNTA 3: arquitectura CNN, flujo de tensores 4D y validación de shapes.

Entrada:  tensor 4D (B, C, H, W) = (B, 3, 32, 32)
Salida:   tensor 2D (B, 10)  -> logits (sin Softmax, porque CrossEntropyLoss
                                ya aplica LogSoftmax + NLLLoss internamente)

Arquitectura (3 bloques convolucionales + clasificador denso):

    Bloque 1: Conv2d(3→32, 3x3, padding=1)  -> BatchNorm2d -> ReLU -> MaxPool2d(2)   (B, 32, 16, 16)
    Bloque 2: Conv2d(32→64, 3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)   (B, 64,  8,  8)
    Bloque 3: Conv2d(64→128,3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)   (B,128,  4,  4)
    Flatten                                                                         (B, 2048)
    Linear(2048→256) -> ReLU -> Dropout(0.5)                                        (B, 256)
    Linear(256→10)                                                                  (B, 10)

Uso desde la raíz del proyecto:
    python -m src.model      # imprime los shapes intermedios y ejecuta las aserciones
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from src import config


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloque convolucional: Conv2d(3x3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2).

    padding=1 conserva H y W en la convolución; MaxPool2d(2) los reduce a la mitad.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class CNNCifar10(nn.Module):
    """CNN de tres bloques convolucionales para CIFAR-10 (10 clases)."""

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.block1 = conv_block(3, 32)     # (B,  3,32,32) -> (B, 32,16,16)
        self.block2 = conv_block(32, 64)    # (B, 32,16,16) -> (B, 64, 8, 8)
        self.block3 = conv_block(64, 128)   # (B, 64, 8, 8) -> (B,128, 4, 4)
        self.flatten = nn.Flatten()         # (B,128, 4, 4) -> (B, 2048)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)  # capa de salida: 10 logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Se esperaba un tensor 4D (B, C, H, W) y se recibió {tuple(x.shape)}")
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)  # logits, SIN Softmax

    @torch.no_grad()
    def forward_with_shapes(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        """Recorre la red capa por capa y registra la forma del tensor tras cada una."""
        shapes = [("Entrada", tuple(x.shape))]
        for bname in ("block1", "block2", "block3"):
            block = getattr(self, bname)
            for layer in block:
                x = layer(x)
                shapes.append((f"{bname}.{layer.__class__.__name__}", tuple(x.shape)))
        for name in ("flatten", "fc1", "relu", "dropout", "fc2"):
            x = getattr(self, name)(x)
            shapes.append((name + f" ({getattr(self, name).__class__.__name__})", tuple(x.shape)))
        return shapes


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_per_layer(model: nn.Module) -> List[Tuple[str, int]]:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            rows.append((name, sum(p.numel() for p in module.parameters())))
    return rows


def verify_shapes(batch_size: int = 8, device: torch.device | str | None = None,
                  verbose: bool = True) -> List[Tuple[str, Tuple[int, ...]]]:
    """Pasa un lote simulado por el modelo, imprime los shapes y verifica con aserciones."""
    config.set_seed(config.SEMILLA)
    device = torch.device(device) if device is not None else config.get_device()
    model = CNNCifar10().to(device).eval()
    x = torch.randn(batch_size, config.IMG_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH, device=device)
    shapes = model.forward_with_shapes(x)

    if verbose:
        print("=" * 60)
        print(f"VALIDACIÓN DEL FLUJO DE TENSORES 4D  (dispositivo: {device}, SEMILLA = {config.SEMILLA})")
        print("=" * 60)
        for name, shp in shapes:
            print(f"{name:<28} -> {shp}")
        print("-" * 60)
        print(f"Parámetros entrenables: {count_parameters(model):,}")

    d = dict(shapes)
    B = batch_size
    assert d["Entrada"] == (B, 3, 32, 32), "La entrada debe ser (B, 3, 32, 32)"
    assert d["block1.MaxPool2d"] == (B, 32, 16, 16), "Shape incorrecto tras el bloque 1"
    assert d["block2.MaxPool2d"] == (B, 64, 8, 8), "Shape incorrecto tras el bloque 2"
    assert d["block3.MaxPool2d"] == (B, 128, 4, 4), "Shape incorrecto tras el bloque 3"
    assert d["flatten (Flatten)"] == (B, 2048), "Flatten debe producir (B, 2048)"
    assert d["fc1 (Linear)"] == (B, 256), "fc1 debe producir (B, 256)"
    out = model(x)
    assert out.shape == (B, config.NUM_CLASSES), f"La salida debe ser (B, 10); se obtuvo {tuple(out.shape)}"
    assert d["fc2 (Linear)"] == tuple(out.shape)
    assert torch.isfinite(out).all(), "La salida contiene NaN o infinitos"
    if verbose:
        print(f"Aserciones superadas: entrada {d['Entrada']} -> salida {tuple(out.shape)}")
        print("=" * 60)
    return shapes


if __name__ == "__main__":
    verify_shapes(batch_size=8, device="cpu")
    if torch.cuda.is_available():
        verify_shapes(batch_size=8, device="cuda")
    else:
        print("CUDA no está disponible: la verificación se ejecutó solo en CPU.")
    print("\nParámetros por capa:")
    for name, n in parameters_per_layer(CNNCifar10()):
        print(f"  {name:<12} {n:>9,}")
