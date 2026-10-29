"""
train.py — PREGUNTA 4: ciclo de entrenamiento, métricas y curvas de aprendizaje.

* Pérdida: torch.nn.CrossEntropyLoss
* Optimizador: torch.optim.Adam
* SEMILLA = 65 (desde src/config.py)
* Registro por época de loss/accuracy de entrenamiento y validación + tiempo.
* Se conserva el modelo con mejor accuracy de validación y se evalúa UNA vez
  sobre el conjunto de prueba oficial.
* Artefactos en outputs/: loss_accuracy.png, confusion_matrix.png, metrics.json,
  history.csv, classification_report.txt, best_model.pth

Uso desde la raíz del proyecto:
    python -m src.train                       # entrenamiento definitivo (config.py)
    python -m src.train --epochs 20 --batch-size 64 --lr 0.0005
    python -m src.train --quick               # prueba rápida (pocas imágenes, 1 época)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config
from src.dataset import get_dataloaders
from src.model import CNNCifar10, count_parameters

# ---------------------------------------------------------------------------
# Entrenamiento y evaluación de una época
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Una época de entrenamiento. Devuelve (loss promedio, accuracy)."""
    model.train()                       # activa Dropout y BatchNorm en modo entrenamiento
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)                 # (B, 10)
        loss = criterion(logits, labels)       # escalar (media del lote)
        loss.backward()                        # retropropagación de gradientes
        optimizer.step()                       # actualización de pesos (Adam)
        bs = labels.size(0)
        total_loss += loss.item() * bs         # se acumula la suma, no la media
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluación sin gradientes. Devuelve (loss, accuracy, predicciones, etiquetas)."""
    model.eval()                        # desactiva Dropout; BatchNorm usa estadísticas acumuladas
    total_loss, correct, total = 0.0, 0, 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return total_loss / total, correct / total, np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def plot_learning_curves(history: Dict[str, list], save_path: Path, best_epoch: Optional[int] = None):
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(epochs, history["train_loss"], "o-", label="Entrenamiento", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validación", color="#d62728")
    ax1.set_title("Pérdida (CrossEntropyLoss) por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax2.plot(epochs, [100 * a for a in history["train_acc"]], "o-", label="Entrenamiento", color="#1f77b4")
    ax2.plot(epochs, [100 * a for a in history["val_acc"]], "s-", label="Validación", color="#d62728")
    ax2.set_title("Exactitud (accuracy) por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy (%)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xticks(epochs)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Mejor época val. ({best_epoch})")
        ax.legend()
    fig.suptitle(f"Curvas de aprendizaje — CNN CIFAR-10 (SEMILLA = {config.SEMILLA})", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title_extra: str = ""):
    import matplotlib.pyplot as plt

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporción de la clase real (recall por fila)")
    n = len(config.CLASSES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{100 * cm_norm[i, j]:.0f}%", ha="center", va="center",
                    fontsize=7.5, color=color)
    ax.set_title(f"Matriz de confusión — conjunto de prueba (SEMILLA = {config.SEMILLA}){title_extra}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrenamiento de la CNN en CIFAR-10 (SEMILLA = 65)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--limit-train", type=int, default=None,
                   help="usar solo N imágenes del train oficial (pruebas rápidas)")
    p.add_argument("--limit-test", type=int, default=None,
                   help="usar solo N imágenes del test oficial (pruebas rápidas)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="carpeta de salida (por defecto outputs/, o outputs/quick_test con --quick)")
    p.add_argument("--no-download", action="store_true", help="no intentar descargar CIFAR-10")
    p.add_argument("--quick", action="store_true",
                   help="prueba rápida: 1 época, 2000 imágenes de train y 1000 de test")
    p.add_argument("--nota", type=str, default=None, help="nota libre que se guarda en metrics.json")
    args = p.parse_args(argv)
    if args.quick:
        args.epochs = 1
        args.limit_train = args.limit_train or 2000
        args.limit_test = args.limit_test or 1000
    if args.output_dir is None:
        args.output_dir = str(config.OUTPUTS_DIR / ("quick_test" if args.quick else ""))
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config.set_seed(config.SEMILLA)               # SEMILLA = 65 antes de crear datos y modelo
    device = config.get_device()
    print(f"SEMILLA = {config.SEMILLA} | dispositivo: {device} | PyTorch {torch.__version__}")
    tipo = "PRUEBA RÁPIDA" if args.quick else "ENTRENAMIENTO DEFINITIVO"
    print(f"Modo: {tipo} | épocas={args.epochs} | batch={args.batch_size} | lr={args.lr} "
          f"| weight_decay={args.weight_decay}")

    train_loader, val_loader, test_loader, info = get_dataloaders(
        batch_size=args.batch_size, val_fraction=args.val_fraction, num_workers=args.num_workers,
        download=not args.no_download, limit_train=args.limit_train, limit_test=args.limit_test,
    )

    model = CNNCifar10().to(device)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    EPOCHS = args.epochs
    history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_s")}
    best_val_acc, best_epoch, best_state = -1.0, 0, None
    t_start = time.time()

    print(f"\n{'Época':>6} | {'Train loss':>10} {'Train acc':>9} | {'Val loss':>9} {'Val acc':>8} | {'Tiempo':>7}")
    print("-" * 66)
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        for k, v in zip(history, (epoch + 1, tr_loss, tr_acc, va_loss, va_acc, dt)):
            history[k].append(v)
        flag = ""
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            flag = "  *"
        print(f"{epoch + 1:>6} | {tr_loss:>10.4f} {100 * tr_acc:>8.2f}% | {va_loss:>9.4f} "
              f"{100 * va_acc:>7.2f}% | {dt:>6.1f}s{flag}")
    total_time = time.time() - t_start
    print(f"\nEntrenamiento terminado en {total_time / 60:.1f} min. "
          f"Mejor accuracy de validación: {100 * best_val_acc:.2f}% (época {best_epoch})")

    # Se restaura el mejor modelo (según VALIDACIÓN) y se evalúa una sola vez en PRUEBA
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pth")
    te_loss, te_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    labels = list(range(config.NUM_CLASSES))
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                             average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                       average="weighted", zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=config.CLASSES,
                                        output_dict=True, zero_division=0, digits=4)
    report_txt = classification_report(y_true, y_pred, labels=labels, target_names=config.CLASSES,
                                       zero_division=0, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 66)
    print(f"EVALUACIÓN FINAL EN EL CONJUNTO DE PRUEBA ({len(y_true)} imágenes)")
    print("=" * 66)
    print(f"Test loss: {te_loss:.4f} | Test accuracy: {100 * accuracy_score(y_true, y_pred):.2f}%")
    print(f"Precision macro: {p_mac:.4f} | Recall macro: {r_mac:.4f} | F1 macro: {f_mac:.4f}")
    print(report_txt)

    last = -1
    metrics = {
        "proyecto": "Clasificación de CIFAR-10 con CNN",
        "institucion": config.INSTITUCION,
        "curso": config.CURSO,
        "codigo_matricula": config.CODIGO_MATRICULA,
        "SEMILLA": config.SEMILLA,
        "tipo_ejecucion": "prueba_rapida" if args.quick else "entrenamiento_definitivo",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "nota": args.nota,
        "entorno": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "plataforma": platform.platform(),
            "dispositivo": str(device),
        },
        "hiperparametros": {
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
            "weight_decay": args.weight_decay, "optimizador": "Adam",
            "funcion_perdida": "CrossEntropyLoss", "val_fraction": args.val_fraction,
            "data_augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip(p=0.5)"],
            "normalizacion": {"mean": list(config.CIFAR10_MEAN), "std": list(config.CIFAR10_STD)},
        },
        "particion": {"n_train": info["n_train"], "n_val": info["n_val"], "n_test": info["n_test"],
                      "dist_train": info["dist_train"], "dist_val": info["dist_val"],
                      "dist_test": info["dist_test"]},
        "parametros_entrenables": count_parameters(model),
        "tiempo_total_s": round(total_time, 1),
        "mejor_epoca_validacion": best_epoch,
        "mejor_val_accuracy": round(best_val_acc, 4),
        "ultima_epoca": {
            "train_loss": round(history["train_loss"][last], 4),
            "train_accuracy": round(history["train_acc"][last], 4),
            "val_loss": round(history["val_loss"][last], 4),
            "val_accuracy": round(history["val_acc"][last], 4),
        },
        "test": {
            "loss": round(te_loss, 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_macro": round(float(p_mac), 4),
            "recall_macro": round(float(r_mac), 4),
            "f1_macro": round(float(f_mac), 4),
            "precision_weighted": round(float(p_w), 4),
            "recall_weighted": round(float(r_w), 4),
            "f1_weighted": round(float(f_w), 4),
        },
        "reporte_por_clase": {c: {k: round(float(v), 4) for k, v in report_dict[c].items()}
                              for c in config.CLASSES},
        "matriz_confusion": cm.tolist(),
        "historial": {k: [round(float(x), 4) for x in v] for k, v in history.items()},
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(history.keys())
        w.writerows(zip(*history.values()))
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"SEMILLA = {config.SEMILLA} | {metrics['tipo_ejecucion']}\n\n{report_txt}")
    plot_learning_curves(history, out_dir / "loss_accuracy.png", best_epoch)
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png",
                          "" if not args.quick else " [prueba rápida]")
    print(f"Artefactos guardados en: {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
