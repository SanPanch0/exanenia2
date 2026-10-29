"""Pruebas del modelo (Pregunta 3). Ejecutar desde la raíz: python -m pytest -v"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.model import CNNCifar10, count_parameters, verify_shapes  # noqa: E402
from src.train import evaluate, train_one_epoch  # noqa: E402


@pytest.fixture
def model():
    config.set_seed(config.SEMILLA)
    return CNNCifar10()


def test_semilla_es_65():
    assert config.SEMILLA == 65


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_forma_salida(model, batch_size):
    out = model.eval()(torch.randn(batch_size, 3, 32, 32))
    assert out.shape == (batch_size, 10)


def test_shapes_intermedios():
    shapes = dict(verify_shapes(batch_size=8, device="cpu", verbose=False))
    assert shapes["Entrada"] == (8, 3, 32, 32)
    assert shapes["block1.MaxPool2d"] == (8, 32, 16, 16)
    assert shapes["block2.MaxPool2d"] == (8, 64, 8, 8)
    assert shapes["block3.MaxPool2d"] == (8, 128, 4, 4)
    assert shapes["flatten (Flatten)"] == (8, 2048)
    assert shapes["fc2 (Linear)"] == (8, 10)


def test_rechaza_tensor_no_4d(model):
    with pytest.raises(ValueError):
        model(torch.randn(3, 32, 32))


def test_sin_softmax_al_final(model):
    # La última capa es Linear: la salida son logits (no suman 1, pueden ser negativos)
    assert isinstance(model.fc2, nn.Linear)
    assert not any(isinstance(m, (nn.Softmax, nn.LogSoftmax)) for m in model.modules())


def test_bloques_contienen_conv_relu_maxpool(model):
    for block in (model.block1, model.block2, model.block3):
        tipos = [type(m) for m in block]
        assert nn.Conv2d in tipos and nn.ReLU in tipos and nn.MaxPool2d in tipos


def test_inicializacion_reproducible_con_semilla_65():
    config.set_seed(65)
    a = CNNCifar10().state_dict()
    config.set_seed(65)
    b = CNNCifar10().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_perdida_y_gradientes(model):
    model.train()
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def _loader(n=32, bs=8):
    g = torch.Generator().manual_seed(config.SEMILLA)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)


def test_entrenamiento_actualiza_pesos(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = train_one_epoch(model, _loader(), nn.CrossEntropyLoss(), opt, torch.device("cpu"))
    assert 0.0 <= acc <= 1.0 and loss > 0
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert any("conv" in k or k.startswith("block") for k in changed)
    assert "fc2.weight" in changed


def test_evaluacion_no_actualiza_pesos(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss, acc, preds, labels = evaluate(model, _loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    assert len(preds) == len(labels) == 32
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_numero_parametros(model):
    # 896 + 64 + 18 496 + 128 + 73 856 + 256 + 524 544 + 2 570 = 620 810
    assert count_parameters(model) == 620_810


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no disponible")
def test_forward_cuda():
    m = CNNCifar10().cuda().eval()
    assert m(torch.randn(8, 3, 32, 32, device="cuda")).shape == (8, 10)
