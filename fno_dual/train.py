import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    add_grid_channel_1d,
    add_grid_channels_2d,
    generate_darcy_dataset,
    generate_wave_dataset,
)
from .models import FNO1d, FNO2d
from .normalizer import UnitGaussianNormalizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_val(A, U, val_fraction=0.0):
    if val_fraction <= 0:
        return A, U, None, None
    n_val = int(A.shape[0] * val_fraction)
    if n_val <= 0:
        return A, U, None, None
    perm = torch.randperm(A.shape[0], device=A.device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return A[train_idx], U[train_idx], A[val_idx], U[val_idx]


@torch.no_grad()
def evaluate(model, A, U, y_normalizer):
    model.eval()
    pred = y_normalizer.decode(model(A))
    true = y_normalizer.decode(U)
    mse = F.mse_loss(pred, true).item()
    global_rel_l2 = (torch.norm(pred - true) / (torch.norm(true) + 1e-12)).item()
    flat_err = (pred - true).reshape(pred.shape[0], -1)
    flat_true = true.reshape(true.shape[0], -1)
    sample_rel_l2 = torch.linalg.norm(flat_err, dim=1) / (torch.linalg.norm(flat_true, dim=1) + 1e-12)
    return {
        "mse": mse,
        "rel_l2": global_rel_l2,
        "global_rel_l2": global_rel_l2,
        "sample_rel_l2_mean": sample_rel_l2.mean().item(),
        "sample_rel_l2_std": sample_rel_l2.std(unbiased=False).item(),
        "sample_rel_l2_median": sample_rel_l2.median().item(),
    }


def parameter_metrics(model):
    """Report total/active separately; MSSA currently uses dense execution."""
    total = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "total_params": total,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "active_params": total,
    }
    if hasattr(model, "parameter_breakdown"):
        metrics.update({f"{name}_params": count for name, count in model.parameter_breakdown().items()
                        if name != "total"})
    return metrics


def train_model(
    model,
    A_train,
    U_train,
    A_test,
    U_test,
    y_normalizer,
    epochs,
    batch_size,
    lr=1e-3,
    log_interval=0,
    adapter_warmup_fraction=0.0,
    A_val=None,
    U_val=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.5)
    best_val = float("inf")
    best_val_epoch = 0
    best_state = None

    warmup_epochs = int(epochs * adapter_warmup_fraction)
    if hasattr(model, "set_adapter_enabled"):
        model.set_adapter_enabled(warmup_epochs == 0)

    for ep in range(1, epochs + 1):
        if hasattr(model, "set_adapter_enabled") and ep == warmup_epochs + 1:
            model.set_adapter_enabled(True)
        model.train()
        perm = torch.randperm(A_train.shape[0], device=A_train.device)
        train_loss = 0.0
        for start in range(0, A_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            pred = model(A_train[idx])
            loss = F.mse_loss(pred, U_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * idx.shape[0]
        scheduler.step()
        if A_val is not None and U_val is not None:
            val_metrics = evaluate(model, A_val, U_val, y_normalizer)
            if val_metrics["rel_l2"] < best_val:
                best_val = val_metrics["rel_l2"]
                best_val_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log_interval and (ep == 1 or ep % log_interval == 0 or ep == epochs):
            train_loss /= A_train.shape[0]
            if A_val is not None and U_val is not None:
                metrics = val_metrics
                metric_split = "val"
            else:
                metrics = evaluate(model, A_test, U_test, y_normalizer)
                metric_split = "test"
            print(
                f"  epoch={ep:03d} train_mse_norm={train_loss:.4e} "
                f"{metric_split}_mse={metrics['mse']:.4e} {metric_split}_rel_l2={metrics['rel_l2']:.4e}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, A_test, U_test, y_normalizer)
    if best_state is not None:
        metrics["best_val_rel_l2"] = best_val
        metrics["best_val_epoch"] = best_val_epoch
    return metrics


def run_wave_once(
    method,
    seed=0,
    device="cpu",
    n_train=512,
    n_test=128,
    n_grid=128,
    epochs=100,
    batch_size=32,
    width=32,
    modes=16,
    log_interval=0,
):
    data_seed = method.get("data_seed", seed)
    model_seed = method.get("model_seed", seed)
    set_seed(data_seed)
    A_train, U_train = generate_wave_dataset(n_samples=n_train, n_grid=n_grid, seed=data_seed, device=device)
    A_test, U_test = generate_wave_dataset(n_samples=n_test, n_grid=n_grid, seed=1000 + data_seed, device=device)
    A_train = add_grid_channel_1d(A_train)
    A_test = add_grid_channel_1d(A_test)
    A_train, U_train, A_val, U_val = split_train_val(A_train, U_train, method.get("val_fraction", 0.0))

    x_normalizer = UnitGaussianNormalizer(A_train)
    y_normalizer = UnitGaussianNormalizer(U_train)
    A_train_n = x_normalizer.encode(A_train)
    A_test_n = x_normalizer.encode(A_test)
    U_train_n = y_normalizer.encode(U_train)
    U_test_n = y_normalizer.encode(U_test)
    A_val_n = x_normalizer.encode(A_val) if A_val is not None else None
    U_val_n = y_normalizer.encode(U_val) if U_val is not None else None

    set_seed(model_seed)
    model = FNO1d(
        in_dim=3,
        width=width,
        modes=modes,
        gate_variant=method.get("gate_variant"),
        gate_learnable=method.get("gate_learnable", True),
        gate_strength=method.get("gate_strength", 1.0),
        highfreq_only=method.get("highfreq_only", False),
        low_keep=method.get("low_keep", 4),
        gate_mode=method.get("gate_mode", "legacy"),
        residual_beta=method.get("residual_beta", 0.2),
        use_dynamic_score=method.get("use_dynamic_score", True),
        use_model_score=method.get("use_model_score", True),
    ).to(device)
    model.fit_gates_from_data(U_train)
    metrics = train_model(
        model,
        A_train_n,
        U_train_n,
        A_test_n,
        U_test_n,
        y_normalizer,
        epochs,
        batch_size,
        log_interval=log_interval,
        adapter_warmup_fraction=method.get("adapter_warmup_fraction", 0.0),
        A_val=A_val_n,
        U_val=U_val_n,
    )
    metrics.update(parameter_metrics(model))
    return metrics


def run_darcy_once(
    method,
    seed=0,
    device="cpu",
    n_train=256,
    n_test=64,
    n_grid=32,
    epochs=40,
    batch_size=16,
    width=32,
    modes=12,
    log_interval=0,
):
    data_seed = method.get("data_seed", seed)
    model_seed = method.get("model_seed", seed)
    set_seed(data_seed)
    A_train, U_train = generate_darcy_dataset(n_samples=n_train, n_grid=n_grid, seed=data_seed, device=device)
    A_test, U_test = generate_darcy_dataset(n_samples=n_test, n_grid=n_grid, seed=1000 + data_seed, device=device)
    A_train = add_grid_channels_2d(A_train)
    A_test = add_grid_channels_2d(A_test)
    A_train, U_train, A_val, U_val = split_train_val(A_train, U_train, method.get("val_fraction", 0.0))

    x_normalizer = UnitGaussianNormalizer(A_train)
    y_normalizer = UnitGaussianNormalizer(U_train)
    A_train_n = x_normalizer.encode(A_train)
    A_test_n = x_normalizer.encode(A_test)
    U_train_n = y_normalizer.encode(U_train)
    U_test_n = y_normalizer.encode(U_test)
    A_val_n = x_normalizer.encode(A_val) if A_val is not None else None
    U_val_n = y_normalizer.encode(U_val) if U_val is not None else None

    set_seed(model_seed)
    if method.get("backend") == "neuralop":
        from neuralop.models import FNO

        A_train_model = A_train_n.permute(0, 3, 1, 2).contiguous()
        A_test_model = A_test_n.permute(0, 3, 1, 2).contiguous()
        U_train_model = U_train_n.unsqueeze(1).contiguous()
        U_test_model = U_test_n.unsqueeze(1).contiguous()
        A_val_model = A_val_n.permute(0, 3, 1, 2).contiguous() if A_val_n is not None else None
        U_val_model = U_val_n.unsqueeze(1).contiguous() if U_val_n is not None else None
        model = FNO(
            n_modes=(modes, modes),
            in_channels=3,
            out_channels=1,
            hidden_channels=width,
            n_layers=4,
            positional_embedding=None,
        ).to(device)
    else:
        A_train_model = A_train_n
        A_test_model = A_test_n
        U_train_model = U_train_n
        U_test_model = U_test_n
        A_val_model = A_val_n
        U_val_model = U_val_n
        model = FNO2d(
            in_dim=3,
            width=width,
            modes1=modes,
            modes2=modes,
            gate_variant=method.get("gate_variant"),
            gate_learnable=method.get("gate_learnable", True),
            gate_strength=method.get("gate_strength", 0.2),
            highfreq_only=method.get("highfreq_only", False),
            gate_mode=method.get("gate_mode", "legacy"),
            residual_beta=method.get("residual_beta", 0.2),
            use_dynamic_score=method.get("use_dynamic_score", True),
            use_model_score=method.get("use_model_score", True),
            n_bands=method.get("n_bands", 4),
            use_adapter=method.get("use_adapter", False),
            adapter_rank=method.get("adapter_rank", 4),
        ).to(device)
        model.fit_gates_from_data(U_train)

    metrics = train_model(
        model,
        A_train_model,
        U_train_model,
        A_test_model,
        U_test_model,
        y_normalizer,
        epochs,
        batch_size,
        log_interval=log_interval,
        adapter_warmup_fraction=method.get("adapter_warmup_fraction", 0.0),
        A_val=A_val_model,
        U_val=U_val_model,
    )
    metrics.update(parameter_metrics(model))
    artifacts = method.get("artifacts")
    if artifacts is not None:
        artifacts.update(
            {
                "model": model,
                "A_train": A_train_model,
                "U_train": U_train_model,
                "A_val": A_val_model,
                "U_val": U_val_model,
                "A_test": A_test_model,
                "U_test": U_test_model,
                "y_normalizer": y_normalizer,
            }
        )
    return metrics
