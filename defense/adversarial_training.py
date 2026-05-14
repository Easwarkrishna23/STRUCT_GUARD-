"""Adversarial retraining used by the STRUC-GUARD+ defense chain."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets.cora_loader import GraphData
from models.train import GNNTrainState, TrainResult, cross_entropy_loss, eval_step, train_step
from utils.config import DefenseConfig, ModelConfig
from utils.graph_utils import normalize_adjacency


def adversarial_retrain_model(
    model,
    graph: GraphData,
    model_cfg: ModelConfig,
    defense_cfg: DefenseConfig,
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """Train with feature FGSM plus small edge-drop/add augmentation each epoch."""
    from models.gat import GAT

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, init_key, dropout_key = jax.random.split(key, 3)

    x_base = jnp.array(graph.features)
    labels = jnp.array(graph.labels)
    train_mask = jnp.array(graph.train_mask)
    val_mask = jnp.array(graph.val_mask)
    use_adj = isinstance(model, GAT)

    init_graph_input = jnp.array(graph.adj if use_adj else graph.adj_norm)
    params = model.init(
        {"params": init_key, "dropout": dropout_key},
        x_base,
        init_graph_input,
        training=False,
    )["params"]
    tx = optax.adamw(learning_rate=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay)
    state = GNNTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_key=dropout_key,
    )

    best_val_acc = 0.0
    best_params: Any = params
    best_epoch = 0
    patience_counter = 0
    train_losses: list[float] = []
    val_accs: list[float] = []

    for epoch in range(1, model_cfg.epochs + 1):
        adj_aug = _augment_edges(
            graph.adj,
            drop_rate=defense_cfg.adv_edge_drop_rate,
            add_rate=defense_cfg.adv_edge_add_rate,
            rng=rng,
        )
        graph_input_np = adj_aug if use_adj else normalize_adjacency(adj_aug)
        graph_input = jnp.array(graph_input_np)
        x_adv = _feature_fgsm(
            model,
            state.params,
            x_base,
            graph_input,
            labels,
            train_mask,
            epsilon=defense_cfg.adv_feature_epsilon,
        )

        state, loss = train_step(state, model, x_adv, graph_input, labels, train_mask)
        _, val_acc, _, _ = eval_step(state.params, model, x_base, init_graph_input, labels, val_mask)

        train_losses.append(float(loss))
        val_accs.append(float(val_acc))

        if float(val_acc) > best_val_acc:
            best_val_acc = float(val_acc)
            best_params = state.params
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 50 == 0:
            print(f"  Adv epoch {epoch:>3d}/{model_cfg.epochs} | "
                  f"loss={float(loss):.4f} | val_acc={float(val_acc):.4f}")

        if patience_counter >= model_cfg.patience:
            break

    return TrainResult(best_params, train_losses, val_accs, best_val_acc, best_epoch)


def _feature_fgsm(model, params, x, graph_input, labels, train_mask, epsilon: float):
    def loss_fn(x_):
        _, logits, _ = model.apply({"params": params}, x_, graph_input, training=False)
        return cross_entropy_loss(logits, labels, train_mask)

    grad = jax.grad(loss_fn)(x)
    mask = train_mask[:, None]
    x_adv = x + epsilon * jnp.sign(jnp.where(mask, grad, 0.0))
    lo = jnp.min(x, axis=0)
    hi = jnp.max(x, axis=0)
    return jnp.clip(x_adv, lo, hi)


def _augment_edges(
    adj: np.ndarray,
    drop_rate: float,
    add_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    adj_aug = (adj > 0).astype(np.float32).copy()
    rows, cols = np.where(np.triu(adj_aug, k=1) > 0)
    n_edges = len(rows)
    n_drop = min(n_edges, int(n_edges * drop_rate))
    if n_drop > 0:
        chosen = rng.choice(n_edges, size=n_drop, replace=False)
        for idx in chosen:
            u, v = rows[idx], cols[idx]
            adj_aug[u, v] = 0.0
            adj_aug[v, u] = 0.0

    n_add = int(n_edges * add_rate)
    n = adj_aug.shape[0]
    added = 0
    attempts = 0
    while added < n_add and attempts < max(n_add * 20, 1):
        u, v = rng.integers(0, n, size=2)
        if u != v and adj_aug[u, v] == 0:
            adj_aug[u, v] = 1.0
            adj_aug[v, u] = 1.0
            added += 1
        attempts += 1
    return adj_aug
