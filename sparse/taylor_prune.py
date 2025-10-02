import torch
from contextlib import contextmanager
from typing import Iterable, Dict, Any, Optional, Tuple


@contextmanager
def _train_mode(model: torch.nn.Module, train: bool):
    prev = model.training
    try:
        model.train(train)
        yield
    finally:
        model.train(prev)


def _is_bias(name: str, p: torch.nn.Parameter) -> bool:
    return p.ndim == 1 or ("bias" in name.lower())


@torch.no_grad()
def _apply_mask_inplace(
    param: torch.nn.Parameter, saliency: torch.Tensor, thresh: float
) -> Tuple[int, int]:
    if saliency.numel() == 0:
        return 0, param.numel()
    mask = (saliency > thresh).reshape_as(param)
    pruned = (~mask).sum().item()
    param.mul_(mask)
    return pruned, param.numel()


def prune_taylor_unstructured(
    model: torch.nn.Module,
    dataloader: Iterable,
    loss_fn,
    sparsity: float,
    device: Optional[torch.device] = None,
    num_batches: Optional[int] = None,
    forward_fn=None,
    skip_modules: Tuple[type, ...] = (),
) -> Dict[str, Any]:
    """
    Perform one-shot first-order Taylor unstructured pruning on the model.

    Args:
        model: nn.Module to prune (modified in-place).
        dataloader: iterable yielding (inputs, targets) or arbitrary batch objects.
        loss_fn: callable taking (outputs, targets) -> scalar loss (requires grad).
        sparsity: global fraction in [0,1) of weights to prune.
        device: torch.device to run saliency pass on. If None, infer from model parameters.
        num_batches: number of mini-batches to accumulate gradients over. If None, use
                     the whole dataloader (be careful with long loaders).
        forward_fn: optional callable(model, batch) -> (outputs, targets).
                    If None, assumes each batch is (inputs, targets).
        skip_modules: tuple of module types to skip (e.g., (torch.nn.Embedding,)).

    Returns:
        stats dict with keys:
          - 'requested_sparsity', 'achieved_sparsity'
          - 'total_weights_considered', 'total_pruned'
          - 'threshold'
          - 'layer_summaries': list of (name, pruned, total) per parameter tensor
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)."

    device = next(model.parameters()).device

    # Collect prunable params
    prunable: Dict[str, torch.nn.Parameter] = {}
    module_skips = set()
    for m in model.modules():
        if isinstance(m, skip_modules):
            module_skips.add(id(m))

    for name, p in model.named_parameters():
        # Skip if in a skipped module
        # (named_parameters returns 'module.sub.weight' — we need to map to a Module id).
        # Easiest: skip by heuristic on name matching child modules:
        skip_this = False
        cursor = model
        for part in name.split(".")[:-1]:
            if hasattr(cursor, part):
                cursor = getattr(cursor, part)
                if id(cursor) in module_skips:
                    skip_this = True
                    break
        if skip_this:
            continue

        if not p.requires_grad:
            continue
        if _is_bias(name, p):
            continue
        prunable[name] = p

    if not prunable:
        raise ValueError("No prunable (non-bias) parameters found.")

    # Zero existing grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Accumulate gradients over a few mini-batches
    batches_seen = 0
    with _train_mode(model, True):
        for batch in dataloader:
            if forward_fn is None:
                inputs, targets = batch
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
            else:
                outputs, targets = forward_fn(model, batch)
                if (
                    not isinstance(targets, torch.Tensor)
                    or not targets.is_cuda
                    and device.type == "cuda"
                ):
                    targets = targets.to(device, non_blocking=True)

            loss = loss_fn(outputs, targets)
            # Backprop to populate .grad for weights used
            model.zero_grad(set_to_none=True)
            loss.backward()  # keep graph=False; we just need parameter grads

            batches_seen += 1
            if num_batches is not None and batches_seen >= num_batches:
                break

    # Build a flat saliency vector for all prunable params
    saliencies_flat = []
    param_keys = []
    param_sizes = []
    total_considered = 0

    for name, p in prunable.items():
        if p.grad is None:
            # Parameter not involved in the loss over sampled batches — skip from pruning pool
            continue
        s = (p.detach() * p.grad.detach()).abs()
        if s.numel() == 0:
            continue
        saliencies_flat.append(s.reshape(-1).cpu())
        param_keys.append(name)
        param_sizes.append(s.numel())
        total_considered += s.numel()

    if total_considered == 0:
        raise RuntimeError(
            "No gradients were observed for prunable parameters; cannot compute Taylor saliency."
        )

    # Concatenate and find global threshold for desired sparsity
    sal_all = torch.cat(saliencies_flat, dim=0)
    k = int(total_considered * sparsity)
    threshold: float
    if k <= 0:
        threshold = float("-inf")  # prune nothing
    elif k >= sal_all.numel():
        threshold = float("inf")  # prune everything (not recommended)
    else:
        # kthvalue finds the k-th smallest; we want the value such that <= thresh are pruned
        threshold = sal_all.kthvalue(k).values.item()

    # Apply masks in-place per-tensor
    stats = {
        "requested_sparsity": sparsity,
        "achieved_sparsity": None,
        "total_weights_considered": total_considered,
        "total_pruned": 0,
        "threshold": threshold,
        "layer_summaries": [],  # (name, pruned, total)
    }

    # We must recompute per-parameter saliency again for masking (to avoid storing all s tensors)
    idx = 0
    for name, p in prunable.items():
        if p.grad is None or p.numel() == 0:
            continue
        s = (p.detach() * p.grad.detach()).abs()
        pruned, total = _apply_mask_inplace(p, s, threshold)
        stats["layer_summaries"].append((name, pruned, total))
        stats["total_pruned"] += pruned
        idx += 1

    stats["achieved_sparsity"] = stats["total_pruned"] / max(
        1, stats["total_weights_considered"]
    )
    return stats
