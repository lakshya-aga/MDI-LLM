#!/usr/bin/env python3

import importlib

from . import functional
from .convert_hf_checkpoint import convert_hf_checkpoint, copy_weights_hf_llama
from .utils import (count_transformer_blocks, deserialize_params,
                    detect_stop_tokens, estimate_loss, find_eot, get_lr,
                    get_obj_size, load_from_hf, load_from_pt, load_sd,
                    loading_bar, remove_prefix, serialize_params,
                    split_and_store, split_parameters, waiting_animation, init_from_state_dict)
from .context_managers import catch_loop_errors

# Names whose modules pull in heavy, inference-irrelevant dependencies (numpy/tiktoken
# for data loading, matplotlib for plots, packaging/pkg_resources for downloads).
# They are imported on first access, so inference nodes (e.g., phones) don't need them.
_LAZY = {
    "get_batch": ".data_loader",
    "load_dataset": ".data_loader",
    "split_dataset": ".data_loader",
    "download_from_hub": ".download",
    "plot_tokens_per_time": ".plots",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "convert_hf_checkpoint",
    "copy_weights_hf_llama",
    "get_batch",
    "load_dataset",
    "split_dataset",
    "download_from_hub",
    "plot_tokens_per_time",
    "count_transformer_blocks",
    "deserialize_params",
    "estimate_loss",
    "find_eot",
    "get_lr",
    "get_obj_size",
    "load_from_hf",
    "load_from_pt",
    "loading_bar",
    "remove_prefix",
    "serialize_params",
    "split_parameters",
    "load_sd",
    "split_and_store",
    "functional",
    "detect_stop_tokens",
    "waiting_animation",
    "catch_loop_errors",
    "init_from_state_dict",
]
