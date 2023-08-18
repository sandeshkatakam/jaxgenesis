import logging
import os 
import sys
import warnings
from typing import NewType, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu



try:
    import torch
except ImportError:
    warnings.warn("PyTorch is required for loading the pre-trained weights")

_TEMP_DIR = "/tmp/.eqx"
_Url = NewType("_Url", str)

GAN_MODELS_URLS = {
    "vanilla_gan": "",
    "dcgan":"",
}

VAE_MODELS_URLS = {
    "originalvae": "",
    "cvae": ""
}


def load_torch_weights(
    model: eqx.Module,
    torch_weights: str = None,
) -> eqx.Module:
    """Loads weights from a PyTorch serialised file.

    ???+ warning

        - This method requires installation of the [`torch`](https://pypi.org/project/torch/) package.

    !!! note

        - This function assumes that Eqxvision's ordering of class
          attributes mirrors the `torchvision.models` implementation.
        - This method assumes the `eqxvision` model is *not* initialised.
            Problems arise due to initialised `BN` modules.
        - The saved checkpoint should **only** contain model parameters as keys.

    !!! info
        A full list of pretrained URLs is provided
        [here](https://github.com/paganpasta/eqxvision/blob/main/eqxvision/utils.py).

    **Arguments:**

    - `model`: An `eqx.Module` for which the `jnp.ndarray` leaves are
        replaced by corresponding `PyTorch` weights.
    - `torch_weights`: A string either pointing to `PyTorch` weights on disk or the download `URL`.

    **Returns:**
        The model with weights loaded from the `PyTorch` checkpoint.
    """
    if "torch" not in sys.modules:
        raise RuntimeError(
            " Torch package not found! Pretrained is only supported with the torch package."
        )

    if torch_weights is None:
        raise ValueError("torch_weights parameter cannot be empty!")

    if not os.path.exists(torch_weights):
        global _TEMP_DIR
        filepath = os.path.join(_TEMP_DIR, os.path.basename(torch_weights))
        if os.path.exists(filepath):
            logging.info(
                f"Downloaded file exists at f{filepath}. Using the cached file!"
            )
        else:
            os.makedirs(_TEMP_DIR, exist_ok=True)
            torch.hub.download_url_to_file(torch_weights, filepath)
    else:
        filepath = torch_weights
    saved_weights = torch.load(filepath, map_location="cpu")
    weights_iterator = iter(
        [
            (name, jnp.asarray(weight.detach().numpy()))
            for name, weight in saved_weights.items()
            if "running" not in name and "num_batches" not in name
        ]
    )

    bn_s = []
    for name, weight in saved_weights.items():
        if "running_mean" in name:
            bn_s.append(False)
            bn_s.append(jnp.asarray(weight.detach().numpy()))
        elif "running_var" in name:
            bn_s.append(jnp.asarray(weight.detach().numpy()))
    bn_iterator = iter(bn_s)

    leaves, tree_def = jtu.tree_flatten(model)

    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and not (
            leaf.size == 1 and isinstance(leaf.item(), bool)
        ):
            (weight_name, new_weights) = next(weights_iterator)
            new_leaves.append(jnp.reshape(new_weights, leaf.shape))
        else:
            new_leaves.append(leaf)

    model = jtu.tree_unflatten(tree_def, new_leaves)

    def set_experimental(iter_bn, x):
        def set_values(y):
            if isinstance(y, eqx.experimental.StateIndex):
                current_val = next(iter_bn)
                if isinstance(current_val, bool):
                    eqx.experimental.set_state(y, jnp.asarray(False))
                else:
                    running_mean, running_var = current_val, next(iter_bn)
                    eqx.experimental.set_state(y, (running_mean, running_var))
            return y

        return jtu.tree_map(
            set_values, x, is_leaf=lambda _: isinstance(_, eqx.experimental.StateIndex)
        )

    model = jtu.tree_map(set_experimental, bn_iterator, model)
    return model