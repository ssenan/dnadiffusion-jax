import pickle
from typing import Any, Callable

import flax
import flax.linen as nn
import jax
import optax
from flax.training.train_state import TrainState
from google.cloud import storage
from jax import numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from orbax.checkpoint.checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerOptions,
)

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import p_loss
from dnadiffusion.utils.sample_utils import create_sample


def create_mesh() -> tuple[jax.sharding.Mesh, NamedSharding, NamedSharding, P, jax.Array]:
    jax.distributed.initialize()
    # jax.distributed.initialize(coordinator_address="localhost:8000", num_processes=1, process_id=0)

    if jax.process_index() == 0:
        print("Number of devices: ", jax.device_count())
        print("Local devices: ", jax.local_device_count())

    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
    repl_sharding = NamedSharding(
        mesh,
        P(),
    )

    data_sharding = NamedSharding(
        mesh,
        P(
            "data",
        ),
    )
    data_spec = P(
        "data",
    )

    rng = jax.random.PRNGKey(0)

    return mesh, repl_sharding, data_sharding, data_spec, rng


def init_train_state(
    model: nn.Module, rng: jax.Array, mesh: jax.sharding.Mesh, tx: optax.GradientTransformation
) -> tuple[TrainState, Any]:
    x = jax.ShapeDtypeStruct((16, 4, 200, 1), jnp.float32)
    t = jax.ShapeDtypeStruct((16,), jnp.int32)
    classes = jax.ShapeDtypeStruct((16,), jnp.int32)

    def init(rng, x, t, classes):
        params = model.init(rng, x, t, classes)
        return TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)

    params = jax.eval_shape(init, rng, x, t, classes)
    shardings = nn.get_sharding(params, mesh)
    state = jax.jit(init, out_shardings=shardings)(rng, x, t, classes)
    return state, shardings


def get_loss_fn(loss_type: str) -> Callable:
    if loss_type == "diffusion":
        return p_loss
    else:
        raise ValueError(f"Loss type {loss_type} not supported.")


def train_step(
    state: TrainState,
    loss_fn: Callable,
    rng: jax.Array,
    x: jax.Array,
    classes: jax.Array,
    d_params: dict[
        str,
        int,
    ],
    sharding: dict[str, NamedSharding] | None = None,
) -> tuple[TrainState, jnp.float32]:
    rng, step_rng = jax.random.split(rng)
    if sharding is not None:
        x = jax.lax.with_sharding_constraint(x, sharding)
        classes = jax.lax.with_sharding_constraint(classes, sharding)

    def loss_step(params):
        return loss_fn(
            rng=step_rng,
            state=state.replace(params=params),
            x=x,
            classes=classes,
            diffusion_params=d_params,
        )

    grad_fn = jax.value_and_grad(loss_step)
    loss, grads = grad_fn(state.params)

    # Optimizer update
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)

    # Model update
    params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        params=params,
        opt_state=opt_state,
        step=state.step + 1,
    )

    return new_state, loss


def val_step(
    state: TrainState,
    loss_fn: Callable,
    rng: jax.Array,
    x: jnp.ndarray,
    classes: jnp.ndarray,
    d_params: dict[str, int],
    sharding: dict[str, NamedSharding] | None = None,
) -> jnp.float32:
    rng, step_rng = jax.random.split(rng)
    if sharding is not None:
        x = jax.lax.with_sharding_constraint(x, sharding)
        classes = jax.lax.with_sharding_constraint(classes, sharding)

    def loss_step(params):
        return loss_fn(
            rng=step_rng,
            state=state.replace(params=params),
            x=x,
            classes=classes,
            diffusion_params=d_params,
        )

    loss = loss_step(state.params)

    return loss


def sample_step(
    state: TrainState,
    rng: jax.Array,
    sample_bs: int,
    sequence_length: int,
    group_number: int,
    number_of_samples: int,
    cell_num_list: list,
    d_params: dict[str, jax.Array] | None = None,
) -> jnp.ndarray:
    samples = create_sample(
        state=state,
        rng=rng,
        cell_types=cell_num_list,
        number_of_samples=number_of_samples,
        sample_bs=sample_bs,
        sequence_length=sequence_length,
        group_number=group_number,
        cond_weight_to_metric=1,
        diffusion_params=d_params,
    )
    return samples


def create_checkpoint_manager(
    checkpoint_dir: str, item_names: tuple, save_interval_steps: int = 1, use_async: bool = True
) -> CheckpointManager:
    manager = CheckpointManager(
        checkpoint_dir,
        item_names=item_names,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=use_async,
        ),
    )
    return manager


def get_dataset(
    data_path: str,
    saved_data_path: str,
    subset_list: list[str],
    limit_total_sequences: int,
    num_sampling_to_compare_cells: int,
    load_saved_data: bool,
    debug: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int], dict]:
    encode_data = load_data(
        data_path,
        saved_data_path,
        subset_list,
        limit_total_sequences,
        num_sampling_to_compare_cells,
        load_saved_data,
    )
    if debug:
        x_data = encode_data["X_train"][:1]
        y_data = encode_data["x_train_cell_type"][:1]
        x_val_data = encode_data["X_val"][:1]
        y_val_data = encode_data["x_val_cell_type"][:1]

    else:
        x_data = encode_data["X_train"]
        y_data = encode_data["x_train_cell_type"]
        x_val_data = encode_data["X_val"]
        y_val_data = encode_data["x_val_cell_type"]

    cell_num_list = encode_data["cell_types"]
    numeric_to_tag_dict = encode_data["numeric_to_tag"]

    return x_data, y_data, x_val_data, y_val_data, cell_num_list, numeric_to_tag_dict


def save_state(state: TrainState, path: str, epoch: int) -> None:
    tree = flax.serialization.to_state_dict(state)
    flattened_tree = flax.traverse_util.flatten_dict(tree, keep_empty_nodes=True)

    # Gather state from all devices
    state = multihost_utils.process_allgather(flattened_tree)

    # Save state to GCS
    storage_client = storage.Client()
    bucket_name = "dnadiffusion-bucket"
    bucket = storage_client.bucket(bucket_name)

    if jax.process_index() == 0:
        with open(path, "wb") as f:
            pickle.dump(state, f)

        blob_name = f"checkpoints/state_{epoch}.pkl"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(path)
        print(f"Saved checkpoint to {blob_name}")
