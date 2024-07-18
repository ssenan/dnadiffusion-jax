from typing import Any

import flax.linen as nn
import jax
import optax
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from orbax.checkpoint.checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerOptions,
)

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.utils.sample_utils import create_sample


def create_mesh(batch_size: int) -> tuple[jax.sharding.Mesh, NamedSharding, NamedSharding, P, jax.Array, int]:
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

    batch_size = batch_size * jax.device_count()

    return mesh, repl_sharding, data_sharding, data_spec, rng, batch_size


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


def train_step(
    state: TrainState,
    loss_fn: Any,
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
    loss_fn: Any,
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
    timesteps: int,
    sample_bs: int,
    sequence_length: int,
    group_number: int,
    number_of_samples: int,
    d_params: dict[str, jax.Array],
    cell_num_list: list,
) -> jnp.ndarray:
    samples = create_sample(
        state=state,
        rng=rng,
        timesteps=timesteps,
        diffusion_params=d_params,
        cell_types=cell_num_list,
        number_of_samples=number_of_samples,
        sample_bs=sample_bs,
        sequence_length=sequence_length,
        group_number=group_number,
        cond_weight_to_metric=1,
    )
    return samples


def create_checkpoint_manager(
    checkpoint_dir: str, item_names: tuple, save_interval_steps: int = 1, use_async: bool = True
) -> CheckpointManager:
    manager = CheckpointManager(
        checkpoint_dir,
        item_names=item_names,
        options=CheckpointManagerOptions(
            create=True,
            # save_interval_steps=save_interval_steps,
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
