import os

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from orbax.checkpoint.checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerOptions,
)
from tqdm import tqdm

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import p_loss
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.sample_util import create_sample
from dnadiffusion.utils.utils import convert_to_seq_jax, diffusion_params


def train_step(
    state: TrainState,
    rng: jax.Array,
    x: jnp.ndarray,
    classes: jnp.ndarray,
    timesteps: int,
    d_params: dict[
        str,
        int,
    ],
) -> tuple[TrainState, jnp.float32]:
    # rng = jax.random.fold_in(rng, jax.lax.axis_index("data"))
    rng, step_rng = jax.random.split(rng)

    def loss_fn(params):
        return p_loss(
            rng=step_rng,
            state=state.replace(params=params),
            x=x,
            classes=classes,
            timesteps=timesteps,
            diffusion_params=d_params,
        )

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Average gradients and loss across data parallel devices
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")

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
    rng: jax.Array,
    x: jnp.ndarray,
    classes: jnp.ndarray,
    timesteps: int,
    d_params: dict[str, int],
) -> jnp.float32:
    rng, step_rng = jax.random.split(rng)

    def loss_fn(params):
        return p_loss(
            rng=step_rng,
            state=state.replace(params=params),
            x=x,
            classes=classes,
            timesteps=timesteps,
            diffusion_params=d_params,
        )

    loss = loss_fn(state.params)
    loss = jax.lax.pmean(loss, axis_name="data")

    return loss


def sample_step(
    state: TrainState,
    rng: jax.Array,
    timesteps: int,
    sample_bs: int,
    sequence_length: int,
    group_number: int,
    number_of_samples: int,
    d_params: dict[str, jnp.ndarray],
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
    checkpoint_dir: str,
    item_names: tuple,
    save_interval_steps: int = 1,
    use_async: bool = True,
) -> CheckpointManager:
    manager = CheckpointManager(
        checkpoint_dir,
        item_names=item_names,
        options=CheckpointManagerOptions(
            create=True,
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=use_async,
        ),
    )
    return manager


def create_mesh_and_model(batch_size: int) -> tuple[jax.sharding.Mesh, NamedSharding, NamedSharding, P, jax.Array, int]:
    jax.distributed.initialize()
    # jax.distributed.initialize(coordinator_address="localhost:8000", num_processes=1, process_id=0)

    if jax.process_index() == 0:
        print("Number of devices: ", jax.device_count())
        print("Local devices: ", jax.local_device_count())

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = jax.sharding.Mesh(devices, ("data",))
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


def get_dataset(debug: bool = True) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int], dict]:
    encode_data = load_data(
        data_path="dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        saved_data_path="dnadiffusion/data/encode_data.pkl",
        subset_list=[
            "GM12878_ENCLB441ZZZ",
            "hESCT0_ENCLB449ZZZ",
            "K562_ENCLB843GMH",
            "HepG2_ENCLB029COU",
        ],
        limit_total_sequences=0,
        num_sampling_to_compare_cells=1000,
        load_saved_data=True,
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


def train(
    batch_size: int,
    sample_epoch: int,
    checkpoint_epoch: int,
    number_of_samples: int,
    use_wandb: bool,
) -> None:
    mesh, repl_sharding, data_sharding, data_spec, rng, batch_size = create_mesh_and_model(batch_size)

    # Creating the model
    rng, init_rng = jax.random.split(rng)
    unet = UNet(dim=200, resnet_block_groups=4)
    tx = optax.adam(1e-4)
    x_init = jnp.ones((16, 4, 200, 1), jnp.float32)
    class_init = jnp.ones((16,), jnp.int32)
    t_init = jnp.ones((16,), jnp.int32)

    # state_shape, state_spec = get_abstract_state(
    #     x_init.shape,
    #     t_init.shape,
    #     class_init.shape,
    #     unet,
    #     tx,
    #     rng,
    # )

    d_params = diffusion_params(timesteps=50, beta_start=0.0001, beta_end=0.2)

    train_step_fn = jax.jit(
        shard_map(
            train_step,
            mesh=mesh,
            in_specs=(P(), P(), P("data"), P("data"), P(), P()),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )

    val_step_fn = jax.jit(
        shard_map(
            val_step,
            mesh=mesh,
            in_specs=(P(), P(), P("data"), P("data"), P(), P()),
            out_specs=P(),
            check_rep=False,
        )
    )

    if jax.process_index() == 0 and use_wandb:
        id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=id)

    # Data parallel init
    with mesh:
        params = jax.jit(
            unet.init,
            in_shardings=(repl_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=repl_sharding,
        )(init_rng, x_init, t_init, class_init)

        # opt_shape = jax.eval_shape(tx.init, params_shape)

        state_dp = TrainState.create(
            apply_fn=unet.apply,
            params=params["params"],
            tx=tx,
        )

    # Diffusion parameters
    timesteps = 50
    d_params = diffusion_params(timesteps=50, beta_start=0.0001, beta_end=0.2)

    def batch_sharding(batch):
        def per_device_init_fn(index):
            return batch[index]

        global_input = jax.make_array_from_callback(batch.shape, data_sharding, per_device_init_fn)
        return global_input

    if jax.process_index() == 0 and use_wandb:
        id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=id)

    @jax.jit
    def get_batch(rng, dataset):
        idx = jax.random.choice(rng, len(dataset[0]), [batch_size])
        x, y = dataset[0][idx], dataset[1][idx]
        x = jnp.expand_dims(x, axis=-1)
        return x, y

    # Training
    x_data, y_data, x_val_data, y_val_data, cell_num_list, numeric_to_tag = get_dataset()
    train_dataset_size = len(x_data)
    train_steps_per_epoch = np.ceil(train_dataset_size / batch_size).astype(int)
    val_dataset_size = len(x_val_data)
    val_steps_per_epoch = np.ceil(val_dataset_size / batch_size).astype(int)

    # Get current absolute path
    absolute_path = os.path.abspath(os.getcwd())
    # Join the absolute path with the relative path
    path = os.path.join(absolute_path, "jax_checkpoints")

    checkpoint_manager = create_checkpoint_manager(path, ("state", "epoch", "global_step"))

    num_epochs = 2200
    global_step = 0
    # loss = 0.0

    for epoch in tqdm(range(num_epochs), disable=not jax.process_index() == 0, desc="Epochs"):
        rng, *rngs = jax.random.split(rng, 3)
        for _ in range(train_steps_per_epoch):
            x, y = get_batch(rngs[0], (x_data, y_data))

            x = batch_sharding(x)
            y = batch_sharding(y)
            # x, y = next(train_loader)
            state_dp, loss = train_step_fn(state_dp, rngs[1], x, y, timesteps, d_params)
            global_step += 1

            if jax.process_index() == 0 and use_wandb:
                if global_step % 10 == 0:
                    wandb.log(
                        {
                            "loss": loss,
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )

        for _ in range(val_steps_per_epoch):
            x_val, y_val = get_batch(rngs[0], (x_val_data, y_val_data))

            x_val = batch_sharding(x_val)
            y_val = batch_sharding(y_val)
            # x_val, y_val = next(val_loader)
            val_loss = val_step_fn(state_dp, rngs[1], x_val, y_val, timesteps, d_params)

        print(f"Epoch: {epoch}, Step: {global_step}, val_loss:{val_loss:.4f}, loss: {loss:.4f}")

        if jax.process_index() == 0 and use_wandb:
            wandb.log(
                {
                    "loss": loss,
                    "val_loss": val_loss,
                    "step": global_step,
                    "epoch": epoch,
                }
            )

        if (epoch + 1) % sample_epoch == 0:
            # Sampling
            rng, sample_rng = jax.random.split(rng)

            with mesh:
                sample_fn = jax.jit(
                    sample_step,
                    in_shardings=(
                        None,
                        repl_sharding,
                        repl_sharding,
                        repl_sharding,
                        repl_sharding,
                    ),
                    out_shardings=repl_sharding,
                    static_argnums=(3, 4, 5, 6),
                )
                for i in cell_num_list:
                    samples = sample_fn(
                        state_dp,
                        sample_rng,
                        timesteps,
                        1,
                        200,
                        i,
                        number_of_samples,
                        d_params,
                        cell_num_list,
                    )

                    collected_samples = multihost_utils.process_allgather(samples)
                    if jax.process_index() == 0:
                        sequences = [convert_to_seq_jax(x, ["A", "C", "G", "T"]) for x in collected_samples]
                        sequences = jax.device_get(sequences)
                        with open(f"./{numeric_to_tag[i]}.txt", "w") as f:
                            f.write("\n".join(sequences))

        if (epoch + 1) % checkpoint_epoch == 0:
            # Save checkpoint
            checkpoint_manager.save(
                global_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state_dp),
                    epoch=ocp.args.JsonSave(epoch),
                    global_step=ocp.args.JsonSave(global_step),
                ),
            )

    if jax.process_index() == 0 and use_wandb:
        wandb.finish()
        print("Finished training")


if __name__ == "__main__":
    # train(
    #     batch_size=1,
    #     sample_epoch=1,
    #     checkpoint_epoch=500,
    #     number_of_samples=1,
    #     use_wandb=False,
    # )
    train(
        batch_size=120,
        sample_epoch=10,
        checkpoint_epoch=10,
        number_of_samples=1000,
        use_wandb=True,
    )
