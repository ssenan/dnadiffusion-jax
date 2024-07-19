import functools
from typing import Callable

import flax.linen as nn
import hydra
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jax import numpy as jnp
from jax.experimental import multihost_utils
from omegaconf import DictConfig
from tqdm import tqdm

from dnadiffusion.utils.train_utils import (
    create_checkpoint_manager,
    create_mesh,
    init_train_state,
    sample_step,
    train_step,
    val_step,
)
from dnadiffusion.utils.utils import convert_to_seq_jax


def train(
    mesh: jax.sharding.Mesh,
    repl_sharding: jax.sharding.NamedSharding,
    data_sharding: jax.sharding.NamedSharding,
    data_spec: jax.sharding.PartitionSpec,
    rng: jax.Array,
    model: nn.Module,
    tx: optax.GradientTransformation,
    loss_fn: Callable,
    dataset: tuple,
    batch_size: int,
    sample_batch_size: int,
    sequence_length: int,
    num_epochs: int,
    sample_epoch: int,
    checkpoint_epoch: int,
    number_of_samples: int,
    use_wandb: bool,
    path: str,
    d_params: dict | None = None,
) -> None:
    rng, init_rng = jax.random.split(rng)
    state, shardings = init_train_state(model, init_rng, mesh, tx)

    train_step_fn = jax.jit(
        functools.partial(train_step),
        in_shardings=(
            shardings,
            repl_sharding,
            data_sharding,
            data_sharding,
            repl_sharding,
        ),
        out_shardings=(shardings, repl_sharding),
        static_argnames=("loss_fn"),
    )
    val_step_fn = jax.jit(
        functools.partial(val_step),
        in_shardings=(
            shardings,
            repl_sharding,
            data_sharding,
            data_sharding,
            repl_sharding,
        ),
        out_shardings=repl_sharding,
        static_argnames=("loss_fn"),
    )

    sample_fn = jax.jit(
        functools.partial(sample_step),
        in_shardings=(
            shardings,
            repl_sharding,
            repl_sharding,
            repl_sharding,
        ),
        out_shardings=repl_sharding,
        static_argnums=(2, 3, 4, 5),
    )

    if jax.process_index() == 0 and use_wandb:
        id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=id)

    def batch_sharding(batch):
        def per_device_init_fn(index):
            return batch[index]

        global_input = jax.make_array_from_callback(batch.shape, data_sharding, per_device_init_fn)
        return global_input

    @jax.jit
    def get_batch(rng, dataset):
        idx = jax.random.choice(rng, len(dataset[0]), [batch_size])
        x, y = dataset[0][idx], dataset[1][idx]
        x = jnp.expand_dims(x, axis=-1)
        return x, y

    # Training
    x_data, y_data, x_val_data, y_val_data, cell_num_list, numeric_to_tag = dataset
    train_dataset_size = len(x_data)
    train_steps_per_epoch = np.ceil(train_dataset_size / batch_size).astype(int)
    val_dataset_size = len(x_val_data)
    val_steps_per_epoch = np.ceil(val_dataset_size / batch_size).astype(int)

    # Checkpointing
    # path = Path("checkpoints")
    # path = path.absolute()
    # path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = create_checkpoint_manager(path, ("state", "epoch"))

    global_step = 0

    for epoch in tqdm(range(num_epochs), disable=not jax.process_index() == 0, desc="Epochs"):
        rng, *rngs = jax.random.split(rng, 3)
        for _ in range(train_steps_per_epoch):
            x, y = get_batch(rngs[0], (x_data, y_data))

            x = batch_sharding(x)
            y = batch_sharding(y)
            # x, y = next(train_loader)
            state, loss = train_step_fn(state, loss_fn, rngs[1], x, y, d_params)
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
            val_loss = val_step_fn(state, loss_fn, rngs[1], x_val, y_val, d_params)

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

            for i in cell_num_list:
                samples = sample_fn(
                    state,
                    sample_rng,
                    sample_batch_size,
                    sequence_length,
                    i,
                    number_of_samples,
                    cell_num_list,
                    d_params,
                )

                multihost_utils.sync_global_devices("allgather")
                collected_samples = multihost_utils.process_allgather(samples, tiled=True)
                print(collected_samples.shape)
                if jax.process_index() == 0:
                    sequences = [convert_to_seq_jax(x, ["A", "C", "G", "T"]) for x in collected_samples]
                    sequences = jax.device_get(sequences)
                    with open(f"{path}/{numeric_to_tag[i]}.txt", "w") as f:
                        f.write("\n".join(sequences))

        if (epoch + 1) % checkpoint_epoch == 0:
            # multihost_utils.sync_global_devices("checkpointing")
            # Save checkpoint
            checkpoint_manager.save(
                global_step,
                args=ocp.args.Composite(
                    state=ocp.args.PyTreeSave(state),
                    epoch=ocp.args.JsonSave(epoch),
                ),
                force=True,
            )
            # checkpoint_manager.wait_until_finished()

    if jax.process_index() == 0 and use_wandb:
        wandb.finish()
        checkpoint_manager.wait_until_finished()
        print("Finished training")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    mesh, repl_sharding, data_sharding, data_spec, rng = create_mesh()

    # Training
    train_setup = {**cfg.training}
    train_setup["batch_size"] = train_setup["batch_size"] * jax.device_count()
    unet = hydra.utils.instantiate(cfg.models)
    d_params = hydra.utils.instantiate(cfg.diffusion)
    data = hydra.utils.instantiate(cfg.data)
    tx = hydra.utils.instantiate(cfg.optimizer)
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    train(
        mesh=mesh,
        repl_sharding=repl_sharding,
        data_sharding=data_sharding,
        data_spec=data_spec,
        rng=rng,
        model=unet,
        tx=tx,
        loss_fn=loss_fn,
        dataset=data,
        **train_setup,
        d_params=d_params,
    )


if __name__ == "__main__":
    main()
