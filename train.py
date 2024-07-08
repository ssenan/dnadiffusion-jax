import os
from functools import partial

import jax
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax.experimental import mesh_utils
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
from dnadiffusion.utils.utils import diffusion_params, get_abstract_state, get_init_state


def train_step(state, rng, x, classes, timesteps, d_params):
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


def val_step(state, rng, x, classes, timesteps, d_params):
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


def create_checkpoint_manager(
    checkpoint_dir: str,
    use_async: bool,
    item_names: tuple[str],
    save_interval_steps: int,
):
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


def create_mesh_and_model():
    jax.distributed.initialize()

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

    return mesh, repl_sharding, data_sharding, data_spec, rng


def sample_step(
    state: TrainState,
    rng: jax.Array,
    timesteps: int,
    sample_bs: int,
    sequence_length: int,
    number_of_samples: int,
    d_params: dict[str, jnp.ndarray],
    cell_num_list: list,
):
    for i in cell_num_list:
        samples = create_sample(
            state=state,
            rng=rng,
            timesteps=timesteps,
            diffusion_params=d_params,
            cell_types=cell_num_list,
            number_of_samples=number_of_samples,
            sample_bs=sample_bs,
            sequence_length=sequence_length,
            group_number=i,
            cond_weight_to_metric=1,
        )
        return samples


def train():
    mesh, repl_sharding, data_sharding, data_spec, rng = create_mesh_and_model()

    # Loading data
    encode_data = load_data(
        data_path="./K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        saved_data_path="./encode_data.pkl",
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

    # Creating the model
    unet = UNet(dim=200, resnet_block_groups=4)
    tx = optax.adam(1e-4)
    x_init = jnp.ones((16, 4, 200, 1), jnp.float32)
    class_init = jnp.ones((16,), jnp.int32)
    t_init = jnp.ones((16,), jnp.int32)

    state_shape, state_spec = get_abstract_state(
        x_init.shape,
        t_init.shape,
        class_init.shape,
        unet,
        tx,
        rng,
    )

    with mesh:
        state = jax.jit(
            partial(get_init_state, model=unet, tx=tx),
            in_shardings=(repl_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=repl_sharding,
        )(rng, x_init, t_init, class_init)

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

    encode_data = load_data(
        data_path="./K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        saved_data_path="./encode_data.pkl",
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

    # NOTE: put back
    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=id)

    # Data parallel init
    with mesh:
        params = jax.jit(
            unet.init,
            in_shardings=(repl_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=repl_sharding,
        )(init_rng, x_init, t_init, class_init)

        opt_shape = jax.eval_shape(tx.init, params_shape)

        train_state = TrainState.create(
            apply_fn=unet.apply,
            params=params["params"],
            tx=tx,
        )

    # Diffusion parameters
    timesteps = 50
    # d_params = diffusion_params(timesteps=50, beta_start=0.0001, beta_end=0.005)
    d_params = diffusion_params(timesteps=50, beta_start=0.0001, beta_end=0.2)

    def batch_sharding(batch):
        def per_device_init_fn(index):
            return batch[index]

        global_input = jax.make_array_from_callback(batch.shape, data_sharding, per_device_init_fn)
        return global_input

    # NOTE: put back
    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=id)

    @jax.jit
    def get_batch(rng, dataset):
        idx = jax.random.choice(rng, len(dataset[0]), [batch_size])
        x, y = dataset[0][idx], dataset[1][idx]
        x = jnp.expand_dims(x, axis=-1)
        return x, y

    train_dataset_size = len(encode_data["X_train"])
    train_steps_per_epoch = np.ceil(train_dataset_size / batch_size).astype(int)
    val_dataset_size = len(encode_data["X_val"])
    val_steps_per_epoch = np.ceil(val_dataset_size / batch_size).astype(int)

    num_epochs = 2200
    global_step = 0
    # loss = 0.0

    # Get current absolute path
    absolute_path = os.path.abspath(os.getcwd())
    # Join the absolute path with the relative path
    path = os.path.join(absolute_path, "jax_checkpoints")
    # path = ocp.test_utils.erase_and_create_empty(path)
    # Checkpoint manager
    options = ocp.CheckpointManagerOptions(create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(path),
        item_names=("state", "epoch", "global_step"),
        options=options,
    )

    # NOTE: put back
    # x_data = encode_data["X_train"][:1]
    # y_data = encode_data["x_train_cell_type"][:1]
    # x_val_data = encode_data["X_val"][:1]
    # y_val_data = encode_data["x_val_cell_type"][:1]
    # train_steps_per_epoch = 1
    # val_steps_per_epoch = 1

    for epoch in tqdm(range(num_epochs), disable=not jax.process_index() == 0, desc="Epochs"):
        rng, *rngs = jax.random.split(rng, 3)
        for _ in range(train_steps_per_epoch):
            # NOTE:
            x, y = get_batch(rngs[0], (encode_data["X_train"], encode_data["x_train_cell_type"]))
            # x, y = get_batch(rngs[0], (x_data, y_data))

            x = batch_sharding(x)
            y = batch_sharding(y)
            # x, y = next(train_loader)
            state_dp, loss = train_step_fn(state_dp, rngs[1], x, y, timesteps, d_params)
            global_step += 1

            # NOTE:
            if jax.process_index() == 0:
                if global_step % 10 == 0:
                    wandb.log(
                        {
                            "loss": loss,
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )

        for _ in range(val_steps_per_epoch):
            # NOTE:
            x_val, y_val = get_batch(rngs[0], (encode_data["X_val"], encode_data["x_val_cell_type"]))
            # x_val, y_val = get_batch(rngs[0], (x_val_data, y_val_data))

            x_val = batch_sharding(x_val)
            y_val = batch_sharding(y_val)
            # x_val, y_val = next(val_loader)
            val_loss = val_step_fn(state_dp, rngs[1], x_val, y_val, timesteps, d_params)

        # print(
        #     f"Epoch: {epoch}, Step: {global_step}, val_loss:{val_loss:.4f}, loss: {loss:.4f}"
        # )

        # NOTE: put back
        if jax.process_index() == 0:
            wandb.log(
                {
                    "loss": loss,
                    "val_loss": val_loss,
                    "step": global_step,
                    "epoch": epoch,
                }
            )

        # if (epoch + 1) % 1 == 0:
        #     print(f"Sampling at epoch: {epoch+1}")
        #     # Sampling
        #     rng, sample_rng = jax.random.split(rng)
        #
        #     number_of_samples = 100
        #
        #     cell_num_list = encode_data["cell_types"]
        #
        #     # FIX: Hack for using shard_map with static_argnums
        #     sample_timesteps = jnp.zeros((50,), dtype=jnp.int32)
        #     number_of_samples = 800 // jax.device_count()
        #     number_of_samples = jnp.zeros((number_of_samples,), dtype=jnp.int32)
        #     sample_bs = jnp.zeros((10,), dtype=jnp.int32)
        #     sequence_length = jnp.zeros((200,), dtype=jnp.int32)
        #     for i in cell_num_list:
        #         # samples = create_sample_dp(
        #         #     state=state_dp,
        #         #     rng=sample_rng,
        #         #     timesteps=timesteps,
        #         #     diffusion_params=d_params,
        #         #     cell_types=encode_data["cell_types"],
        #         #     # conditional_numeric_to_tag=encode_data["numeric_to_tag"],
        #         #     number_of_samples=100,
        #         #     group_number=i,
        #         #     # cond_weight_to_metric=0.0,
        #         #     cond_weight_to_metric=1,
        #         #     # save_datafame=False,
        #         # )
        #         i = jnp.zeros((i,), dtype=jnp.int32)
        #         samples = create_sample_dp(
        #             state_dp,
        #             sample_rng,
        #             sample_timesteps,
        #             d_params,
        #             jnp.array(cell_num_list),
        #             number_of_samples,
        #             sample_bs,
        #             sequence_length,
        #             i,
        #             1,
        #         )
        #         print("Sample Success")
        #
        #         if jax.process_index() == 0:
        #             collected_samples = multihost_utils.process_allgather(samples)
        #             sequences = [
        #                 convert_to_seq_jax(x, ["A", "C", "G", "T"])
        #                 for x in collected_samples
        #             ]
        #             sequences = jax.device_get(sequences)
        #             print(
        #                 f"Saving sequences for {encode_data['numeric_to_tag'][i.shape[0]]}"
        #             )
        #             with open(
        #                 f"./{encode_data['numeric_to_tag'][i.shape[0]]}.txt", "w"
        #             ) as f:
        #                 f.write("\n".join(sequences))

    # NOTE:
    if jax.process_index() == 0:
        wandb.finish()
        print("Finished training")


def old_train():
    # NOTE:
    jax.distributed.initialize()
    # jax.distributed.initialize(
    #     coordinator_address="localhost:8000", num_processes=1, process_id=0
    # )
    if jax.process_index() == 0:
        print("Number of devices: ", jax.device_count())
        print("Local devices: ", jax.local_device_count())
    # NOTE: REMOVE
    # batch_size_per_device = 1
    batch_size_per_device = 120
    batch_size = batch_size_per_device * jax.device_count()

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

    # Loading data
    encode_data = load_data(
        data_path="./K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        saved_data_path="./encode_data.pkl",
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

    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    x_init = jnp.ones((16, 4, 200, 1), jnp.float32)
    class_init = jnp.ones((16,), jnp.int32)
    t_init = jnp.ones((16,), jnp.int32)

    # Creating the model
    unet = UNet(dim=200, resnet_block_groups=4)
    # Print total parameter count for the model
    tx = optax.adam(1e-4)

    def init_fn(rng, x, t, in_class, model, tx):
        variables = model.init(rng, x, t, in_class)
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=tx,
        )
        return state

    params_shape = jax.eval_shape(unet.init, init_rng, x_init, t_init, class_init)
    params_spec = nn.get_partition_spec(params_shape)
    grad_spec = params_spec

    # Data parallel init
    with mesh:
        params = jax.jit(
            unet.init,
            in_shardings=(repl_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=repl_sharding,
        )(init_rng, x_init, t_init, class_init)

        opt_shape = jax.eval_shape(tx.init, params_shape)

        state_dp = TrainState.create(
            apply_fn=unet.apply,
            params=params["params"],
            tx=tx,
        )

    # Diffusion parameters
    timesteps = 50
    # d_params = diffusion_params(timesteps=50, beta_start=0.0001, beta_end=0.005)
    d_params = diffusion_params(timesteps=50, beta_start=0.0001, beta_end=0.2)

    # d_params = shard_map(
    #     diffusion_params,
    #     mesh=mesh,
    #     in_specs=(P(), P(), P()),
    #     out_specs=P(),
    #     check_rep=False,
    # )(timesteps, 0.0001, 0.005)

    # Calculating loss

    def train_step(state, rng, x, classes, timesteps, d_params):
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

    def val_step(state, rng, x, classes, timesteps, d_params):
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

    create_sample_dp = shard_map(
        create_sample,
        mesh=mesh,
        # in_specs=(P(), P(), P(), P(), P(), None, None, None, None, None),
        in_specs=(P(), P(), P(), P(), P(), P(), P(), P(), P(), P()),
        out_specs=P(),
        check_rep=False,
    )

    # Training loop

    # Batch sharding
    # def batch_sharding(x):
    # return multihost_utils.host_local_array_to_global_array(x, mesh, data_spec)
    def batch_sharding(batch):
        def per_device_init_fn(index):
            return batch[index]

        global_input = jax.make_array_from_callback(batch.shape, data_sharding, per_device_init_fn)
        return global_input

    # NOTE: put back
    if jax.process_index() == 0:
        id = wandb.util.generate_id()
        wandb.init(project="dnadiffusion", id=id)

    @jax.jit
    def get_batch(rng, dataset):
        idx = jax.random.choice(rng, len(dataset[0]), [batch_size])
        x, y = dataset[0][idx], dataset[1][idx]
        x = jnp.expand_dims(x, axis=-1)
        return x, y

    # @jax.jit
    # def data_loader(rng, dataset, steps, batch_size):
    #     dataset_x, dataset_y = dataset
    #     while True:
    #         rng, subkey = jax.random.split(rng)
    #         for _ in range(steps):
    #             idx = jax.random.choice(subkey, len(dataset_x), (batch_size,))
    #             x, y = dataset_x[idx], dataset_y[idx]
    #             x = jnp.expand_dims(x, axis=-1)
    #             yield batch_sharding(x), batch_sharding(y)

    train_dataset_size = len(encode_data["X_train"])
    train_steps_per_epoch = np.ceil(train_dataset_size / batch_size).astype(int)
    val_dataset_size = len(encode_data["X_val"])
    val_steps_per_epoch = np.ceil(val_dataset_size / batch_size).astype(int)

    num_epochs = 2200
    global_step = 0
    # loss = 0.0

    # Get current absolute path
    absolute_path = os.path.abspath(os.getcwd())
    # Join the absolute path with the relative path
    path = os.path.join(absolute_path, "jax_checkpoints")
    # path = ocp.test_utils.erase_and_create_empty(path)
    # Checkpoint manager
    options = ocp.CheckpointManagerOptions(create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(path),
        item_names=("state", "epoch", "global_step"),
        options=options,
    )

    # train_loader = data_loader(
    #     rng,
    #     (encode_data["X_train"], encode_data["x_train_cell_type"]),
    #     train_steps_per_epoch,
    #     batch_size,
    # )

    # val_loader = data_loader(
    #     rng,
    #     (encode_data["X_val"], encode_data["x_val_cell_type"]),
    #     val_steps_per_epoch,
    #     batch_size,
    # )

    # NOTE: put back
    # x_data = encode_data["X_train"][:1]
    # y_data = encode_data["x_train_cell_type"][:1]
    # x_val_data = encode_data["X_val"][:1]
    # y_val_data = encode_data["x_val_cell_type"][:1]
    # train_steps_per_epoch = 1
    # val_steps_per_epoch = 1

    for epoch in tqdm(range(num_epochs), disable=not jax.process_index() == 0, desc="Epochs"):
        rng, *rngs = jax.random.split(rng, 3)
        for _ in range(train_steps_per_epoch):
            # NOTE:
            x, y = get_batch(rngs[0], (encode_data["X_train"], encode_data["x_train_cell_type"]))
            # x, y = get_batch(rngs[0], (x_data, y_data))

            x = batch_sharding(x)
            y = batch_sharding(y)
            # x, y = next(train_loader)
            state_dp, loss = train_step_fn(state_dp, rngs[1], x, y, timesteps, d_params)
            global_step += 1

            # NOTE:
            if jax.process_index() == 0:
                if global_step % 10 == 0:
                    wandb.log(
                        {
                            "loss": loss,
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )

        for _ in range(val_steps_per_epoch):
            # NOTE:
            x_val, y_val = get_batch(rngs[0], (encode_data["X_val"], encode_data["x_val_cell_type"]))
            # x_val, y_val = get_batch(rngs[0], (x_val_data, y_val_data))

            x_val = batch_sharding(x_val)
            y_val = batch_sharding(y_val)
            # x_val, y_val = next(val_loader)
            val_loss = val_step_fn(state_dp, rngs[1], x_val, y_val, timesteps, d_params)

        # print(
        #     f"Epoch: {epoch}, Step: {global_step}, val_loss:{val_loss:.4f}, loss: {loss:.4f}"
        # )

        # NOTE: put back
        if jax.process_index() == 0:
            wandb.log(
                {
                    "loss": loss,
                    "val_loss": val_loss,
                    "step": global_step,
                    "epoch": epoch,
                }
            )

        # if (epoch + 1) % 1 == 0:
        #     print(f"Sampling at epoch: {epoch+1}")
        #     # Sampling
        #     rng, sample_rng = jax.random.split(rng)
        #
        #     number_of_samples = 100
        #
        #     cell_num_list = encode_data["cell_types"]
        #
        #     # FIX: Hack for using shard_map with static_argnums
        #     sample_timesteps = jnp.zeros((50,), dtype=jnp.int32)
        #     number_of_samples = 800 // jax.device_count()
        #     number_of_samples = jnp.zeros((number_of_samples,), dtype=jnp.int32)
        #     sample_bs = jnp.zeros((10,), dtype=jnp.int32)
        #     sequence_length = jnp.zeros((200,), dtype=jnp.int32)
        #     for i in cell_num_list:
        #         # samples = create_sample_dp(
        #         #     state=state_dp,
        #         #     rng=sample_rng,
        #         #     timesteps=timesteps,
        #         #     diffusion_params=d_params,
        #         #     cell_types=encode_data["cell_types"],
        #         #     # conditional_numeric_to_tag=encode_data["numeric_to_tag"],
        #         #     number_of_samples=100,
        #         #     group_number=i,
        #         #     # cond_weight_to_metric=0.0,
        #         #     cond_weight_to_metric=1,
        #         #     # save_datafame=False,
        #         # )
        #         i = jnp.zeros((i,), dtype=jnp.int32)
        #         samples = create_sample_dp(
        #             state_dp,
        #             sample_rng,
        #             sample_timesteps,
        #             d_params,
        #             jnp.array(cell_num_list),
        #             number_of_samples,
        #             sample_bs,
        #             sequence_length,
        #             i,
        #             1,
        #         )
        #         print("Sample Success")
        #
        #         if jax.process_index() == 0:
        #             collected_samples = multihost_utils.process_allgather(samples)
        #             sequences = [
        #                 convert_to_seq_jax(x, ["A", "C", "G", "T"])
        #                 for x in collected_samples
        #             ]
        #             sequences = jax.device_get(sequences)
        #             print(
        #                 f"Saving sequences for {encode_data['numeric_to_tag'][i.shape[0]]}"
        #             )
        #             with open(
        #                 f"./{encode_data['numeric_to_tag'][i.shape[0]]}.txt", "w"
        #             ) as f:
        #                 f.write("\n".join(sequences))

    # NOTE:
    if jax.process_index() == 0:
        wandb.finish()
        print("Finished training")


if __name__ == "__main__":
    train()
