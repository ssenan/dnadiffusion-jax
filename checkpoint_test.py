import os

import jax
import numpy as np
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils


def test_checkpointing():
    jax.distributed.initialize()

    # path = ocp.test_utils.erase_and_create_empty("/tmp/checkpoint_manager_sharded")
    path = os.path.join("/tmp/checkpoint_manager_sharded", "test_checkpointing")
    multihost_utils.sync_global_devices("create_directory")

    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("model",)),
        jax.sharding.PartitionSpec(
            "model",
        ),
    )
    create_sharded_array = lambda x: jax.device_put(x, sharding)
    train_state = {
        "a": np.arange(16),
        "b": np.ones(16),
    }
    train_state = jax.tree_util.tree_map(create_sharded_array, train_state)
    jax.tree_util.tree_map(lambda x: x.sharding, train_state)
    num_steps = 10
    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(path, options=options)

    @jax.jit
    def train_fn(state):
        return jax.tree_util.tree_map(lambda x: x + 1, state)

    for step in range(num_steps):
        train_state = train_fn(train_state)
        mngr.save(step, args=ocp.args.StandardSave(train_state))
    mngr.wait_until_finished()
    print("checkpointing done")


if __name__ == "__main__":
    test_checkpointing()
