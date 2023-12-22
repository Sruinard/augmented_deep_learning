import datetime
import os
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
import orbax.checkpoint as ocp
from clu import metric_writers
from flax.training import train_state
from orbax.export import ExportManager, JaxModule, ServingConfig

import input_pipeline as ip
import metrics


class CreditCardFraudModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def init_model(rng, input_shape):
    model = CreditCardFraudModel()
    params = model.init(rng, jnp.ones(input_shape, jnp.float32))
    return model, params


def create_train_state(rng, input_shape, learning_rate=1e-3):
    """Creates initial `TrainState`."""
    model, params = init_model(rng, input_shape)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: train_state.TrainState, x, y, train_metrics):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn(params, x=x)
        loss = optax.sigmoid_binary_cross_entropy(logits, y)
        loss = jnp.mean(loss)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    logits = jnp.concatenate([1 - logits, logits], axis=-1)

    return state, train_metrics.merge(
        metrics.MetricCollection.single_from_model_output(
            loss=loss,
            labels=y.squeeze(),
            logits=logits,
        )
    )


@jax.jit
def eval_step(state: train_state.TrainState, x, y, eval_metrics):
    """Evaluates `state` on `x` and `y`."""
    logits = state.apply_fn(state.params, x=x)
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    loss = jnp.mean(loss)

    logits = jnp.concatenate([1 - logits, logits], axis=-1)

    return eval_metrics.merge(
        metrics.MetricCollection.single_from_model_output(
            loss=loss,
            labels=y.squeeze(),
            logits=logits,
        )
    )


def create_manager(model_dir):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3, save_interval_steps=2, create=True
    )

    mngr = ocp.CheckpointManager(model_dir, ocp.PyTreeCheckpointer(), options=options)
    return mngr


def restore_or_create_state(mngr, rng, input_shape, reinit=False):
    if mngr.latest_step() is None or reinit:
        return create_train_state(rng, input_shape)
    target = {"model": create_train_state(rng, input_shape)}
    restored_state = mngr.restore(mngr.latest_step(), items=target)["model"]
    return restored_state


def to_saved_model(
    state, preprocessing_fn, model_serving_dir, etr=None, model_name="creditcard"
):
    # Construct a JaxModule where JAX->TF conversion happens.
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    jax_module = JaxModule(
        state.params,
        state.apply_fn,
        trainable=False,
        jit_compile=False,
        jax2tf_kwargs={"enable_xla": False},
        input_polymorphic_shape="(b, ...)",
    )
    # Export the JaxModule along with one or more serving configs.
    export_mgr = ExportManager(
        jax_module,
        [
            ServingConfig(
                "serving_default",
                tf_preprocessor=preprocessing_fn,
                # tf_postprocessor=exampe1_postprocess
                extra_trackable_resources=etr,
            ),
        ],
    )
    export_mgr.save(os.path.join(model_serving_dir, model_name, timestamp))


def run_loop(
    state,
    train_ds,
    val_ds,
    n_train_steps,
    n_eval_steps,
    global_step,
    hooks,
    writer,
    mngr,
    metric_collection,
):
    train_metrics = metric_collection.empty()
    eval_metrics = metric_collection.empty()
    for step in range(n_train_steps):
        x, y = next(train_ds)
        state, train_metrics = train_step(state, x, y, train_metrics)

    for step in range(n_eval_steps):
        x, y = next(val_ds)
        eval_metrics = eval_step(state, x, y, eval_metrics)
        for hook in hooks:
            hook(
                global_step,
                writer=writer,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

        mngr.save(step, {"model": state})
        global_step += 1

    return state, train_metrics, eval_metrics, global_step


def train_and_eval(config: ml_collections.ConfigDict):
    path = Path("./models/checkpoints/")
    model_dir = path.absolute()
    p = ip.Preprocessor()
    train_ds, val_ds = ip.get_datasets(
        preprocessor=p,
        train_src="data/example_gen/train.tfrecord",
        val_src="data/example_gen/test.tfrecord",
    )

    x, _ = next(train_ds)
    logdir = "./logs"
    n_epochs = 10
    rng = jax.random.PRNGKey(42)
    n_batches_per_epoch = 1000
    total_steps = n_epochs * n_batches_per_epoch
    input_shape = jnp.shape(x)
    n_train_steps = 1000
    n_eval_staps = 100
    saved_model_dir = "./models/saved_model"

    metric_collection = metrics.MetricCollection.empty()

    writer = metric_writers.create_default_writer(logdir)
    hooks = [
        # Outputs progress via metric writer (in this case logs & TensorBoard).
        metrics.ReportProgress(
            num_train_steps=total_steps, every_steps=n_batches_per_epoch, writer=writer
        ),
        metrics.Profile(logdir=logdir),
        metrics.TensorboardCallback(
            callback_fn=metrics.TensorboardCallback.write_metrics,
            every_steps=n_batches_per_epoch,
        ),
    ]

    mngr = create_manager(model_dir)
    state = restore_or_create_state(mngr, rng, input_shape)

    n_steps_taken = 0
    for epoch in range(10):
        train_metrics = metric_collection.empty()
        eval_metrics = metric_collection.empty()
        for step in range(n_train_steps):
            x, y = next(train_ds)
            state, train_metrics = train_step(state, x, y, train_metrics)

        for step in range(n_eval_staps):
            x, y = next(val_ds)
            eval_metrics = eval_step(state, x, y, eval_metrics)
            for hook in hooks:
                hook(
                    n_steps_taken,
                    writer=writer,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                )

            mngr.save(step, {"model": state})
            n_steps_taken += 1

    print(
        train_metrics.compute(),
        eval_metrics.compute(),
    )
    restored_state = restore_or_create_state(mngr, rng, input_shape)

    eval_m = metric_collection.empty()
    for _ in range(100):
        x, y = next(val_ds)
        eval_m = eval_step(restored_state, x, y, eval_m)

    print(eval_m.compute())

    to_saved_model(
        restored_state, p.serving_fn, saved_model_dir, etr={"preprocessor": p.norm}
    )
