import datetime
import os

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


def train_and_eval(cfg: ml_collections.ConfigDict):
    p = ip.Preprocessor()
    train_ds, val_ds = ip.get_datasets(
        preprocessor=p,
        train_src=cfg.train_src,
        val_src=cfg.val_src,
    )

    x, _ = next(train_ds)

    rng = jax.random.PRNGKey(cfg.seed)
    metric_collection = metrics.MetricCollection.empty()

    writer = metric_writers.create_default_writer(cfg.logdir)
    hooks = [
        # Outputs progress via metric writer (in this case logs & TensorBoard).
        metrics.ReportProgress(
            num_train_steps=cfg.n_steps_per_epoch * cfg.n_epochs,
            every_steps=cfg.n_steps_per_epoch,
            writer=writer,
        ),
        metrics.Profile(logdir=cfg.logdir),
        metrics.TensorboardCallback(
            callback_fn=metrics.TensorboardCallback.write_metrics,
            every_steps=cfg.n_steps_per_epoch,
        ),
    ]

    mngr = create_manager(cfg.checkpoint_dir)
    state = restore_or_create_state(mngr, rng, x.shape)

    global_step = 0
    for _ in range(cfg.n_epochs):
        state, train_metrics, eval_metrics, global_step = run_loop(
            state=state,
            train_ds=train_ds,
            val_ds=val_ds,
            n_train_steps=cfg.n_steps_per_epoch,
            n_eval_steps=cfg.n_steps_per_epoch // 10,
            global_step=global_step,
            hooks=hooks,
            writer=writer,
            mngr=mngr,
            metric_collection=metric_collection,
        )

        print(f"train_metrics: {train_metrics.compute()}")
        print(f"eval_metrics: {eval_metrics.compute()}")

    best_model_state = restore_or_create_state(mngr, rng, x.shape)
    to_saved_model(
        best_model_state,
        p.serving_fn,
        cfg.model_serving_dir,
        etr={"preprocessor": p.norm},
    )
