import flax
from flax.training import train_state
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax
import clu
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