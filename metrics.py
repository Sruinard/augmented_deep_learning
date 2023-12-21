from clu import metrics, periodic_actions
import flax
import jax.numpy as jnp


@flax.struct.dataclass
class Precision(metrics.Metric):
    """Computes the precision from model outputs `logits` and `labels`."""

    true_positives: jnp.array
    pred_positives: jnp.array

    @classmethod
    def from_model_output(
        cls, *, logits: jnp.array, labels: jnp.array, **_
    ) -> metrics.Metric:
        assert logits.shape[-1] == 2, "Expected binary logits."
        preds = logits.argmax(axis=-1)
        return cls(
            true_positives=((preds == 1) & (labels == 1)).sum(),
            pred_positives=(preds == 1).sum(),
        )

    def merge(self, other: metrics.Metric) -> metrics.Metric:
        # Note that for precision we cannot average metric values because the
        # denominator of the metric value is pred_positives and not every batch of
        # examples has the same number of pred_positives (as opposed to e.g.
        # accuracy where every batch has the same number of)
        return type(self)(
            true_positives=self.true_positives + other.true_positives,
            pred_positives=self.pred_positives + other.pred_positives,
        )

    def compute(self):
        return self.true_positives / self.pred_positives

    def empty(self):
        return type(self)(true_positives=0, pred_positives=0)

    @classmethod
    def empty(cls) -> "Precision":
        return cls(true_positives=0, pred_positives=0)


@flax.struct.dataclass  # <-- required for JAX transformations
class MetricCollection(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy
    precision: Precision


class TensorboardCallback(periodic_actions.PeriodicCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def write_metrics(step: int, t: float, *, writer, train_metrics, eval_metrics):
        writer.write_scalars(
            step, {f"train/{k}": v for k, v in train_metrics.compute().items()}
        )
        writer.write_scalars(
            step, {f"eval/{k}": v for k, v in eval_metrics.compute().items()}
        )


class ReportProgress(periodic_actions.ReportProgress):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, step, t=None, **kwargs):
        return super().__call__(step, t)

    def _apply(self, step, t, **kwargs):
        super()._apply(step, t)


class Profile(periodic_actions.Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, step, t=None, **kwargs):
        return super().__call__(step, t)

    def _apply(self, step, t, **kwargs):
        super()._apply(step, t)
