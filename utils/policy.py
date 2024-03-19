from typing import Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from utils.networks import MLP

default_init = nn.initializers.xavier_uniform

tfd = tfp.distributions
tfb = tfp.bijectors


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -5
    tanh_squash: bool = True
    deterministic: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False):
        outputs = MLP(self.hidden_dims, dropout_rate=self.dropout_rate)(observations, training=training)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.deterministic:
            return nn.tanh(means)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim, ))

        log_stds = jnp.clip(log_stds, self.log_std_min, 2)

        return tfd.MultivariateNormalDiag(loc=nn.tanh(means), scale_diag=jnp.exp(log_stds) * temperature)


