import gym
from functools import partial
from typing import Sequence, Any
from flax import struct
from flax.training.train_state import TrainState
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils import policy as policy
from utils.networks import MLP, Ensemble, StateActionValue


PRNGKey = Any


class Temperature(nn.Module):
    initial_temperature: float = 1.0
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param("log_temp", init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

@partial(jax.jit)
def _sample_actions(rng, actor, observations, temperature):
    dist = actor.apply_fn({'params': actor.params}, observations, temperature)
    key, rng = jax.random.split(rng)
    return dist.sample(seed=key), rng


class CQL(struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    discount: float
    target_entropy: float
    min_q_weight: float

    @classmethod
    def create(cls,
               seed: int,
               observation_space: gym.Space,
               action_space: gym.Space,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               temp_lr: float = 3e-4,
               init_temperature: float = 0.0,
               log_std_min: float = -5,
               hidden_dims: Sequence[int] = (256, 256),
               discount: float = 0.99,
               tau: float = 0.005,
               critic_layer_norm: bool = False,
               num_qs: int = 2,
               num_min_qs: int = None,
               min_q_weight: float = 5.0):

        action_dim = action_space.shape[-1]
        target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        observations = observation_space.sample()
        actions = action_space.sample()

        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_min=log_std_min,
                                            state_dependent_std=False,
                                            tanh_squash=False)
        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr))

        critic_base_cls = partial(MLP,
                                  hidden_dims=hidden_dims,
                                  dropout_rate=None,
                                  use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr))

        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(apply_fn=target_critic_def.apply,
                                          params=critic_params,
                                          tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))

        return cls(rng=rng,
                   actor=actor,
                   critic=critic,
                   target_critic=target_critic,
                   temp=temp,
                   target_entropy=target_entropy,
                   tau=tau,
                   discount=discount,
                   min_q_weight=min_q_weight)

    def sample_actions(self, observations, temperature=1.0):
        actions, rng = _sample_actions(self.rng, self.actor, observations, temperature)
        if temperature == 1.0:
            return np.asarray(actions), self.replace(rng=rng)
        else:
            return np.asarray(actions)

    def update_temp(self, entropy):
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({'params': temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {'temp_loss': temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    # cql update
    @partial(jax.jit)
    def update_offline(self, batch):
        new_agent = self
        new_agent, critic_info = new_agent.update_critic(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, temp_info = new_agent.update_temp(actor_info['actor_logprob'])
        return new_agent, {**temp_info, **actor_info, **critic_info}

    def update_critic(self, batch):
        dist = self.actor.apply_fn({'params': self.actor.params}, batch['next_observations'])
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        target_params = self.target_critic.params
        next_qs = self.target_critic.apply_fn({'params': target_params}, batch['next_observations'], next_actions)
        target_q = batch['rewards'] + self.discount * batch['masks'] * next_qs.min(axis=0)

        def critic_loss_fn(critic_params):

            def forward_policy(actor, observations, rng):
                dist = actor.apply_fn({'params': actor.params}, observations)
                key, rng = jax.random.split(rng)
                actions = dist.sample(seed=key, sample_shape=10)
                log_probs = dist.log_prob(actions)
                return actions.transpose(1, 0, 2), log_probs.transpose(1, 0)

            qs = self.critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            critic_loss = ((qs - target_q) ** 2).mean()
            # cql
            cql_random_actions = jax.random.uniform(key, shape=(256, 10, batch['actions'].shape[-1]), minval=-1.0, maxval=1.0)
            cql_current_actions, cql_current_log_pis = forward_policy(self.actor, batch['observations'], rng)
            cql_next_actions, cql_next_log_pis = forward_policy(self.actor, batch['next_observations'], rng)
            repeat_observations = jnp.repeat(jnp.expand_dims(batch['observations'], axis=1), repeats=10, axis=1)
            cql_q_rand = self.critic.apply_fn({'params': critic_params}, repeat_observations, cql_random_actions)
            cql_q_current_actions = self.critic.apply_fn({'params': critic_params}, repeat_observations, cql_current_actions)
            cql_q_next_actions = self.critic.apply_fn({'params': critic_params}, repeat_observations, cql_next_actions)
            random_density = np.log(0.5 ** batch['actions'].shape[-1])
            cql_cat_q = jnp.concatenate([cql_q_rand - random_density, cql_q_next_actions - cql_next_log_pis,
                                         cql_q_current_actions - cql_current_log_pis], axis=2)
            cql_qf_ood = jax.scipy.special.logsumexp(cql_cat_q, axis=2)
            critic_loss = critic_loss + (cql_qf_ood - qs).mean() * self.min_q_weight
            return critic_loss, {'critic_loss': critic_loss}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_actor(self, batch):
        key, rng = jax.random.split(self.rng)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn({'params': self.critic.params}, batch["observations"], actions)
            q = qs.min(axis=0)
            actor_loss = (log_probs * self.temp.apply_fn({'params': self.temp.params}) - q).mean()
            return actor_loss, {'actor_loss': actor_loss, 'actor_logprob': -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor, rng=rng), actor_info