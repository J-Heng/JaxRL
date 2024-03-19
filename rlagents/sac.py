"""Implementations of algorithms for continuous control."""
import pdb
from functools import partial
from typing import Optional, Sequence, Any
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
import flax.linen as nn
from utils.networks import MLP, Ensemble, StateActionValue, ValueCritic
from utils import policy as policy

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


class SAC(struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    value: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    sampled_backup: bool = struct.field(pytree_node=False)
    iql_expectile: float = struct.field(pytree_node=False)
    iql_temp: float = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               seed: int,
               observation_space: gym.Space,
               action_space: gym.Space,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               value_lr: float = 3e-4,
               temp_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256),
               discount: float = 0.99,
               tau: float = 0.005,
               num_qs: int = 2,
               num_min_qs: Optional[int] = None,
               critic_dropout_rate: Optional[float] = None,
               critic_layer_norm: bool = False,
               init_temperature: float = 0.1,
               sampled_backup: bool = True,
               iql_expectile: float = 0.7,
               iql_temp: float = 3.0,
               log_std_min: float = -20,
               state_dependent_std: bool = True,
               tanh_squash: bool = True):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key_off, critic_key, value_key, temp_key = jax.random.split(rng, 6)

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()
        target_entropy = -action_dim / 2

        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_min=log_std_min,
                                            state_dependent_std=state_dependent_std,
                                            tanh_squash=tanh_squash)
        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr))

        c_base_cls = partial(MLP,
                             hidden_dims=hidden_dims,
                             dropout_rate=critic_dropout_rate,
                             use_layer_norm=critic_layer_norm)

        c_cls = partial(StateActionValue, base_cls=c_base_cls)
        c_def = Ensemble(c_cls, num=num_qs)
        c_params = c_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(apply_fn=c_def.apply,
                                   params=c_params,
                                   tx=optax.adam(learning_rate=critic_lr))

        target_critic_def = Ensemble(c_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(apply_fn=target_critic_def.apply,
                                          params=c_params,
                                          tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        value_def = ValueCritic(hidden_dims)
        value_params = value_def.init(value_key, observations)['params']
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=optax.adam(learning_rate=value_lr))

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))

        return cls(rng=rng,
                   actor=actor,
                   value=value,
                   critic=critic,
                   target_critic=target_critic,
                   temp=temp,
                   target_entropy=target_entropy,
                   tau=tau,
                   discount=discount,
                   num_qs=num_qs,
                   num_min_qs=num_min_qs,
                   sampled_backup=sampled_backup,
                   iql_expectile=iql_expectile,
                   iql_temp=iql_temp)

    def sample_actions(self, observations, temperature=1.0):
        actions, rng = _sample_actions(self.rng, self.actor, observations, temperature)
        if temperature == 1.0:
            return np.asarray(actions), self.replace(rng=rng)
        else:
            return np.asarray(actions)

    # iql update
    @partial(jax.jit)
    def update_offline(self, batch):
        new_agent = self
        new_agent, value_info = new_agent.update_value(batch)
        new_agent, actor_info = new_agent.update_actor_offline(batch)
        new_agent, critic_info = new_agent.update_critic_offline(batch)
        return new_agent, {**value_info, **actor_info, **critic_info}

    def update_value(self, batch):
        q = self.target_critic.apply_fn({'params': self.target_critic.params}, batch['observations'],
                                        batch['actions']).min(axis=0)

        def value_loss_fn(value_params):
            v = self.value.apply_fn({'params': value_params}, batch["observations"])
            value_loss = (jnp.where((q - v) > 0, self.iql_expectile, (1 - self.iql_expectile)) * ((q - v) ** 2)).mean()
            return value_loss, {'value_loss': value_loss, 'v': v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(self.value.params)
        value = self.value.apply_gradients(grads=grads)
        return self.replace(value=value), info

    def update_critic_offline(self, batch):
        next_v = self.value.apply_fn({'params': self.value.params}, batch["next_observations"])
        target_q = batch["rewards"] + self.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {'critic_loss': critic_loss}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic), info

    def update_actor_offline(self, batch):
        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            v = self.value.apply_fn({'params': self.value.params}, batch["observations"])
            qs = self.target_critic.apply_fn({'params': self.target_critic.params},
                                             batch["observations"], batch["actions"])
            q = qs.min(axis=0)
            exp_a = jnp.exp((q - v) * self.iql_temp)
            exp_a = jnp.minimum(exp_a, 100.0)
            log_probs = dist.log_prob(batch["actions"])
            actor_loss = - (exp_a * log_probs).mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor), actor_info

    # sac update
    @partial(jax.jit, static_argnames='utd')
    def update(self, batch, utd):
        new_agent = self
        for i in range(utd):
            def slice(x):
                assert x.shape[0] % utd == 0
                batch_size = x.shape[0] // utd
                return x[batch_size * i:batch_size * (i + 1)]
            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)
        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temp(actor_info['actor_logprob'])
        return new_agent, {**temp_info, **actor_info, **critic_info}

    def update_critic(self, batch):
        dist = self.actor.apply_fn({'params': self.actor.params}, batch['next_observations'])
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)
        key2, rng = jax.random.split(rng)

        if self.num_min_qs is None:
            target_params = self.target_critic.params
        else:
            all_indx = jnp.arange(0, self.num_qs)
            rng, key = jax.random.split(rng)
            indx = jax.random.choice(key, a=all_indx, shape=(self.num_min_qs,), replace=False)
            target_params = jax.tree_util.tree_map(lambda param: param[indx], self.target_critic.params)

        next_qs = self.target_critic.apply_fn({'params': target_params}, batch['next_observations'], next_actions)

        if self.sampled_backup:
            next_log_probs = dist.log_prob(next_actions)
            target_q = batch['rewards'] + self.discount * batch['masks'] * \
                       (next_qs.min(axis=0) - self.temp.apply_fn({'params': self.temp.params}) * next_log_probs)
        else:
            target_q = batch['rewards'] + self.discount * batch['masks'] * next_qs.mean(axis=0)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {'critic_loss': critic_loss}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic), info

    def update_actor(self, batch):
        key, rng = jax.random.split(self.rng)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn({'params': self.critic.params}, batch["observations"],
                                      actions)  # iql用的是target_critic
            q = qs.mean(axis=0)
            actor_loss = (log_probs * self.temp.apply_fn({'params': self.temp.params}) - q).mean()
            return actor_loss, {'actor_loss': actor_loss, 'actor_logprob': -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor, rng=rng), actor_info

    def update_temp(self, entropy):
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({'params': temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {'temp_loss': temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info