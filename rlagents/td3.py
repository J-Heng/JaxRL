import gym
from functools import partial
from typing import Sequence, Any
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils import policy as policy
from utils.networks import MLP, Ensemble, StateActionValue

PRNGKey = Any


@partial(jax.jit)
def _sample_actions(actor, observations):
    return actor.apply_fn({'params': actor.params}, observations)


class TD3(struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    tau: float
    discount: float

    @classmethod
    def create(cls,
               seed: int,
               observation_space: gym.Space,
               action_space: gym.Space,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256),
               discount: float = 0.99,
               tau: float = 0.005,
               critic_layer_norm: bool = False,
               num_qs: int = 2,
               num_min_qs: int = None):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        actor_def = policy.NormalTanhPolicy(hidden_dims, action_dim, deterministic=True)
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

        return cls(rng=rng,
                   actor=actor,
                   critic=critic,
                   target_critic=target_critic,
                   tau=tau,
                   discount=discount)

    def sample_actions(self, observations, temperature=1.0):
        actions = np.asarray(_sample_actions(self.actor, observations))
        if temperature == 1.0:
            return actions, self.replace(rng=self.rng)
        else:
            return actions

    def update_target(self):
        target_critic_params = optax.incremental_update(self.critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(target_critic=target_critic)

    # td3bc update
    @partial(jax.jit)
    def update_q(self, batch):
        new_agent = self
        new_agent, critic_info = new_agent.update_critic(batch)
        new_agent = new_agent.update_target()
        return new_agent, critic_info

    @partial(jax.jit)
    def update_pi_offline(self, batch, bc=2.5):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor_offline(batch, bc)
        return new_agent, actor_info

    def update_critic(self, batch):
        rng = self.rng
        key, rng = jax.random.split(rng)
        noise = jnp.clip(jax.random.normal(key, batch['actions'].shape) * 0.2, -0.5, 0.5)
        next_actions = self.actor.apply_fn({'params': self.actor.params}, batch['next_observations'])
        next_actions = jnp.clip(next_actions + noise, -1, 1)
        qs = self.target_critic.apply_fn({'params': self.target_critic.params},
                                         batch['next_observations'], next_actions)
        target_q = batch['rewards'] + self.discount * batch['masks'] * qs.min(axis=0)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {'critic_loss': critic_loss}

        grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        return self.replace(critic=critic, rng=rng), critic_info

    def update_actor_offline(self, batch, bc):
        def actor_loss_fn(actor_params):
            actions = self.actor.apply_fn({'params': actor_params}, batch["observations"])
            qs = self.critic.apply_fn({'params': self.critic.params}, batch["observations"], actions)
            lmbda = jax.lax.stop_gradient(bc / jnp.abs(qs[1]).mean())
            actor_loss = -(lmbda * qs[1]).mean() + jnp.mean(jnp.square(actions - batch['actions']))
            return actor_loss, {'actor_loss': actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor), actor_info

    # td3 update
    @partial(jax.jit)
    def update_pi(self, batch):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info

    def update_actor(self, batch):
        def actor_loss_fn(actor_params):
            actions = self.actor.apply_fn({'params': actor_params}, batch["observations"])
            qs = self.critic.apply_fn({'params': self.critic.params}, batch['observations'], actions)
            actor_loss = - qs[1].mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor), actor_info
