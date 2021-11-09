import numpy as np
import torch

from spirl.utils.general_utils import ParamDict, AttrDict, map_dict, ConstantSchedule
from spirl.utils.pytorch_utils import map2torch, map2np, ten2ar, ar2ten, avg_grad_norm, update_optimizer_lr
from spirl.rl.agents.ac_agent import ACAgent
from spirl.rl.components.normalization import DummyNormalizer


class PPOAgent(ACAgent):
    """Implements PPO algorithm."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # build old actor policy
        self.old_policy = self._hp.policy(self._hp.policy_params)

        # build critic and critic optimizer
        self.critic = self._hp.critic(self._hp.critic_params)
        self.critic_opt = self._get_optimizer(self._hp.optimizer, self.critic, self._hp.critic_lr)
        self._lr = self._hp.lr_schedule(self._hp.lr_schedule_params)

        # build trajectory buffer and reward normalizer
        self.trajectory_buffer = self._hp.buffer(self._hp.buffer_params)
        self._reward_normalizer = self._hp.reward_normalizer(self._hp.reward_normalizer_params)

        self._update_steps = 0
        self._env_steps = 0

    def _default_hparams(self):
        default_dict = ParamDict({
            'critic': None,         # critic class
            'critic_params': None,  # parameters for the critic class
            'critic_lr': 3e-4,      # learning rate for critic update
            'buffer': None,         # trajectory buffer class
            'buffer_params': None,  # parameters for trajectory buffer
            'clip_ratio': 0.2,      # policy update clipping value
            'entropy_coefficient': 0.0,             # coefficient for weighting of entropy loss
            'gae_lambda': 0.95,                     # GAE lambda coefficient
            'target_network_update_factor': 1.0,    # always overwrite old actor policy completely
            'gradient_clip': 0.5,                   # overwrite default to cligrad norm at 0.5
            'clip_value_loss': False,               # if True, applies clipping to value loss
            'reward_normalizer': DummyNormalizer,   # normalizer for rewards
            'reward_normalizer_params': {},         # optional parameters for reward normalizer
            'lr_schedule': ConstantSchedule,        # schedule for learning rate
            'lr_schedule_params': AttrDict(p=3e-4), # parameters for learning rate schedule
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        """Updates actor and critic."""
        # normalize experience batch
        experience_batch = self._normalize_batch(experience_batch)

        # perform any auxiliary updates
        self.add_aux_experience(experience_batch)
        self._aux_updates()

        # prepare experience batch for policy update
        self.add_experience(experience_batch)
        self._env_steps += self.trajectory_buffer.size
        self._update_lr()

        # copy actor weights
        self._soft_update_target_network(self.old_policy, self.policy)

        for _ in range(self._hp.update_iterations):
            # sample update sample
            experience_batch = self.trajectory_buffer.sample(n_samples=self._hp.batch_size)
            experience_batch = map2torch(experience_batch, device=self.device)

            # compute policy loss
            policy_loss, entropy, pi_ratio = self._compute_policy_loss(experience_batch)

            # compute critic loss
            critic_loss = self._compute_critic_loss(experience_batch)

            # update networks & learning rate
            self._update_steps += 1
            self._perform_update(policy_loss, self.policy_opt, self.policy)
            self._perform_update(critic_loss, self.critic_opt, self.critic)

            # log info
            info = AttrDict(
                policy_loss=policy_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                pi_ratio=pi_ratio.mean(),
                lr=self._lr(self.schedule_steps),
            )
            if self._update_steps % 100 == 0:
                info.update(AttrDict(       # gradient norms
                    policy_grad_norm=avg_grad_norm(self.policy),
                    critic_grad_norm=avg_grad_norm(self.critic),
                ))
        info = map_dict(ten2ar, info)
        return info

    def _normalize_batch(self, experience_batch):
        self._obs_normalizer.update(experience_batch.observation)
        self._reward_normalizer.update(experience_batch.reward)
        experience_batch.observation = self._obs_normalizer(experience_batch.observation)
        experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
        experience_batch.reward = self._reward_normalizer(experience_batch.reward)
        return experience_batch

    def add_experience(self, experience_batch):
        experience_batch = self._preprocess_experience(map2torch(experience_batch, self.device))
        experience_batch = self._compute_advantage(map2np(experience_batch))
        self.trajectory_buffer.reset()
        self.trajectory_buffer.append(experience_batch)

    def _compute_advantage(self, experience_batch):
        """Computes advantage and return of input trajectories using critic."""
        n_steps = len(experience_batch.observation) - 1

        # compute estimated value
        with torch.no_grad():
            value = ten2ar(self.critic(
                ar2ten(np.array(experience_batch.observation, dtype=np.float32), device=self.device)).q).squeeze(-1)

        # recursively compute returns and advantage
        advantage = np.empty_like(value[:-1])
        last_adv = 0
        for t in reversed(range(n_steps)):
            advantage[t] = experience_batch.reward[t] \
                           + (1 - experience_batch.done[t]) * self._hp.discount_factor * value[t+1] \
                           - value[t] \
                           + self._hp.discount_factor * self._hp.gae_lambda * (1 - experience_batch.done[t]) * last_adv
            last_adv = advantage[t]

        # compute returns and normalized advantage
        returns = advantage + value[:-1]
        norm_advantage = (advantage - advantage.mean()) / advantage.std()

        # remove final transitions for which we don't have advantages + add computed adv to experience batch
        for key in experience_batch:
            experience_batch[key] = experience_batch[key][:advantage.shape[0]]
        experience_batch.returns = [r for r in returns]
        experience_batch.advantage = [a for a in norm_advantage]
        experience_batch.value_pred = [v for v in value[:-1]]

        return experience_batch

    def _compute_policy_loss(self, experience_batch):
        """Computes policy update loss."""
        # run actors
        policy_output = self.policy(experience_batch.observation)
        old_policy_output = self.old_policy(experience_batch.observation)
        log_pi, old_log_pi = policy_output.dist.log_prob(experience_batch.action), \
                             old_policy_output.dist.log_prob(experience_batch.action)

        # compute actor loss
        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = ratio * experience_batch.advantage
        surr2 = torch.clamp(ratio, 1.0 - self._hp.clip_ratio, 1.0 + self._hp.clip_ratio) * experience_batch.advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # compute entropy loss
        entropy_loss = -1 * policy_output.dist.entropy().mean()

        return actor_loss + self._hp.entropy_coefficient * entropy_loss, -1 * entropy_loss, ratio

    def _compute_critic_loss(self, experience_batch):
        value = self.critic(experience_batch.observation).q.squeeze(-1)
        if not self._hp.clip_value_loss:
            return 0.5 * (experience_batch.returns - value).pow(2).mean()
        else:
            value_clipped = experience_batch.value_pred + \
                                 (value - experience_batch.value_pred).clamp(-self._hp.clip_ratio, self._hp.clip_ratio)
            value_losses = (experience_batch.returns - value).pow(2)
            value_losses_clipped = (value_clipped - experience_batch.returns).pow(2)
            return 0.5 * torch.max(value_losses, value_losses_clipped).mean()

    def _update_lr(self):
        """Updates learning rates with schedule."""
        if not isinstance(self._lr, ConstantSchedule):
            update_optimizer_lr(self.policy_opt, self._lr(self.schedule_steps))
            update_optimizer_lr(self.critic_opt, self._lr(self.schedule_steps))

    @property
    def schedule_steps(self):
        return self._env_steps
