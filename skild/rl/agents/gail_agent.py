import copy
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch import autograd

from spirl.utils.general_utils import ParamDict, AttrDict, map_dict, ConstantSchedule
from spirl.utils.pytorch_utils import map2torch, map2np, ten2ar, update_optimizer_lr
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.components.agent import BaseAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from skild.rl.agents.ppo_agent import PPOAgent


class GAILAgent(PPOAgent):
    """Implements GAIL-based agent. Discriminator determines reward, policy update is inherited."""
    EPS = 1e-20     # constant for numerical stability in computing discriminator-based rewards

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gail()

    def _init_gail(self):
        # set up discriminator
        self.discriminator = self._hp.discriminator(self._hp.discriminator_params)
        self.discriminator_opt = self._get_optimizer(self._hp.optimizer, self.discriminator, self._hp.discriminator_lr)
        if self._hp.discriminator_checkpoint is not None:
            BaseAgent.load_model_weights(self.discriminator,
                                         self._hp.discriminator_checkpoint,
                                         self._hp.discriminator_epoch)

        # load demo dataset
        self._hp.expert_data_conf.device = self.device.type
        self._expert_data_loader = self._hp.expert_data_conf.dataset_spec.dataset_class(
            self._hp.expert_data_path, self._hp.expert_data_conf, resolution=self._hp.expert_data_conf.dataset_spec.res,
            phase="train", shuffle=True).get_data_loader(self._hp.batch_size, n_repeat=10000)  # making new iterators is slow, so repeat often
        self._expert_data_iter = iter(self._expert_data_loader)

        # set up trajectory buffer for discriminator training
        self.gail_trajectory_buffer = self._hp.buffer(self._hp.buffer_params) \
            if 'buffer' in self._hp and self._hp.buffer is not None \
            else self._hp.replay(self._hp.replay_params)     # in case we use GAIL w/ SAC
        self.gail_trajectory_buffer.reset()

        # misc
        self._discriminator_update_cycles = 0
        self._lambda_gail = self._hp.lambda_gail_schedule(self._hp.lambda_gail_schedule_params)

        # optionally run BC for policy init
        if self._hp.bc_init_steps > 0:
            self._run_bc_init()

    def _default_hparams(self):
        default_dict = ParamDict({
            'discriminator': None,          # discriminator class
            'discriminator_params': None,   # parameters for the discriminator class
            'discriminator_checkpoint': None,  # checkpoint to load discriminator from
            'discriminator_epoch': 'latest',   # epoch at which to load discriminator weights
            'discriminator_lr': 3e-4,       # learning rate for discriminator update
            'freeze_discriminator': False,  # if True, does not update discriminator
            'expert_data_conf': None,       # data config for expert sequences
            'expert_data_path': None,       # path to expert data sequences
            'reset_buffer': True,           # if True, resets online buffer every update iteration
            'discriminator_updates': 5,     # number of discriminator updates per PPO policy update cycle
            'lambda_gail_schedule': ConstantSchedule,        # schedule for lambda parameter
            'lambda_gail_schedule_params': AttrDict(p=0.0),  # factor for original reward when mixing with GAIL reward
            'grad_penalty_coefficient': 0.0,  # discriminator gradient penalty coefficient
            'entropy_coefficient_gail': 0.0,  # discriminator entropy loss coefficient
            'warmup_cycles': 0,             # number of first calls to update() in which only discriminator gets trained
            'bc_init_steps': 0,             # number of BC steps for policy before GAIL training starts
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        self.gail_info = {}
        if self._lr(self._env_steps) < 1e-10: return {}     # stop running updates if learning rate is decayed to 0
        if self._discriminator_update_cycles < self._hp.warmup_cycles:
            # only train discriminator during warmup, do not update policy
            self._add_experience_discriminator_buffer(experience_batch)
            self._update_discriminator()
            return self.gail_info
        else:
            # after warmup we first update discriminator, then policy (both handled by super().update())
            info = super().update(experience_batch)
            info.update(self.gail_info)
            return info

    def _update_discriminator(self):
        """Performs one training update for the discriminator."""
        if self._hp.freeze_discriminator:
            return      # do not update discriminator if it is frozen

        n_discriminator_updates = self._hp.discriminator_updates if self._hp.discriminator_updates >= 1 else \
                                    int(np.random.rand() < self._hp.discriminator_updates)
        for _ in range(n_discriminator_updates):
            # sample expert and policy data batches
            expert_batch = self._get_expert_batch()
            policy_batch = self.gail_trajectory_buffer.sample(n_samples=self._hp.batch_size)
            policy_batch = map2torch(policy_batch, self._hp.device)

            # run discriminator
            expert_disc_outputs = self.discriminator(self.discriminator._discriminator_input(
                                        AttrDict(states=expert_batch.states,
                                                 actions=expert_batch.actions)))
            policy_disc_outputs = self.discriminator(self.discriminator._discriminator_input(
                                        AttrDict(states=policy_batch.observation[:, None],
                                                 actions=policy_batch.action[:, None])))

            # compute discriminator losses: cross-entropy, entropy and gradient penalty loss
            expert_logits, policy_logits = expert_disc_outputs, policy_disc_outputs
            logits = torch.cat((expert_logits, policy_logits))
            targets = torch.cat((torch.ones_like(expert_logits), torch.zeros_like(policy_logits)))
            discriminator_loss = BCEWithLogitsLoss()(logits, targets)
            discriminator_entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
            discriminator_loss -= self._hp.entropy_coefficient_gail * discriminator_entropy
            discriminator_accuracy = ((torch.sigmoid(logits) > 0.5).float() == targets).float().mean()
            if self._hp.grad_penalty_coefficient > 0:
                grad_penalty_loss = self._hp.grad_penalty_coefficient * self._compute_gradient_penalty(expert_batch,
                                                                                                       policy_batch)
                discriminator_loss += grad_penalty_loss
            discriminator_loss += self._regularization_losses(expert_disc_outputs, policy_disc_outputs)

            # update discriminator
            self._perform_update(discriminator_loss, self.discriminator_opt, self.discriminator)

            # log info
            info = AttrDict(
                discriminator_loss=discriminator_loss,
                discriminator_entropy=discriminator_entropy,
                discriminator_accuracy=discriminator_accuracy,
                discr_real_output=torch.sigmoid(expert_logits).mean(),
                discr_fake_output=torch.sigmoid(policy_logits).mean(),
            )
            info.update(self._get_obs_norm_info())
            if self._hp.grad_penalty_coefficient > 0:
                info.update(AttrDict(grad_penalty_loss=grad_penalty_loss))
            self.gail_info = map_dict(ten2ar, info)
        self._discriminator_update_cycles += 1

    def _add_experience_discriminator_buffer(self, experience_batch):
        """Normalizes experience and adds to discriminator replay buffer."""
        # fill policy trajectories in buffer
        if self._hp.reset_buffer:
            self.gail_trajectory_buffer.reset()
        self.gail_trajectory_buffer.append(map2np(experience_batch))

    def _aux_updates(self):
        """Update discriminator before updating policy & critic."""
        self._update_discriminator()

    def add_aux_experience(self, experience_batch):
        self._add_experience_discriminator_buffer(experience_batch)

    def _get_expert_batch(self):
        try:
            expert_batch = next(self._expert_data_iter)
        except StopIteration:
            self._expert_data_iter = iter(self._expert_data_loader)
            expert_batch = next(self._expert_data_iter)
        expert_batch = map2np(AttrDict(expert_batch))
        expert_batch.states = self._obs_normalizer(expert_batch.states)
        expert_batch = map2torch(expert_batch, device=self.device)
        return expert_batch

    def _preprocess_experience(self, experience_batch, policy_outputs=None):
        """Trains discriminator and then uses it to relabel rewards."""
        assert isinstance(experience_batch.reward[0], torch.Tensor)       # expects tensors as input
        with torch.no_grad():
            if 'orig_reward' not in experience_batch:
                experience_batch.orig_reward = copy.deepcopy(experience_batch.reward)
            experience_batch.discr_reward, experience_batch.p_demo = \
                self._compute_discriminator_reward(experience_batch, policy_outputs)
            experience_batch.reward = [(1 - self._lambda_gail(self.schedule_steps))
                                       * dr + self._lambda_gail(self.schedule_steps) * r \
                    for dr, r in zip(experience_batch.discr_reward, experience_batch.orig_reward)]
            if isinstance(experience_batch.orig_reward, torch.Tensor):
                # merge list into tensor in case input is also tensor not list (during RL update)
                experience_batch.reward = torch.tensor(experience_batch.reward,
                                                       device=experience_batch.orig_reward.device)
            self.gail_info.update({'discriminator_reward': np.mean(map2np(experience_batch.discr_reward)),
                                   'rl_training_reward': np.mean(map2np(experience_batch.reward)),
                                   'lambda_gail': self._lambda_gail(self.schedule_steps),
                                   'buffer_size': self.gail_trajectory_buffer.size,})
        return experience_batch

    def _compute_discriminator_reward(self, experience_batch, unused_policy_outputs):
        """Uses discriminator to compute GAIL reward."""
        logits = self._run_discriminator(experience_batch, unused_policy_outputs)
        D = torch.sigmoid(logits)
        discriminator_reward = (D + self.EPS).log() - (1 - D + self.EPS).log()
        return [r for r in discriminator_reward], D

    def _run_discriminator(self, experience_batch, unused_policy_outputs):
        """Runs discriminator on experience batch [obs, act], returns logits."""
        input_states = torch.stack(experience_batch.observation) if isinstance(experience_batch.observation, list) \
                            else experience_batch.observation
        input_actions = torch.stack(experience_batch.action) if isinstance(experience_batch.action, list) \
                            else experience_batch.action
        discr_output = self.discriminator(self.discriminator._discriminator_input(
            AttrDict(states=input_states[:, None], actions=input_actions[:, None])))
        return discr_output[:, 0]

    def _compute_gradient_penalty(self, expert_batch, policy_batch):
        """Computes mixup gradient penalty for discriminator."""
        # create mixed policy + expert input
        alpha = torch.rand([policy_batch.observation.shape[0], 1], device=policy_batch.observation.device)
        mixup_state = alpha * policy_batch.observation + (1-alpha) * expert_batch.states[:, 0]
        mixup_action = alpha * policy_batch.action + (1-alpha) * expert_batch.actions[:, 0]
        mixup_state.requires_grad = True; mixup_action.requires_grad = True

        # compute discriminator gradients
        disc_output = self.discriminator(mixup_state, mixup_action).q[:, 0]
        grad = torch.cat(autograd.grad(outputs=disc_output,
                                       inputs=[mixup_state, mixup_action],
                                       grad_outputs=torch.ones_like(disc_output),
                                       create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True), dim=-1)

        # compute gradient penalty
        grad_penalty = (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_penalty

    def _regularization_losses(self, *unused_args, **unused_kwargs):
        """Optionally add more regularization losses to discriminator update."""
        return 0.

    def _run_bc_init(self):
        """Performs BC-based policy initialization."""
        self.to(self.device)
        policy_bc_opt = self._get_optimizer(self._hp.optimizer, self.policy, self._hp.policy_lr)
        for step in range(self._hp.bc_init_steps):
            data = self._get_expert_batch()
            policy_output = self.policy(data.states[:, 0])
            loss = -1 * policy_output.dist.log_prob(data.actions[:, 0]).mean()
            self._perform_update(loss, policy_bc_opt, self.policy)
            if step % int(self._hp.bc_init_steps / 100) == 0:
                print("It {}: \tBC loss: {}, \tEntropy: {}"
                      .format(step, loss, policy_output.dist.entropy().mean().data.cpu().numpy()))

    def _update_lr(self):
        super()._update_lr()
        if not isinstance(self._lr, ConstantSchedule):
            update_optimizer_lr(self.discriminator_opt, self._lr(self._env_steps))


class GAILSACAgent(SACAgent, GAILAgent):
    """GAIL agent that optimizes the discriminator reward using SAC."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gail()

    def _default_hparams(self):
        params = SACAgent._default_hparams(self)
        params.update(GAILAgent._default_hparams(self))
        return params

    def update(self, experience_batch):
        if self._discriminator_update_cycles < self._hp.warmup_cycles:
            # only train discriminator during warmup, do not update policy
            self._add_experience_discriminator_buffer(experience_batch)
            self._update_discriminator()
            return self.gail_info
        else:
            # after warmup we first update discriminator, then policy (both handled by super().update())
            info = SACAgent.update(self, experience_batch)
            info.update(self.gail_info)
            return info

    def add_experience(self, experience_batch):
        self._add_experience_discriminator_buffer(experience_batch)
        SACAgent.add_experience(self, experience_batch)

    def _preprocess_experience(self, experience_batch, policy_outputs=None):
        processed_experience = GAILAgent._preprocess_experience(self, experience_batch, policy_outputs)
        if hasattr(self, 'vis_replay_buffer'):
            self.vis_replay_buffer.append(map2np(processed_experience))    # for visualization
        return processed_experience


class GAILActionPriorSACAgent(ActionPriorSACAgent, GAILAgent):
    """GAIL agent that optimizes the discriminator reward using SPiRL."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gail()

    def _default_hparams(self):
        params = ActionPriorSACAgent._default_hparams(self)
        params.update(GAILAgent._default_hparams(self))
        return params

    def update(self, experience_batch):
        self.gail_info = {}
        if self._discriminator_update_cycles < self._hp.warmup_cycles:
            # only train discriminator during warmup, do not update policy
            self._add_experience_discriminator_buffer(experience_batch)
            self._update_discriminator()
            return self.gail_info
        else:
            # after warmup we first update discriminator, then policy (both handled by super().update())
            info = ActionPriorSACAgent.update(self, experience_batch)
            info.update(self.gail_info)
            return info

    def _preprocess_experience(self, experience_batch, policy_outputs=None):
        processed_experience = GAILAgent._preprocess_experience(self, experience_batch, policy_outputs)
        if hasattr(self, 'vis_replay_buffer'):
            self.vis_replay_buffer.append(map2np(processed_experience))    # for visualization
        return processed_experience

    def add_experience(self, experience_batch):
        self._add_experience_discriminator_buffer(experience_batch)
        ActionPriorSACAgent.add_experience(self, experience_batch)
