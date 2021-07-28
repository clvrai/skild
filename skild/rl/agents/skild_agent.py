import torch
import numpy as np

from spirl.utils.general_utils import ParamDict, ConstantSchedule, AttrDict
from spirl.utils.pytorch_utils import TensorModule, check_shape, ar2ten
from skild.rl.agents.gail_agent import GAILActionPriorSACAgent


class SkiLDAgent(GAILActionPriorSACAgent):
    """Implements the SkiLD algorithm."""
    def __init__(self, *args, **kwargs):
        GAILActionPriorSACAgent.__init__(self, *args, **kwargs)
        self._posterior_target_divergence = self._hp.tdq_schedule(self._hp.tdq_schedule_params)

        # define posterior divergence multiplier alpha_q
        if self._hp.fixed_alpha_q is not None:
            self._log_alpha_q = TensorModule(np.log(self._hp.fixed_alpha_q)
                                             * torch.ones(1, requires_grad=False, device=self._hp.device))
        else:
            self._log_alpha_q = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
            self.alpha_q_opt = self._get_optimizer(self._hp.optimizer, self._log_alpha_q, self._hp.alpha_lr)

    def _default_hparams(self):
        return GAILActionPriorSACAgent._default_hparams(self).overwrite(ParamDict({
            'tdq_schedule': ConstantSchedule,  # schedule used for posterior target divergence param
            'tdq_schedule_params': AttrDict(   # parameters for posterior target divergence schedule
                p = 1.,
            ),
            'action_cond_discriminator': False,     # if True, conditions discriminator on actions
            'fixed_alpha_q': None,
        }))

    def update(self, experience_batch):
        info = GAILActionPriorSACAgent.update(self, experience_batch)
        info.posterior_target_divergence = self._posterior_target_divergence(self.schedule_steps)
        return info

    def _update_alpha(self, experience_batch, policy_output):
        # update alpha_q
        if self._hp.fixed_alpha_q is None:
            self.alpha_q_loss = (self._compute_alpha_q_loss(policy_output) * experience_batch.p_demo.detach()).mean()
            self._perform_update(self.alpha_q_loss, self.alpha_q_opt, self._log_alpha_q)
        else:
            self.alpha_q_loss = 0.

        # update alpha
        alpha_loss = (self._compute_alpha_loss(policy_output) * (1-experience_batch.p_demo).detach()).mean()
        self._perform_update(alpha_loss, self.alpha_opt, self._log_alpha)
        return alpha_loss

    def _compute_alpha_q_loss(self, policy_output):
        return self.alpha_q * (self._posterior_target_divergence(self.schedule_steps)
                               - policy_output.posterior_divergence).detach()

    def _compute_alpha_loss(self, policy_output):
        self._update_steps += 1
        return self.alpha * (self._target_divergence(self.schedule_steps) - policy_output.prior_divergence).detach()

    def _compute_policy_loss(self, experience_batch, policy_output):
        q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
                                      for critic in self.critics])
        weighted_divergence = self.alpha * policy_output.prior_divergence[:, None] \
                                    * (1 - experience_batch.p_demo[:, None]) \
                            + self.alpha_q * policy_output.posterior_divergence[:, None] \
                                    * experience_batch.p_demo[:, None]
        policy_loss = -1 * q_est + weighted_divergence
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                             for critic_target in self.critic_targets])
        weighted_divergence = self.alpha * policy_output.prior_divergence[:, None] \
                                    * (1 - experience_batch.p_demo[:, None]) \
                            + self.alpha_q * policy_output.posterior_divergence[:, None] \
                                    * experience_batch.p_demo[:, None]
        next_val = (q_next - weighted_divergence)
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1)

    def _aux_info(self, experience_batch, policy_output):
        aux_info = GAILActionPriorSACAgent._aux_info(self, experience_batch, policy_output)
        aux_info.update(AttrDict(
            prior_divergence=(policy_output.prior_divergence[experience_batch.p_demo < 0.5]).mean(),
            posterior_divergence=(policy_output.posterior_divergence[experience_batch.p_demo > 0.5]).mean(),
            alpha_q_loss=self.alpha_q_loss,
            alpha_q=self.alpha_q,
            p_demo=experience_batch.p_demo.mean(),
        ))
        aux_info.update(AttrDict(       # log all reward components
            env_reward=self._hp.reward_scale * self._lambda_gail(self.schedule_steps)
                                    * experience_batch.orig_reward.mean(),
            gail_reward=self._hp.reward_scale * (1-self._lambda_gail(self.schedule_steps))
                                    * torch.stack(experience_batch.discr_reward).mean(),
            prior_reward=self.alpha * (policy_output.prior_divergence * (1 - experience_batch.p_demo)).mean(),
            posterior_reward=self.alpha_q * (policy_output.posterior_divergence * experience_batch.p_demo).mean(),
        ))
        return aux_info

    def _run_discriminator(self, experience_batch, policy_output=None):
        # optionally unflatten observation (in case we have image inputs)
        if self._hp.action_cond_discriminator and policy_output is None:
            # first call -- before policy was called
            return torch.zeros_like(experience_batch.observation[:, 0])
        if hasattr(self.policy.net, "unflatten_obs"):
            discriminator_input = self.discriminator.filter_input(self.policy.net.unflatten_obs(
                ar2ten(experience_batch.observation, device=self.device)).prior_obs)
        else:
            discriminator_input = ar2ten(experience_batch.observation, device=self.device)
        if self._hp.action_cond_discriminator:
            discriminator_input = torch.cat((discriminator_input, policy_output.action), dim=-1)
        return self.discriminator(discriminator_input)[..., 0]

    def _update_experience(self, experience_batch, policy_outputs):
        """Run discriminator with action input."""
        if not self._hp.action_cond_discriminator:
            return super()._update_experience(experience_batch, policy_outputs)
        return self._preprocess_experience(experience_batch, policy_outputs)

    def state_dict(self, *args, **kwargs):
        d = GAILActionPriorSACAgent.state_dict(self)
        if hasattr(self, 'alpha_q_opt'):
            d['alpha_q_opt'] = self.alpha_q_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        if 'alpha_q_opt' in state_dict:
            self.alpha_q_opt.load_state_dict(state_dict.pop('alpha_q_opt'))
        GAILActionPriorSACAgent.load_state_dict(self, state_dict, *args, **kwargs)

    @property
    def alpha_q(self):
        if self._hp.alpha_min is not None:
            return torch.clamp(self._log_alpha_q().exp(), min=self._hp.alpha_min)
        return self._log_alpha_q().exp()
