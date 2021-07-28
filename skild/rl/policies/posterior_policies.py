import torch

from spirl.utils.pytorch_utils import no_batchnorm_update
from spirl.utils.general_utils import ParamDict, AttrDict
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.components.agent import BaseAgent


class LearnedPPPolicy(LearnedPriorAugmentedPIPolicy):
    """Computes both learned prior and posterior distribution."""
    def __init__(self, *args, **kwargs):
        LearnedPriorAugmentedPIPolicy.__init__(self, *args, **kwargs)
        self.posterior_net = self._hp.posterior_model(self._hp.posterior_model_params, None)
        BaseAgent.load_model_weights(self.posterior_net,
                                     self._hp.posterior_model_checkpoint,
                                     self._hp.posterior_model_epoch)

    def _default_hparams(self):
        return LearnedPriorAugmentedPIPolicy._default_hparams(self).overwrite(ParamDict({
            'posterior_model': None,                # posterior model class
            'posterior_model_params': None,         # parameters for the posterior model
            'posterior_model_checkpoint': None,     # checkpoint path of the posterior model
            'posterior_model_epoch': 'latest',      # epoch that checkpoint should be loaded for (defaults to latest)
        }))

    def forward(self, obs):
        policy_output = LearnedPriorAugmentedPIPolicy.forward(self, obs)
        if not self._rollout_mode:
            raw_posterior_divergence, policy_output.posterior_dist = \
                self._compute_posterior_divergence(policy_output, obs)
            policy_output.posterior_divergence = self.clamp_divergence(raw_posterior_divergence)
        return policy_output

    def _compute_posterior_divergence(self, policy_output, obs):
        with no_batchnorm_update(self.posterior_net):
            posterior_dist = self.posterior_net.compute_learned_prior(obs, first_only=True).detach()
            if self._hp.analytic_KL:
                return self._analytic_divergence(policy_output, posterior_dist), posterior_dist
            return self._mc_divergence(policy_output, posterior_dist), posterior_dist


class ACLearnedPPPolicy(LearnedPPPolicy):
    """LearnedPPPolicy for case with separate prior obs --> uses prior observation as input only."""
    def forward(self, obs):
        if obs.shape[0] == 1:
            return super().forward(self.net.unflatten_obs(obs).prior_obs)   # use policy_net or batch_size 1 inputs
        return super().forward(self.prior_net.unflatten_obs(obs).prior_obs)
