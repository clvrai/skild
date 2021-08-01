import os
import copy
import torch

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.components.sampler import HierarchicalSampler
from spirl.rl.components.critic import MLPCritic, SplitObsMLPCritic
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.envs.maze import ACRandMaze0S40Env
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.configs.default_data_configs.maze import data_spec

from skild.rl.policies.posterior_policies import LearnedPPPolicy
from skild.models.demo_discriminator import DemoDiscriminator
from skild.rl.agents.skild_agent import SkiLDAgent
from skild.data.maze.src.maze_agents import MazeSkiLDAgent


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': ACRandMaze0S40Env,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 200,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e5,
    'log_output_per_epoch': 1000,
    'n_warmup_steps': 2e3,
}
configuration = AttrDict(configuration)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=128,
)

###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    nz_vae=10,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior/maze/maze_prior"),
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    output_dim=1,
    action_input=True,
    unused_obs_size=ll_model_params.nz_vae,     # ignore HL policy z output in observation for LL critic
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=ll_critic_params,
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    squash_output_dist=True,
    max_action_range=2.,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model=ll_policy_params.policy_model,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    posterior_model=ll_policy_params.policy_model,
    posterior_model_params=copy.deepcopy(ll_policy_params.policy_model_params),
    posterior_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_posterior/maze/maze_post"),
)
hl_policy_params.posterior_model_params.batch_size = base_agent_params.batch_size

hl_policy_params.policy_model = ll_policy_params.policy_model
hl_policy_params.policy_model_params = copy.deepcopy(ll_policy_params.policy_model_params)
hl_policy_params.policy_model_checkpoint = hl_policy_params.prior_model_checkpoint
hl_policy_params.policy_model_params.batch_size = base_agent_params.batch_size


# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,
    nz_mid=256,
    action_input=True,
)

# HL GAIL Demo Dataset
from spirl.components.data_loader import GlobalSplitVideoDataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.update(AttrDict(
    crop_rand_subseq=True,
    subseq_len=2,
    n_seqs=5,
    seq_repeat=100,
))

# HL Pre-Trained Demo Discriminator
demo_discriminator_config = AttrDict(
    state_dim=data_spec.state_dim,
    normalization='none',
    demo_data_conf=data_config,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=LearnedPPPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
    discriminator=DemoDiscriminator,
    discriminator_params=demo_discriminator_config,
    discriminator_checkpoint=os.path.join(os.environ["EXP_DIR"], "demo_discriminator/maze/maze_discr"),
    freeze_discriminator=True,      # don't update pretrained discriminator
    buffer=UniformReplayBuffer,
    buffer_params={'capacity': 1e6,},
    reset_buffer=False,
    replay=UniformReplayBuffer,
    replay_params={'dump_replay': False, 'capacity': 2e6},
    expert_data_conf=data_config,
    expert_data_path=os.path.join(os.environ['DATA_DIR'], 'maze_demos'),
))

# SkiLD Parameters
hl_agent_config.update(AttrDict(
    lambda_gail_schedule_params=AttrDict(p=0.9),
    td_schedule_params=AttrDict(p=10.0),
    tdq_schedule_params=AttrDict(p=1.0),
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=MazeSkiLDAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=True,
    update_hl=True,
    update_ll=False,
)

# Sampler
sampler_config = AttrDict(
)

# Environment
env_config = AttrDict(
    reward_norm=1,
)

