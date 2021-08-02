import os
import copy

from spirl.utils.general_utils import AttrDict
from skild.models.demo_discriminator import DemoDiscriminator, DemoDiscriminatorLogger
from spirl.configs.default_data_configs.office import data_spec
from spirl.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': DemoDiscriminator,
    'model_test': DemoDiscriminator,
    'logger': DemoDiscriminatorLogger,
    'logger_test': DemoDiscriminatorLogger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'office_TA'),
    'num_epochs': 100,
    'epoch_cycles_train': 300,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    action_dim=data_spec.n_actions,
    normalization='none',
)

# Demo Dataset
demo_data_config = AttrDict()
demo_data_config.dataset_spec = copy.deepcopy(data_spec)
demo_data_config.dataset_spec.crop_rand_subseq = True
demo_data_config.dataset_spec.subseq_len = 1+1
demo_data_config.dataset_spec.n_seqs = 100          # number of demos used
demo_data_config.dataset_spec.seq_repeat = 100      # repeat those demos N times
demo_data_config.dataset_spec.split = AttrDict(train=0.5, val=0.5, test=0.0)
model_config.demo_data_conf = demo_data_config
model_config.demo_data_path = os.path.join(os.environ['DATA_DIR'], 'office_demos')

# Non-demo Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.crop_rand_subseq = True
data_config.dataset_spec.subseq_len = 1+1
