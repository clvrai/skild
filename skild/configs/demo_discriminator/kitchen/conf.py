import os
import copy

from spirl.utils.general_utils import AttrDict
from skild.models.demo_discriminator import DemoDiscriminator, DemoDiscriminatorLogger
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': DemoDiscriminator,
    'model_test': DemoDiscriminator,
    'logger': DemoDiscriminatorLogger,
    'logger_test': DemoDiscriminatorLogger,
    'data_dir': ".",
    'num_epochs': 200,
    'epoch_cycles_train': 100,
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
demo_data_config.dataset_spec.filter_indices = [[320, 337], [339, 344]]  # use only demos for one task (here: KBTS)
demo_data_config.dataset_spec.demo_repeats = 10                          # repeat those demos N times
model_config.demo_data_conf = demo_data_config

# Non-demo Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.crop_rand_subseq = True
data_config.dataset_spec.subseq_len = 1+1
