from skild.configs.skill_prior.office.conf import *

configuration['data_dir'] = os.path.join(os.environ['DATA_DIR'], 'office_demos')
data_config.dataset_spec.n_seqs = 100         # number of demos
data_config.dataset_spec.seq_repeat = 100     # how often to repeat these demos
data_config.dataset_spec.split = AttrDict(train=0.5, val=0.5, test=0.0)   # use half of the demos for validation

configuration['epoch_cycles_train'] = 1000

model_config.embedding_checkpoint = os.path.join(os.environ["EXP_DIR"],
                  "skill_prior/office/office_prior/weights")
