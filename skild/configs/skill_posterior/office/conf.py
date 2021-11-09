from skild.configs.skill_prior.office.conf import *

configuration['data_dir'] = os.path.join(os.environ['DATA_DIR'], 'office_demos')
data_config.dataset_spec.n_seqs = 50         # number of demos
data_config.dataset_spec.seq_repeat = 3      # how often to repeat these demos

configuration['epoch_cycles_train'] = 6000

model_config.embedding_checkpoint = os.path.join(os.environ["EXP_DIR"],
                  "skill_prior/office/office_prior/weights")
