from skild.configs.skill_prior.maze.conf import *

configuration['data_dir'] = os.path.join(os.environ['DATA_DIR'], 'maze_demos')
data_config.dataset_spec.n_seqs = 5          # number of demos
data_config.dataset_spec.seq_repeat = 30     # how often to repeat these demos

configuration['epoch_cycles_train'] = 4200

model_config.embedding_checkpoint = os.path.join(os.environ["EXP_DIR"],
                  "skill_prior/maze/maze_prior/weights")
