from skild.configs.skill_prior.kitchen.conf import *

data_config.dataset_spec.filter_indices = [[320, 337], [339, 344]]      # use only demos for one task (here: KBTS)
data_config.dataset_spec.demo_repeats = 10                              # repeat those demos N times

model_config.embedding_checkpoint = os.path.join(os.environ["EXP_DIR"],
                  "skill_prior/kitchen/kitchen_prior/weights")
