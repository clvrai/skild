import tqdm
import gym
import d4rl
import numpy as np

from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.kitchen import data_spec


OBJS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3


## Demo Dataset
demo_data_config = AttrDict()
demo_data_config.device = 'cpu'
demo_data_config.dataset_spec = data_spec
demo_data_config.dataset_spec.crop_rand_subseq = True
demo_data_config.dataset_spec.subseq_len = 1+1+(3-1)

loader = data_spec.dataset_class('.', demo_data_config, resolution=32, phase='train', shuffle=True, dataset_size=-1)
seqs = loader.seqs

## determine achieved subgoals + respective time steps
n_seqs, n_objs = len(seqs), len(OBJS)
subtask_steps = np.Inf * np.ones((n_seqs, n_objs))
for s_idx, seq in tqdm.tqdm(enumerate(seqs)):
    for o_idx, obj in enumerate(OBJS):
        for t, state in enumerate(seq.states):
            obj_state, obj_goal = state[OBS_ELEMENT_INDICES[obj]], OBS_ELEMENT_GOALS[obj]
            dist = np.linalg.norm(obj_state - obj_goal)
            if dist < BONUS_THRESH and subtask_steps[s_idx, o_idx] == np.Inf:
                subtask_steps[s_idx, o_idx] = t

## print subtask orders
print("\n\n")

subtask_freqs = {k+'_'+j+'_'+i+'_'+kk: 0 for k in OBJS for j in OBJS for i in OBJS for kk in OBJS}
for s_idx, subtasks in enumerate(subtask_steps):
    min_task_idxs = np.argsort(subtasks)[:4]
    objs = [OBJS[i] for i in min_task_idxs]
    subtask_freqs[OBJS[min_task_idxs[0]]+'_'+OBJS[min_task_idxs[1]]\
                  +'_'+OBJS[min_task_idxs[2]]+'_'+OBJS[min_task_idxs[3]]] += 1
    print("seq {}: {}".format(s_idx, objs))

print("\n\n")
for k in subtask_freqs:
    if subtask_freqs[k] > 0:
        print(k,": ", subtask_freqs[k])



