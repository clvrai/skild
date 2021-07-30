# Demonstration-Guided Reinforcement Learning with Learned Skills
#### [[Project Website]](https://clvrai.github.io/skild/) [[Paper]](https://arxiv.org/abs/2107.10253)

[Karl Pertsch](https://kpertsch.github.io/)<sup>1</sup>, [Youngwoon Lee](https://youngwoon.github.io/)<sup>1</sup>, 
[Yue Wu](https://ventusyue.github.io/)<sup>1</sup>, [Joseph Lim](https://www.clvrai.com/)<sup>1</sup>

<sup>1</sup>CLVR Lab, University of Southern California 

<a href="https://clvrai.github.io/skild/">
<p align="center">
<img src="docs/resources/skild_teaser.png" width="600">
</p>
</img></a>

This is the official PyTorch implementation of the paper "**Demonstration-Guided Reinforcement Learning with Learned Skills**".

## Requirements

- python 3.7+
- mujoco 2.0 (for RL experiments)
- Ubuntu 18.04

## Installation Instructions

Create a virtual environment and install all required packages:
```
cd skild
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies and package
pip3 install -r requirements.txt
pip3 install -e .
```

Install [SPiRL](https://github.com/clvrai/spirl) as a git submodule:
```
# Download SPiRL as a submodule (all requirements should already be installed)
git submodule update --init --recursive
cd spirl
pip3 install -e .
cd ..
```

Set the environment variables that specify the root experiment and data directories. For example: 
```
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

Finally, for running RL experiments on maze or kitchen environments, install our fork of the
[D4RL benchmark](https://github.com/kpertsch/d4rl) repository by following its installation instructions. Also make sure
to place your Mujoco license file `mj_key.txt` in `~/.mujoco`.
For running RL in the office environment, install our fork of the [Roboverse repo](https://github.com/VentusYue/roboverse)
and follow it's installation instructions for installing PyBullet.

## Example Commands
Our skill-based imitation / demo-guided RL pipeline is run in four steps: (1) train skill embedding and skill prior, 
(2) train skill posterior, (3) train demo discriminator, (4) use all components for demo-guided RL or imitation learning
on the downstream task.

All results will be written to [WandB](https://www.wandb.com/). Before running any of the commands below, 
create an account and then change the WandB entity and project name at the top of [train.py](https://github.com/clvrai/spirl/spirl/train.py) and
[rl/train.py](https://github.com/clvrai/spirl/spirl/rl/train.py) to match your account.

#### Skill Embedding & Prior
To train a skill prior model for the kitchen environment we'll use SPiRL -- run:
```
python3 spirl/spirl/train.py --path=spirl/spirl/configs/skill_prior_learning/kitchen/hierarchical_cl --val_data_size=160
```

#### Skill Posterior
For training the skill posterior on the demonstration data, run:
```
python3 spirl/spirl/train.py --path=skild/configs/skill_posterior/kitchen --val_data_size=160
```
Note that the skill posterior can only be trained once skill embedding and prior training is completed 
since it leverages the pre-trained skill embedding.

#### Demo Discriminator
For training the demonstration discriminator, run:
```
python3 spirl/spirl/train.py --path=skild/configs/demo_discriminator/kitchen --val_data_size=160
```

#### Demonstration-Guided RL
For training a SkiLD agent on the kitchen environment using the pre-trained components from above, run:
```
python3 spirl/spirl/rl/train.py --path=skild/configs/demo_rl/kitchen --seed=0 --prefix=SkiLD_demoRL_kitchen_seed0
```

#### Imitation Learning
For training a SkiLD agent on the kitchen environment with pure imitation learning, run:
```
python3 spirl/spirl/rl/train.py --path=skild/configs/imitation/kitchen --seed=0 --prefix=SkiLD_IL_kitchen_seed0
```

In all commands above, `kitchen` can be replaced with `maze / office` to run on the respective environment. Before training models
on these environments, the corresponding datasets need to be downloaded (the kitchen dataset gets downloaded automatically) 
-- download links are provided below.


## Datasets

|Dataset        | Link         | Size |
|:------------- |:-------------|:-----|
| Maze Task-Agnostic | [https://drive.google.com/file/d/103RFpEg4ATnH06fd1ps8ZQL4sTtifrvX/view?usp=sharing](https://drive.google.com/file/d/103RFpEg4ATnH06fd1ps8ZQL4sTtifrvX/view?usp=sharing)| 470MB |
| Maze Demos | [https://drive.google.com/file/d/1wTR9ns5QsEJnrMJRXFEJWCMk-d1s4S9t/view?usp=sharing](https://drive.google.com/file/d/1wTR9ns5QsEJnrMJRXFEJWCMk-d1s4S9t/view?usp=sharing)| 100MB |
| Office Cleanup Task-Agnostic | [https://drive.google.com/file/d/1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80/view?usp=sharing](https://drive.google.com/file/d/1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80/view?usp=sharing)| 170MB |
| Office Cleanup Demos | [https://drive.google.com/file/d/149trMTyh3A2KnbUOXwt6Lc3ba-1T9SXj/view?usp=sharing](https://drive.google.com/file/d/149trMTyh3A2KnbUOXwt6Lc3ba-1T9SXj/view?usp=sharing)| 6MB |

To download the dataset files from Google Drive via the command line, you can use the 
[gdown](https://github.com/wkentaro/gdown) package. Install it with:
```
pip install gdown
```

Then navigate to the folder you want to download the data to and run the following commands:
```
# Download Maze Task-Agnostic Dataset
gdown https://drive.google.com/uc?id=103RFpEg4ATnH06fd1ps8ZQL4sTtifrvX

# Download Maze Demonstration Dataset
gdown https://drive.google.com/uc?id=1wTR9ns5QsEJnrMJRXFEJWCMk-d1s4S9t

# Download Office Task-Agnostic Dataset
gdown https://drive.google.com/uc?id=1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80

# Download Office Demonstration Dataset
gdown https://drive.google.com/uc?id=149trMTyh3A2KnbUOXwt6Lc3ba-1T9SXj
``` 

## Code Structure & Modifying the Code
For a more detailed documentation of the code structure and how to extend the code (adding new enviroments, models, RL algos)
please check the [documentation in the SPiRL repo](https://github.com/clvrai/spirl#starting-to-modify-the-code).

## Citation
If you find this work useful in your research, please consider citing:
```
@article{pertsch2021skild,
         title={Demonstration-Guided Reinforcement Learning with Learned Skills},
         author={Karl Pertsch and Youngwoon Lee and Yue Wu and Joseph J. Lim},
         journal={arxiv:210712345},
         year={2021},
}
```

## Acknowledgements
Most of the heavy-lifting in this code is done by the [SPiRL codebase](https://github.com/clvrai/spirl), published as part
of our prior work. 

We thank Justin Fu and Aviral Kumar et al. for providing the [D4RL codebase](https://github.com/rail-berkeley/d4rl)
which we use for some of our experiments. We also thank Avi Singh et al. for open-sourcing the [Roboverse repo](https://github.com/avisingh599/roboverse)
which we build on for our office environment experiments. 

 