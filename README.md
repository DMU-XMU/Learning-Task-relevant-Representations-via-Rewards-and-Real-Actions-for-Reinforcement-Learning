# Learning Task-relevant Representations via Rewards and Real Actions for Reinforcement Learning


This is the code of paper **Learning Task-relevant Representations via Rewards and Real Actions for Reinforcement Learning**.


## Requirements

```
pip install -r requirements.txt
```

## Reproduce the Results on Distracting DeepMind Control

For example, run experiments on Cartpole Swingup with background distractions using our auxiliary task:

``` bash
bash run.sh
```

Modify the `--env` argument in `run.sh` to specify a different task, use the `--agent` argument to select a reinforcement learning agent from either the curl agent or the drq agent, and utilize the `--auxiliary` argument to choose an auxiliary task between cresp and our method (denoted by rra).

## Reproduce the Results on Distracting DeepMind Control
### install CARLA
Please firstly install UE4.26.

Download CARLA from https://github.com/carla-simulator/carla/releases, e.g., https://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz.

Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.6/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.6/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
```

Install:
```
pip install pygame
pip install networkx
```





## Remarks

```
@article{yang2022learning,
  title={Learning Task-relevant Representations for Generalization via Characteristic Functions of Reward Sequence Distributions},
  author={Yang, Rui and Wang, Jie and Geng, Zijie and Ye, Mingxuan and Ji, Shuiwang and Li, Bin and Wu, Feng},
  journal={arXiv preprint arXiv:2205.10218},
  year={2022}
}
```
