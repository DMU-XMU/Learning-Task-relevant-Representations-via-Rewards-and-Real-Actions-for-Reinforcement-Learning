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

Modify the `--env` argument in `run.sh` to specify a different task, employ the `--agent` argument to select a reinforcement learning agent from either the curl agent or the drq agent, utilize the `--auxiliary` argument to choose an auxiliary task between cresp and our method (denoted by rra), and utilize the `-s` argument to set the seed.

## Reproduce the Results on CARLA
### install CARLA
Please first install UE4.26 before installing CARLA.

Download CARLA from https://github.com/carla-simulator/carla/releases, e.g., https://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz.

Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/home/XXXX/CARLA_0.9.6/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/XXXX/CARLA_0.9.6/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/XXXX/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
```

Install:
```
pip install pygame
pip install networkx
```

Move the 'carla_env.py' file to the `/home/XXXX/CARLA_0.9.6/PythonAPI/carla/agents/navigation` directory.

### run experiments on CARLA
First open the CARLA engine:

Terminal 1:
```
cd CARLA_0.9.6
bash CarlaUE4.sh --RenderOffScreen --carla-rpc-port=1314 --fps=20
```

Then run experiments on CARLA using our auxiliary task:

Terminal 2:
```
bash runCarla096.sh
```

All experimental results will be stored under `data` directory.

## Reference
Our code is modified based on: 
1. https://github.com/MIRALab-USTC/RL-CRESP.git 
2. https://github.com/facebookresearch/deep_bisim4control.git 
