# Learning Task-relevant Representations via Rewards and Real Actions for Reinforcement Learning


This is the code of paper **Learning Task-relevant Representations via Rewards and Real Actions for Reinforcement Learning**.


## Requirements

```
pip install -r requirements.txt
```

## Reproduce the Results on Distracting DeepMind Control

For example, run experiments on Cartpole Swingup with background distractions using our auxiliary task

``` bash
bash run.sh
```

Modify the `--env` argument in `run.sh` to specify a different task, use the `--agent` argument to select a reinforcement learning agent from either the curl agent or the drq agent, and utilize the `--auxiliary` argument to choose an auxiliary task between cresp and our method (denoted by rra).


## Citation

If you find this code useful, please consider citing the following paper.

## Remarks

```
@article{yang2022learning,
  title={Learning Task-relevant Representations for Generalization via Characteristic Functions of Reward Sequence Distributions},
  author={Yang, Rui and Wang, Jie and Geng, Zijie and Ye, Mingxuan and Ji, Shuiwang and Li, Bin and Wu, Feng},
  journal={arXiv preprint arXiv:2205.10218},
  year={2022}
}
```
