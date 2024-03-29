python launch.py \
    --env carla.highway \
    --agent curl \
    --base sac \
    --auxiliary rra \
    --num_sources 2 \
    --dynamic -bg -tbg \
    --disenable_default \
    --targ_extr \
    --critic_lr 1e-4 \
    --actor_lr 1e-4 \
    --alpha_lr 1e-4 \
    --extr_lr 1e-4 \
    --nstep_of_rsd 5 \
    --batch_size 128 \
    --num_sample 128 \
    --total_steps 400000 \
    --init_steps 100 \
    --num_eval_episodes 20 \
    --steps_per_epoch 8000 \
    --opt_mode max \
    --omega_opt_mode min_mu \
    --discount_of_rs 0.8 \
    --extr_update_via_qfloss True \
    --cuda_id 0 \
    -s 0
