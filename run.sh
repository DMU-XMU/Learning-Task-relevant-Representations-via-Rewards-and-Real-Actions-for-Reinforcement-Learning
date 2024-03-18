python launch.py \
--env dmc.cartpole.swingup \
--agent curl \
--base sac \
--auxiliary cresp \
--num_sources 2 \
--dynamic -bg -tbg \
--disenable_default \
--targ_extr \
--critic_lr 5e-4 \
--actor_lr 5e-4 \
--alpha_lr 5e-4 \
--extr_lr 5e-4 \
--nstep_of_rsd 5 \
--num_sample 256 \
--opt_mode max \
--omega_opt_mode min_mu \
--rs_fc \
--discount_of_rs 0.8 \
--extr_update_via_qfloss True \
--cuda_id 0

# python launch.py \
# --env dmc.cartpole.swingup_sparse \
# --agent curl \
# --base sac \
# --auxiliary rra \
# --num_sources 2 \
# --dynamic -bg -tbg \
# --disenable_default \
# --targ_extr \
# --critic_lr 5e-4 \
# --actor_lr 5e-4 \
# --alpha_lr 5e-4 \
# --extr_lr 5e-4 \
# --nstep_of_rsd 5 \
# --num_sample 256 \
# --opt_mode max \
# --omega_opt_mode min_mu \
# --rs_fc \
# --discount_of_rs 0.8 \
# --extr_update_via_qfloss True \
# --cuda_id 0

# python launch.py \
# --env dmc.cheetah.run \
# --agent curl \
# --base sac \
# --auxiliary rra \
# --num_sources 2 \
# --dynamic -bg -tbg \
# --disenable_default \
# --targ_extr \
# --critic_lr 5e-4 \
# --actor_lr 5e-4 \
# --alpha_lr 5e-4 \
# --extr_lr 5e-4 \
# --nstep_of_rsd 5 \
# --num_sample 256 \
# --opt_mode max \
# --omega_opt_mode min_mu \
# --rs_fc \
# --discount_of_rs 0.8 \
# --extr_update_via_qfloss True \
# --cuda_id 0

# python launch.py \
# --env dmc.hopper.stand \
# --agent curl \
# --base sac \
# --auxiliary rra \
# --num_sources 2 \
# --dynamic -bg -tbg \
# --disenable_default \
# --targ_extr \
# --critic_lr 5e-4 \
# --actor_lr 5e-4 \
# --alpha_lr 5e-4 \
# --extr_lr 5e-4 \
# --nstep_of_rsd 5 \
# --num_sample 256 \
# --opt_mode max \
# --omega_opt_mode min_mu \
# --rs_fc \
# --discount_of_rs 0.8 \
# --extr_update_via_qfloss True \
# --cuda_id 1

# python launch.py \
# --env dmc.reacher.easy \
# --agent curl \
# --base sac \
# --auxiliary rra \
# --num_sources 2 \
# --dynamic -bg -tbg \
# --disenable_default \
# --targ_extr \
# --critic_lr 5e-4 \
# --actor_lr 5e-4 \
# --alpha_lr 5e-4 \
# --extr_lr 5e-4 \
# --nstep_of_rsd 5 \
# --num_sample 256 \
# --opt_mode max \
# --omega_opt_mode min_mu \
# --rs_fc \
# --discount_of_rs 0.8 \
# --extr_update_via_qfloss True \
# --cuda_id 0

# python launch.py \
# --env dmc.ball_in_cup.catch \
# --agent curl \
# --base sac \
# --auxiliary rra \
# --num_sources 2 \
# --dynamic -bg -tbg \
# --disenable_default \
# --targ_extr \
# --critic_lr 5e-4 \
# --actor_lr 5e-4 \
# --alpha_lr 5e-4 \
# --extr_lr 5e-4 \
# --nstep_of_rsd 5 \
# --num_sample 256 \
# --opt_mode max \
# --omega_opt_mode min_mu \
# --rs_fc \
# --discount_of_rs 0.8 \
# --extr_update_via_qfloss True \
# --cuda_id 1