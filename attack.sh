python2 scripts/Poisoning_attacks/attack_DBGD_base_lr.py --data_sets local_MQ2007 --attacker_click_model frequency_attack\
       --click_models exper1 --log_folder ./log --output_folder ./output --average_folder ./average \
       --n_impr 10000 --n_runs 5 --n_proc 10 --n_results 10 --start 0 --end 1 --which 1 --mf 5 --sd_const 2.0 --num_attacker_relevant 5