python2 ./scripts/Poisoning_attacks/attack_DBGD_base_lr.py --data_sets local_MQ2007_F1 --click_models frequency_attack\
       --user_click_model exper1 --log_folder ./log --output_folder ./output --average_folder ./average \
       --n_impr 1000 --n_runs 1 --n_proc 10 --n_results 10 --start 0 --end 1 --which 3 --mf 5 --sd_const 2.0