## Poisoning Attacks on Online Learning to Rank

This repository contains the code, that we used to show the robustness of DBGD/MGD based algorithms.

Usage
-------
To run the code to generate experimental results you can simply run the attack.sh script. This script in turns calls another script present in the scripts/Poisoning_attacks directory. Four scripts are provided there depending on the algorithm and the learning rate decay.

An example of such a script is given:
```
python2 scripts/Poisoning_attacks/attack_DBGD_base_lr.py --data_sets local_MQ2007 --attacker_click_model frequency_attack\
       --click_models exper1 --log_folder ./log --output_folder ./output --average_folder ./average \
       --n_impr 10000 --n_runs 10 --n_proc 10 --n_results 10 --start 0 --end 1 --which 1 --mf 5 --sd_const 2.0 --num_attacker_relevant 5
```                 

To know the details for each of the arguments, you can look at the **utils/argparsers/simulationargparser.py** file.


