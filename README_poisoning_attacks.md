## Poisoning Attacks on Online Learning to Rank

This repository contains the code, that we used to show the robustness of DBGD/MGD based algorithms.

Usage
-------
To run the code to generate experimental results you can simply run the attack.sh script. This script in turns calls another script present in the scripts/Poisoning_attacks directory. Four scripts are provided there depending on the algorithm and the learning rate decay.

An example of such a script is given:
```
python2 ./scripts/Poisoning_attacks/attack_DBGD_base_lr.py --data_sets local_MQ2007_F1 --click_models frequency_attack\
       --user_click_model exper1 --log_folder ./log --output_folder ./output --average_folder ./average \
       --n_impr 10000 --n_runs 1 --n_proc 10 --n_results 10 --start 0 --end 1 --which 3 --mf 5 --sd_const 2.0
```                 

To know the details for each of the arguments, you can look at the **utils/argparsers/simulationargparser.py** file.


