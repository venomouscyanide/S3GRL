#!/bin/bash

cd ../../

python ray_tuning.py --dataset USAir --sign_type SuP --identifier usair_sup_id --output_path usair_tuning_sup
python ray_tuning.py --dataset USAir --sign_type KSuP --identifier usair_ksup_id --output_path usair_tuning_ksup

python ray_tuning.py --dataset NS --sign_type SuP --identifier ns_sup_id --output_path ns_tuning_sup
python ray_tuning.py --dataset NS --sign_type KSuP --identifier ns_ksup_id --output_path ns_tuning_ksup

python ray_tuning.py --dataset Power --sign_type SuP --identifier power_sup_id --output_path power_tuning_sup
python ray_tuning.py --dataset Power --sign_type KSuP --identifier power_ksup_id --output_path power_tuning_ksup

python ray_tuning.py --dataset Celegans --sign_type SuP --identifier celegans_sup_id --output_path celegans_tuning_sup
python ray_tuning.py --dataset Celegans --sign_type KSuP --identifier celegans_ksup_id --output_path celegans_tuning_ksup

python ray_tuning.py --dataset Router --sign_type SuP --identifier router_sup_id --output_path router_tuning_sup
python ray_tuning.py --dataset Router --sign_type KSuP --identifier router_ksup_id --output_path router_tuning_ksup

python ray_tuning.py --dataset PB --sign_type SuP --identifier pb_sup_id --output_path pb_tuning_sup
python ray_tuning.py --dataset PB --sign_type KSuP --identifier pb_ksup_id --output_path pb_tuning_ksup

python ray_tuning.py --dataset Ecoli --sign_type SuP --identifier ecoli_sup_id --output_path ecoli_tuning_sup
python ray_tuning.py --dataset Ecoli --sign_type KSuP --identifier ecoli_ksup_id --output_path ecoli_tuning_ksup

python ray_tuning.py --dataset Yeast --sign_type SuP --identifier yeast_sup_id --output_path yeast_tuning_sup
python ray_tuning.py --dataset Yeast --sign_type KSuP --identifier yeast_ksup_id --output_path yeast_tuning_ksup

