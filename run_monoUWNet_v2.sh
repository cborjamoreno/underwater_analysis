#!/bin/bash

run_name=prueba1
echo "running flc_new evaluation on ${ds}"
python monoUWNet/evaluate_depth.py --eval_mono --load_weights_folder="20220908_FLC_all_wo_rhf_FLC_4DS_tiny_sky/models/weights_last" --save_pred_disps --use_depth --no_eval --eval_out_dir="./"