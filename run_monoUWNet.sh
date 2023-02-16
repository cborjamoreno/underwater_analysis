#!/bin/bash

run_name=prueba1
for ds in uc flatiron tiny
do
    echo "running flc_new evaluation on ${ds}"
    python monoUWNet/evaluate_depth.py --model_name="${run_name}eval${ds}" --eval_mono --load_weights_folder="20220908_FLC_all_wo_rhf_FLC_4DS_tiny_sky/models/weights_last" --save_pred_disps --use_depth --eval_split=${ds} --eval_sky
done