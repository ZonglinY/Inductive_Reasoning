#!/bin/bash
#SBATCH -J 18
#SBATCH --partition=PA100q
#SBATCH -w node02
#SBATCH --gres=gpu:1
#SBATCH --output /export/home2/zonglin001/Outs/CoLM/new_data_llama_18_5gene_capitalYesNo.out

# Module 2/3/4/5
# llama / vicunallama / gptneox20B / gptj / gptneo2.7B / gptneo1.3B / gptneo125M
# new_data_gptj_12_5gene
python -u main.py --output_dir ~/Checkpoints/CoLM/new_data_llama_12_5gene_capitalYesNo --do_test --setting_selection_M1_forM2M3 1 --setting_selection 2 --generator_model_type llama --dataset_selection 18 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_use_deerlet_val_train_for_test 1


# # use finetuned Checkpoints
# python -u main.py --output_dir /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene/ --do_test --setting_selection_M1_forM2M3 1 --setting_selection 3 --generator_model_type gptj --dataset_selection 20 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_already_fintuned_for_test 1 --finetuned_checkpoint_dir /export/home2/zonglin001/Checkpoints/CoLM/gptj_fintune_on_DEERELET_D18_bs4_accES_5patience/

# python -u main.py --output_dir ~/Checkpoints/CoLM/try12_vicunallama --do_test --setting_selection_M1_forM2M3 1 --setting_selection 1 --generator_model_type vicunallama --dataset_selection 12 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0


# finetune dataset_selection == 18 with t5
# python -u main.py --output_dir ~/Checkpoints/CoLM/t5-11b_fintune_on_DEERELET_D18_try --do_train --do_test --setting_selection_M1_forM2M3 0 --setting_selection 0 --generator_model_type t5-11B --dataset_selection 18 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_use_deerlet_val_train_for_test 0


# finetune; added weight decay, change num_warmup_steps
# python -u main.py --output_dir ~/Checkpoints/CoLM/gptj_fintune_on_DEERELET_D18_bs4_accES_5patience --do_train --do_test --setting_selection_M1_forM2M3 0 --setting_selection 3 --generator_model_type gptj --dataset_selection 18 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_use_deerlet_val_train_for_test 0 --weight_decay 0.1 --eval_per_steps 50 --patience 5 --train_batch_size 4 --dev_batch_size 4 --test_batch_size 4


# python -u main.py --output_dir ./Checkpoints_new/gptj_analysis_100test_newdata_newprompt_10 --do_test --setting_selection_M1_forM2M3 1 --setting_selection 0 --generator_model_type gpt2-lmhead --dataset_selection 18 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_already_fintuned_for_test 1 --finetuned_checkpoint_dir ~/Checkpoints_new/gpt2_fintune_on_DEERELET_D18/


# use DEER train to make generations as in test mode, for annotation of DEERLET
# nohup python -u main.py --output_dir ./Checkpoints_try/gptj_analysis_78train_fortest_newdata_newprompt_numgene10 --do_test --setting_selection_M1_forM2M3 1 --setting_selection 1 --generator_model_type gptj --dataset_selection 12 --num_gene_times 12 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_use_deer_train_data_for_test 1


# use val set as test set to get thresholds or just simply use test set
# python -u main.py --output_dir ./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10 --do_test --setting_selection_M1_forM2M3 1 --setting_selection 2 --generator_model_type gptj --dataset_selection 18 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_use_deerlet_val_train_for_test 0


# use val set as test set to get thresholds (using finetuned checkpoints)
# python -u main.py --output_dir ./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10 --do_test --setting_selection_M1_forM2M3 1 --setting_selection 3 --generator_model_type gptj --dataset_selection 15 --num_gene_times 1 --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --if_use_deerlet_val_train_for_test 1 --if_already_fintuned_for_test 1 --finetuned_checkpoint_dir /export/home2/zonglin001/Checkpoints/CoLM/gptj_fintune_on_DEERELET_D15_bs4_accES/
