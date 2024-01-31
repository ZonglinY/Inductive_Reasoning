# Inductive_Reasoning

This repository is the official implementation of the paper \<Language Models as Inductive Reasoners\>.   
[\[Arxiv version\]](https://arxiv.org/abs/2212.10923).

In general, with this repository, you can     
(1) generate hypotheses with the CoLM framework, and   
(2) display results listed in the paper.

## Generate hypotheses with the CoLM framework
Will be updated soon.

## Display results listed in the paper

### GPT-J's few-shot result
Automatic evaluation (a part of Table 4, full Table 5, and full Table 6): ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptj_12_5gene/ --generator_model_type gptj --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  

Human evaluation (a part of Table 4): ```python final_human_eval_result.py --output_dir ./Checkpoints/gptj_analysis_100test_newdata_newprompt_10 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  

### GPT-J's finetune results   
Automatic evaluation (a part of Table 4): ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptj_12_5gene/ --generator_model_type gptj --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 3 --if_already_fintuned_for_test 1```  

Human evaluation (a part of Table 4): ```python final_human_eval_result.py --output_dir ./Checkpoints/gptj_analysis_100test_newdata_newprompt_10 --setting_selection_M1_forM2M3 1 --setting_selection 3 --if_already_fintuned_for_test 1```  

### Ablation on input facts (Table 7)  
Long fact, 1 full fact: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptj_12_5gene_1fact_long/ --generator_model_type gptj --if_long_or_short_facts 0 --cnt_facts_as_input 1 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  

Short fact, 1 full fact: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptj_12_5gene_1fact/ --generator_model_type gptj --if_long_or_short_facts 1 --cnt_facts_as_input 1 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  

Short fact, 2 full facts: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptj_12_5gene_2fact/ --generator_model_type gptj --if_long_or_short_facts 1 --cnt_facts_as_input 2 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  

Short fact, 3 missing facts: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptj_12_5gene_missingfacts/ --generator_model_type gptj --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 1 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  

[comment]:<### Ablation on model size (Figure 2)  >
[comment]:<gptneo125M: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptneo125M_12_5gene/ --generator_model_type gptneo125M --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```   >

[comment]:<gptneo1.3B: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptneo1.3B_12_5gene/ --generator_model_type gptneo1.3B --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```   >

[comment]:<gptneo2.7B: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptneo2.7B_12_5gene/ --generator_model_type gptneo2.7B --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```   >

[comment]:<gptneox20B: ```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_gptneox20B_12_5gene/ --generator_model_type gptneox20B --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```  >



### Llama's result (Table 9)  
```python bleu_green_calculator_analysis.py --output_dir ./Checkpoints/new_data_llama_12_5gene_capitalYesNo/ --generator_model_type llama --if_long_or_short_facts 1 --cnt_facts_as_input 3 --if_full_or_missing_facts 0 --setting_selection_M1_forM2M3 1 --setting_selection 2 --if_already_fintuned_for_test 0```
