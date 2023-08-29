from utils import load_data_Hypothetical_Induction_Module123, whether_not_included_in_in_context_demonstrations_in_rule_proposer
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
import torch
import argparse, os, copy
import numpy as np
import pandas as pd



## recall*
def get_wrecall(args, ttl_bleu_idx, golden_rules, generated_rule, cnt_valid_bleu, pred_score=None, pred_score14=None, pred_score19=None, thres=None, thres13=None, thres14=None, thres19=None):
    ttl_recall_top_ratio = []
    ttl_recall_star = []
    recalled_data_start = 0
    prev_recall_ratio = 0
    for recall_ratio in np.linspace(0.1, 1, 10):
        # recall_ratio = 0.3
        recalled_data_end = int(cnt_valid_bleu * recall_ratio)
        recalled_data = ttl_bleu_idx[recalled_data_start:recalled_data_end]
        # print("recalled_data: ", recalled_data)
        cnt_yes_in_recalled_data = 0
        for id in range(len(recalled_data)):
            cur_id_key, cur_key, cur_id_rule, cur_bleu = recalled_data[id]
            if args.dataset_selection == 13 or args.dataset_selection == 14 or args.dataset_selection == 19:
                if pred_score[cur_key][cur_id_rule] > thres:
                    cnt_yes_in_recalled_data += 1
            elif args.dataset_selection == 1314:
                if pred_score[cur_key][cur_id_rule] > thres13 and pred_score14[cur_key][cur_id_rule] > thres14:
                    cnt_yes_in_recalled_data += 1
            elif args.dataset_selection == 131419:
                if pred_score[cur_key][cur_id_rule] > thres13 and pred_score14[cur_key][cur_id_rule] > thres14 and pred_score19[cur_key][cur_id_rule] > thres19:
                    cnt_yes_in_recalled_data += 1
                # cnt_yes_in_recalled_data += 1
        recall_star = cnt_yes_in_recalled_data / (recalled_data_end - recalled_data_start)
        ttl_recall_star.append(recall_star)
        ttl_recall_top_ratio.append(recall_ratio)
        # print("recall*: ", recall_star)

        recalled_data_gold_rule = [golden_rules[recalled_data[i][0]] for i in range(len(recalled_data))]
        recalled_data_gene_rule = [generated_rule[recalled_data[i][1]][recalled_data[i][2]] for i in range(len(recalled_data))]
        recalled_data_pred_score = [pred_score[recalled_data[i][1]][recalled_data[i][2]] for i in range(len(recalled_data))]
        assert len(recalled_data_gold_rule) == len(recalled_data_gene_rule)
        assert len(recalled_data_gold_rule) == len(recalled_data_pred_score)
        assert len(recalled_data_gold_rule) == len(recalled_data)
        # for id in range(len(recalled_data)):
            # print("\nid: {}; recalled_gold_rule: {}; recalled_generated_rule: {}; recalled_pred_score: {:.3f}".format(id, recalled_data_gold_rule[id], recalled_data_gene_rule[id], recalled_data_pred_score[id]))
        print("bleu ranking rate range: {:.2f}~{:.2f}; recall*: {:.3f}".format(prev_recall_ratio, recall_ratio, recall_star))
        recalled_data_start = recalled_data_end
        prev_recall_ratio = recall_ratio
    assert recalled_data_end == cnt_valid_bleu
    # ave_weighted_recall (use ttl_recall_star and ttl_recall_top_ratio)
    assert len(ttl_recall_star) == len(ttl_recall_top_ratio)
    # print("ttl_recall_top_ratio: ", ttl_recall_top_ratio)
    ttl_weighted_sum_recall_score = 0
    ttl_weight = 0
    for id in range(len(ttl_recall_star)):
        cur_weight = 100 * (1 - ttl_recall_top_ratio[id]) - 45
        cur_recall_score = ttl_recall_star[id]
        ttl_weighted_sum_recall_score += cur_weight * cur_recall_score
        ttl_weight += cur_weight
        # print("cur_recall_score: {:.3f}; cur_weight: {:.1f}".format(cur_recall_score, cur_weight))
    # ttl_weighted_ave_recall_score = ttl_weighted_sum_recall_score / ttl_weight
    # ttl_recall_score_rate = 1 / (1 + np.exp(-ttl_weighted_sum_recall_score/25))
    ttl_recall_score_rate = (ttl_weighted_sum_recall_score + 125) / 250
    print("ttl_weighted_sum_recall_score: {:.3f}; ttl_recall_score_rate: {:.3f}".format(ttl_weighted_sum_recall_score, ttl_recall_score_rate))


## recall*
## INPUT
#   ttl_bleu_idx: do not need to be sorted outside (will be sorted inside)
def get_wrecall_simple(args, dataset_selection, ttl_bleu_idx, cnt_valid_bleu, pred_score13=None, pred_score14=None, pred_score19=None, pred_score20=None, thres13=None, thres14=None, thres19=None, thres20=None):
    # added in 2022/11/23; I think this assertion should be true
    assert len(ttl_bleu_idx) == cnt_valid_bleu
    # sort ttl_bleu_idx first
    ttl_bleu_idx.sort(key=lambda x: x[3], reverse=True)
    ttl_recall_top_ratio = []
    ttl_recall_star = []
    recalled_data_start = 0
    prev_recall_ratio = 0
    # cnt_valid_bleu = len(ttl_bleu_idx)
    for recall_ratio in np.linspace(0.1, 1, 10):
        # recall_ratio = 0.3
        recalled_data_end = int(cnt_valid_bleu * recall_ratio)
        assert recalled_data_end > recalled_data_start
        recalled_data = ttl_bleu_idx[recalled_data_start:recalled_data_end]
        # print("recalled_data: ", recalled_data)
        cnt_yes_in_recalled_data = 0
        for id in range(len(recalled_data)):
            cur_id_key, cur_key, cur_id_rule, cur_bleu = recalled_data[id]
            cur_if_pass_all_filter = True
            if args.if_consider_M234:
                if '13' in str(dataset_selection):
                    if pred_score13[cur_key][cur_id_rule] <= thres13:
                        cur_if_pass_all_filter = False
                if '14' in str(dataset_selection):
                    if pred_score14[cur_key][cur_id_rule] <= thres14:
                        cur_if_pass_all_filter = False
                if '19' in str(dataset_selection):
                    if pred_score19[cur_key][cur_id_rule] <= thres19:
                        cur_if_pass_all_filter = False
                if '20' in str(dataset_selection):
                    if pred_score20[cur_key][cur_id_rule] <= thres20:
                        cur_if_pass_all_filter = False
            if cur_if_pass_all_filter:
                cnt_yes_in_recalled_data += 1
        recall_star = cnt_yes_in_recalled_data / (recalled_data_end - recalled_data_start)
        # print("cnt_yes_in_recalled_data: {}; recalled_data_end - recalled_data_start: {}".format(cnt_yes_in_recalled_data, recalled_data_end - recalled_data_start))
        ttl_recall_star.append(recall_star)
        ttl_recall_top_ratio.append(recall_ratio)

        # print("bleu ranking rate range: {:.2f}~{:.2f}; recall*: {:.3f}".format(prev_recall_ratio, recall_ratio, recall_star))
        recalled_data_start = recalled_data_end
        prev_recall_ratio = recall_ratio
    # print("ttl_recall_star: {}; dataset_selection: {}".format(ttl_recall_star, dataset_selection))
    assert recalled_data_end == cnt_valid_bleu
    # ave_weighted_recall (use ttl_recall_star and ttl_recall_top_ratio)
    assert len(ttl_recall_star) == len(ttl_recall_top_ratio)
    # print("ttl_recall_top_ratio: ", ttl_recall_top_ratio)

    if args.recall_method == 0:
        ## method 1 for calculate recall
        ttl_weighted_sum_recall_score = 0
        ttl_weight = 0
        for id in range(len(ttl_recall_star)):
            # cur_weight: [45, 35, ..., -45]
            cur_weight = 100 * (1 - ttl_recall_top_ratio[id]) - 45
            cur_recall_score = ttl_recall_star[id]
            ttl_weighted_sum_recall_score += cur_weight * cur_recall_score
            ttl_weight += cur_weight
        ttl_recall_score_rate = (ttl_weighted_sum_recall_score + 125) / 250
    elif args.recall_method == 1:
        ## method 2 for calculate recall
        ttl_weighted_sum_recall_score = 0
        ttl_weight = 0
        recall_weight = np.linspace(1, 0.1, 10)
        # recall_weight = np.linspace(0.1, 1, 10)
        # recall_weight = np.linspace(1, 1, 10)
        assert len(recall_weight) == len(ttl_recall_star)
        for id in range(len(ttl_recall_star)):
            cur_weight = recall_weight[id]
            cur_recall_score = ttl_recall_star[id]
            ttl_weighted_sum_recall_score += cur_weight * cur_recall_score
            ttl_weight += cur_weight
        # ttl_recall_score_rate = (ttl_weighted_sum_recall_score + 125) / 250
        ttl_recall_score_rate = ttl_weighted_sum_recall_score / ttl_weight
    else:
        raise NotImplementError
    return ttl_recall_score_rate




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_selection", type=float, default=1314, help="13: Deduction Consistency Evaluator, using input data generated by Rule Proposer; 14: Indiscriminate Comfirmation Handler, using input data generated by Rule Proposer; 19: if_more_general; 1314: use M2 and M3 together; 131419: use M2 and M3 and M4 together")
    parser.add_argument("--root_data_dir", type=str, default="~/openWorld_Analysis_Inductive_Reasoning_PLM/Data/", help="data dir for current dataset")
    # gptj_analysis_100test_newdata_newprompt
    parser.add_argument("--output_dir", default="./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10/", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_gene_times", type=int, default=1, help="call generate() num_gene_times times for each input sentence; basically num_gene_times has the same target with num_return_sequences, but can be implemented in a GPU-restriced way.; only be used when args.dataset_selection == 12")
    parser.add_argument("--bleu_n", type=int, default=4)
    parser.add_argument("--if_specific_or_general_facts", type=int, default=0, help="when 0, only use specific facts to induce rules and get results; when 1, only use general facts to induce rules and get results")
    parser.add_argument("--if_long_or_short_facts", type=int, default=1, help="when 0, use long facts to induce rules; when 1, use short facts to induce rules")
    parser.add_argument("--cnt_facts_as_input", type=int, default=3, help="can be 1/2/3, indicates how many facts to use to induce rules")
    parser.add_argument("--if_full_or_missing_facts", type=int, default=0, help="when 0, use full facts; when 1, only use part of the fact to induce rules")
    parser.add_argument("--setting_selection_M1_forM2M3", type=int, default=1, help="used to identify which generated rules set to filter, useful when dataset_selection==13/14/15/16/17/18; current choices are 0/1")
    parser.add_argument("--setting_selection", type=int, default=2, help="0: zero-shot setting; 1: few-shot setting; 2: few-shot + chain of thought setting; 3: finetuning setting")
    parser.add_argument("--if_use_deer_train_data_for_test", type=int, default=0, help="just 0 all the time; used for rule_proposer file names; should be used when: 1. only used in --do_test but not --do_train; 2. only used when dataset_selection == 12; FUNCTION: rule proposer do test on deer train data (for annotation of train set of deerlet data)")
    parser.add_argument("--if_already_fintuned_for_test", type=int, default=0, help="always 0, unless when using finetuned checkpoint to only test")
    # parser.add_argument("--if_consider_M5", type=int, default=0, help="For our framework, when using M5, set it to 1; for baselines always set it to 0")
    parser.add_argument("--recall_method", type=int, default=0, help="0: the weights for recall are from [45, 35, ..., -45]; 1: the weights for recall are from [1.0, 0.9, ..., 0.1]")
    args = parser.parse_args()

    assert args.setting_selection == 0 or args.setting_selection == 1 or args.setting_selection == 2
    assert args.setting_selection_M1_forM2M3 == 0 or args.setting_selection_M1_forM2M3 == 1
    assert args.if_use_deer_train_data_for_test == 0 or args.if_use_deer_train_data_for_test == 1
    assert args.if_already_fintuned_for_test == 0 or args.if_already_fintuned_for_test == 1
    # assert args.if_consider_M5 == 0 or args.if_consider_M5 == 1
    assert args.recall_method == 0 or args.recall_method == 1

    # threshold
    if args.if_already_fintuned_for_test == 0:
        if args.setting_selection_M1_forM2M3 == 1 and args.setting_selection == 2:
            if args.dataset_selection == 13:
                thres13 = 0.52
                thres14, thres19 = None, None
            elif args.dataset_selection == 14:
                thres14 = 0.465
                thres13, thres19 = None, None
            elif args.dataset_selection == 19:
                thres19 = 0.48
                thres13, thres14 = None, None
            elif args.dataset_selection == 1314:
                thres13 = 0.52
                thres14 = 0.465
                thres19 = None
            elif args.dataset_selection == 131419:
                thres13 = 0.52
                thres14 = 0.465
                thres19 = 0.48
            else:
                raise NotImplementError
        elif args.setting_selection_M1_forM2M3 == 1 and args.setting_selection == 1:
            if args.dataset_selection == 13:
                thres = 0.505
            elif args.dataset_selection == 14:
                thres = 0.46
            elif args.dataset_selection == 19:
                thres = 0.445
            elif args.dataset_selection == 1314:
                thres13 = 0.505
                thres14 = 0.46
            elif args.dataset_selection == 131419:
                thres13 = 0.505
                thres14 = 0.46
                thres19 = 0.445
            else:
                raise NotImplementError
        elif args.setting_selection_M1_forM2M3 == 0 and args.setting_selection == 0:
            if args.dataset_selection == 13:
                thres = 0.00
            elif args.dataset_selection == 14:
                thres = 0.00
            elif args.dataset_selection == 19:
                thres = 0.00
            elif args.dataset_selection == 1314:
                thres13 = 0.00
                thres14 = 0.00
            elif args.dataset_selection == 131419:
                thres13 = 0.00
                thres14 = 0.00
                thres19 = 0.00
            else:
                raise NotImplementError
        else:
            raise NotImplementError
    else:
        if args.setting_selection_M1_forM2M3 == 1 and args.setting_selection == 0:
            if args.dataset_selection == 13:
                thres = 0.41
            elif args.dataset_selection == 14:
                thres = 0.21
            elif args.dataset_selection == 19:
                thres = 0.27
            elif args.dataset_selection == 1314:
                thres13 = 0.41
                thres14 = 0.21
            elif args.dataset_selection == 131419:
                thres13 = 0.41
                thres14 = 0.21
                thres19 = 0.27
            else:
                raise NotImplementError
        else:
            raise NotImplementError
    # golden_rules: [rule0, rule1, ...]
    golden_rules = []
    args_12 = copy.deepcopy(args)
    args_12.dataset_selection = 12
    test_datasets, test_datasets_notes, dict_id2trueRule_test = load_data_Hypothetical_Induction_Module123(args_12, 'test', if_save_dict_files=False, banned_rule_type='none')
    for id in range(len(test_datasets)):
        golden_rules.append(test_datasets[id][-2])

    # generated_rule: {0: [rule1, rule2]}
    # generated_rule = torch.load(os.path.join(args.output_dir, 'rule_proposer_generated_rules_'+str(args.setting_selection_M1_forM2M3)+'.pt'))
    # print("len(generated_rule): ", len(generated_rule))
    # generated_rule = torch.load(os.path.join(args.output_dir, 'rule_proposer_generated_rules.pt'))
    generated_rule = torch.load(os.path.join(args.output_dir, 'rule_proposer_generated_rules_{:.0f}_{:.0f}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test)))
    print("len(generated_rule): ", len(generated_rule))

    # if args.if_already_fintuned_for_test == 0:
    #     classification_file_suffix = "_{:.0f}_{:.0f}.pt".format(args.setting_selection, args.setting_selection_M1_forM2M3)
    # else:
    #     classification_file_suffix = "_{:.0f}_{:.0f}_{:.0f}.pt".format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)

    classification_file_suffix = "_{:.0f}_{:.0f}_{:.0f}.pt".format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)

    pred_score13, pred_score14, pred_score19, pred_score20 = None, None, None, None
    if '13' in str(args.dataset_selection):
        file_name13 = os.path.join(args.output_dir, "module2_classification_results" + classification_file_suffix)
        pred_score13 = torch.load(file_name13)
        assert len(golden_rules) == len(pred_score13)
    if '14' in str(args.dataset_selection):
        file_name14 = os.path.join(args.output_dir, "module3_classification_results" + classification_file_suffix)
        pred_score14 = torch.load(file_name14)
        assert len(golden_rules) == len(pred_score14)
    if '19' in str(args.dataset_selection):
        file_name19 = os.path.join(args.output_dir, "module4_classification_results" + classification_file_suffix)
        pred_score19 = torch.load(file_name19)
        assert len(golden_rules) == len(pred_score19)
    if '20' in str(args.dataset_selection):
        file_name20 = os.path.join(args.output_dir, "module5_classification_results" + classification_file_suffix)
        pred_score20 = torch.load(file_name20)
        assert len(golden_rules) == len(pred_score20)

    # # pred_score: {0: [0.6, 0.4]}
    # if args.dataset_selection == 13:
    #     file_name = "module2_classification_results" + classification_file_suffix
    #     file_name = os.path.join(args.output_dir, file_name)
    #     pred_score13 = torch.load(file_name)
    #     assert len(golden_rules) == len(pred_score13)
    #     pred_score14 = None
    #     pred_score19 = None
    # elif args.dataset_selection == 14:
    #     file_name = "module3_classification_results" + classification_file_suffix
    #     file_name = os.path.join(args.output_dir, file_name)
    #     pred_score14 = torch.load(file_name)
    #     assert len(golden_rules) == len(pred_score14)
    #     pred_score13 = None
    #     pred_score19 = None
    # elif args.dataset_selection == 19:
    #     file_name = "module4_classification_results" + classification_file_suffix
    #     file_name = os.path.join(args.output_dir, file_name)
    #     pred_score19 = torch.load(file_name)
    #     assert len(golden_rules) == len(pred_score19)
    #     pred_score13 = None
    #     pred_score14 = None
    # elif args.dataset_selection == 1314:
    #     file_name = "module2_classification_results" + classification_file_suffix
    #     file_name14 = "module3_classification_results" + classification_file_suffix
    #     file_name = os.path.join(args.output_dir, file_name)
    #     pred_score13 = torch.load(file_name)
    #     file_name14 = os.path.join(args.output_dir, file_name14)
    #     pred_score14 = torch.load(file_name14)
    #     assert len(golden_rules) == len(pred_score13)
    #     assert len(golden_rules) == len(pred_score14)
    #     pred_score19 = None
    # elif args.dataset_selection == 131419:
    #     file_name = "module2_classification_results" + classification_file_suffix
    #     file_name14 = "module3_classification_results" + classification_file_suffix
    #     file_name19 = "module4_classification_results" + classification_file_suffix
    #     file_name = os.path.join(args.output_dir, file_name)
    #     pred_score13 = torch.load(file_name)
    #     file_name14 = os.path.join(args.output_dir, file_name14)
    #     pred_score14 = torch.load(file_name14)
    #     file_name19 = os.path.join(args.output_dir, file_name19)
    #     pred_score19 = torch.load(file_name19)
    #     assert len(golden_rules) == len(pred_score13)
    #     assert len(golden_rules) == len(pred_score14)
    #     assert len(golden_rules) == len(pred_score19)
    # else:
    #     raise NotImplementError
    #
    # if args.if_consider_M5:
    #     file_name20 = "module5_classification_results" + "_{:.0f}_{:.0f}_{:.0f}.pt".format(0, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)
    #     file_name20 = os.path.join(args.output_dir, file_name20)
    #     pred_score20 = torch.load(file_name20)
    #     assert len(golden_rules) == len(pred_score20)
    #     thres20 = 0.5
    # else:
    #     pred_score20 = None
    #     thres20 = None

    # print("pred_score13[0]: ", pred_score13[0])

    assert len(golden_rules) == len(generated_rule)
    # print("golden_rules[0]: ", golden_rules[0])
    # print("pred_score[0]: ", pred_score[0])
    # print("generated_rule[0]: ", generated_rule[0])

    ## BLEU
    n = args.bleu_n
    weights = [1/n] * n
    def score(hyp, refs):
        # print(hyp)
        # print(refs)
        return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

    # ttl_bleu: {0: [bleu0, bleu1]}
    ttl_bleu = {}
    # ttl_bleu_idx: [[id_key, key, id_rule, bleu]]
    ttl_bleu_idx = []
    # ttl_bleu_idx_cnted: ttl_bleu_idx but require "rule.lower().strip()[0:2] == gold_rule.lower().strip()[0:2]"
    ttl_bleu_idx_cnted = []
    # cnt_valid_bleu: cnt of bleu from matched gene_rule and gold_rule
    cnt_valid_bleu = 0
    for id_key, key in enumerate(sorted(generated_rule.keys())):
        gene_rules = generated_rule[key]
        gold_rule = golden_rules[id_key]
        # if id_key == 30:
        #     print("gold_rule: ", gold_rule)
        #     print("gene_rules: ", gene_rules)
        if_should_keep = whether_not_included_in_in_context_demonstrations_in_rule_proposer(gold_rule)
        # print("if_should_keep: ", if_should_keep)
        if if_should_keep:
            for id_rule, rule in enumerate(gene_rules):
                if rule.lower().strip()[0:2] == gold_rule.lower().strip()[0:2]:
                    tmp_bleu = score(rule.lower().strip().strip('.').split(), [t.lower().strip().strip('.').split() for t in [gold_rule]])
                    cnt_valid_bleu += 1
                    ttl_bleu_idx_cnted.append([id_key, key, id_rule, tmp_bleu])
                else:
                    # add tmp_bleu to ttl_bleu no matter whether rule.lower().strip()[0:2] == gold_rule.lower().strip()[0:2] to keep the index in ttl_bleu; 2022/11/23 adaed: but we don't need ttl_bleu in this code
                    tmp_bleu = -1
                if key not in ttl_bleu:
                    ttl_bleu[key] = [tmp_bleu]
                else:
                    ttl_bleu[key].append(tmp_bleu)
                ttl_bleu_idx.append([id_key, key, id_rule, tmp_bleu])
    # ttl_bleu_idx.sort(key=lambda x: x[3], reverse=True)

    print("len(ttl_bleu_idx_cnted): ", len(ttl_bleu_idx_cnted))
    print("cnt_valid_bleu: ", cnt_valid_bleu)
    final_recall = get_wrecall_simple(args, args.dataset_selection, ttl_bleu_idx_cnted, cnt_valid_bleu, pred_score13, pred_score14, pred_score19, pred_score20, thres13, thres14, thres19, thres20)
    print("final_recall: ", final_recall)
















if __name__ == "__main__":
    main()
