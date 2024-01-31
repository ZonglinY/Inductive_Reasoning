import argparse, os
import torch
import numpy as np
from new_metrics_validity import get_and_concat_selected_gene_and_human_eval_from_files
from utils import threshold_storer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bleu_n", type=int, default=4)
    # ./Checkpoints/baseline_template_with_random_filling/
    # ./Checkpoints_try/gptj_analysis_100test_newdata_newprompt
    # ./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10
    parser.add_argument("--output_dir", default="./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10/", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    # ./Data/DEERLET/setting0/
    # ./Data/DEERLET/
    # ./Data/DEERLET/random_fill_baseline/
    parser.add_argument("--root_data_dir", type=str, default="./Data/DEERLET/", help="data dir for current dataset")
    parser.add_argument("--if_include_all_related_files", type=int, default=1, help="if 1: automatically collect all selected pred_rule and human annotation files for calculation; if 0: use --setting_selection and --if_overlook_existing_annotations to specify single file for calculation.")
    # ======================
    parser.add_argument("--generator_model_type", type=str, default="gptj",
                        help="model type: gptneo125M/gptneo1.3B/gptneo2.7B/gptj/gptneox20B")
    parser.add_argument("--if_long_or_short_facts", type=int, default=1, help="when 0, use long facts to induce rules; when 1, use short facts to induce rules")
    parser.add_argument("--cnt_facts_as_input", type=int, default=3, help="can be 1/2/3, indicates how many facts to use to induce rules")
    parser.add_argument("--if_full_or_missing_facts", type=int, default=0, help="when 0, use full facts; when 1, only use part of the fact to induce rules")
    parser.add_argument("--setting_selection_M1_forM2M3", type=int, default=1, help="only works when --if_include_all_related_files is 0; used to identify which generated rules set to filter, useful when dataset_selection==13/14/15/16/17/18; current choices are 0/1")
    parser.add_argument("--setting_selection", type=int, default=2, help="only works when --if_include_all_related_files is 0; 0: zero-shot setting; 1: few-shot setting; 2: few-shot + chain of thought setting; 3: finetuning setting")
    parser.add_argument("--if_already_fintuned_for_test", type=int, default=0, help="always 0, unless when using finetuned checkpoint to only test")
    # ======================
    parser.add_argument("--if_overlook_existing_annotations", type=int, default=0, help="only works when --if_include_all_related_files is 0; the second number in the name of randomly selected generation files; for original definition please check randomly_select_generations_for_human_eval.py")
    parser.add_argument("--if_also_consider_train_val_annotation", type=int, default=0, help="0: only calculate correlation based on test annotations; 1: calculate correlation based on both train and test annotations.")
    parser.add_argument("--if_use_deer_train_data_for_test", type=int, default=0, help="only used in the file name of 'rule_proposer_generated_rules_'; Should be used when: 1. only used in --do_test but not --do_train; 2. only used when dataset_selection == 12; FUNCTION: rule proposer do test on deer train data (for annotation of train set of deerlet data)")
    parser.add_argument("--if_only_evaluate_on_nontrivial_data", type=int, default=0, help="1: only evaluate those with ground truth label if_trivial==0; else evaluate on every data.")
    parser.add_argument("--if_consider_M234", type=int, default=1, help="For our framework, when using M5, set it to 1; for baselines always set it to 0")
    # parser.add_argument("--if_consider_M5", type=int, default=1, help="For our framework, when using M5, set it to 1; for baselines always set it to 0")
    # parser.add_argument("--min_length_rule_to_be_considered", type=int, default=45, help="the min length of generated rule to be collected for human annotation; in the first 5 train files (train_human_eval_rlt_M1setting_1_0/1/2/3/4.pt) and first 2 test files (human_eval_rlt_M1setting_1_0/1.pt), the value of this hyperparameter is 0, while for others should be 45; this should be 0 for checkpoint gptj_analysis_100test_newdata_newprompt but 45 for gptj_analysis_100test_newdata_newprompt_10")
    args = parser.parse_args()

    assert args.setting_selection == 0 or args.setting_selection == 1 or args.setting_selection == 2 or args.setting_selection == 3
    assert args.if_include_all_related_files == 0 or args.if_include_all_related_files == 1
    assert args.if_also_consider_train_val_annotation == 0 or args.if_also_consider_train_val_annotation == 1
    assert args.if_already_fintuned_for_test == 0 or args.if_already_fintuned_for_test == 1
    # assert args.if_consider_M5 == 0 or args.if_consider_M5 == 1
    assert args.if_consider_M234 == 0 or args.if_consider_M234 == 1
    assert args.if_only_evaluate_on_nontrivial_data == 0 or args.if_only_evaluate_on_nontrivial_data == 1

    # if args.output_dir == "./Checkpoints_try/gptj_analysis_100test_newdata_newprompt/":
    #     allowed_existing_annotation_files_test=["0", "1"]
    # elif args.output_dir == "./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10/":
    #     # allowed_existing_annotation_files_test=["2", "3", "4"]
    #     allowed_existing_annotation_files_test=["3", "4"]
    # elif args.output_dir == "./Checkpoints/baseline_template_with_random_filling/":
    #     allowed_existing_annotation_files_test=["0"]
    # else:
    #     raise NotImplementError
    if "gptj_analysis_100test_newdata_newprompt_10" in args.output_dir:
        allowed_existing_annotation_files_test=["3", "4"]
    elif "baseline_template_with_random_filling" in args.output_dir:
        allowed_existing_annotation_files_test=["0"]
    elif "gptj_analysis_100test_newdata_newprompt" in args.output_dir:
        allowed_existing_annotation_files_test=["0", "1"]
    else:
        raise NotImplementError

    # full_generation
    if "baseline_template_with_random_filling" in args.output_dir:
        f_generation = os.path.join(args.output_dir, "rule_proposer_generated_rules.pt")
    else:
        f_generation = os.path.join(args.output_dir, "rule_proposer_generated_rules_{}_{}.pt".format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test))
    # f_generation = os.path.join(args.output_dir, "rule_proposer_generated_rules_{}.pt".format(args.setting_selection_M1_forM2M3))
    full_generation = torch.load(f_generation)
    print("len(full_generation): ", len(full_generation))
    # selected_gene: [[fact, true_rule, tmp_sel_rule, id_data]]
    if args.if_include_all_related_files:
        selected_gene, human_eval = get_and_concat_selected_gene_and_human_eval_from_files(args, allowed_existing_annotation_files_test)
        print("len(selected_gene): ", len(selected_gene))
        print("len(human_eval): ", len(human_eval))
    else:
        f_selected_generation = os.path.join(args.output_dir, "selection_generation_for_huaman_eval_M1setting_{}_{}.pt".format(args.setting_selection_M1_forM2M3, args.if_overlook_existing_annotations))
        selected_gene = torch.load(f_selected_generation)
        # human_eval
        f_human_eval = os.path.join(args.output_dir, "human_eval_rlt_M1setting_{}_{}.pt".format(args.setting_selection_M1_forM2M3, args.if_overlook_existing_annotations))
        human_eval = torch.load(f_human_eval)
    # module2_rlt, module3_rlt, module4_rlt: [#100[]]
    if args.if_consider_M234 == 1:
        if args.if_already_fintuned_for_test == 0:
            module2_path = os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3))
            if os.path.exists(module2_path):
                module2_rlt = torch.load(os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3)))
                module3_rlt = torch.load(os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3)))
                module4_rlt = torch.load(os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3)))
                module5_rlt = torch.load(os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3)))
            else:
                module2_rlt = torch.load(os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
                module3_rlt = torch.load(os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
                module4_rlt = torch.load(os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
                module5_rlt = torch.load(os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
        else:
            module2_rlt = torch.load(os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            module3_rlt = torch.load(os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            module4_rlt = torch.load(os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            module5_rlt = torch.load(os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))


    if not args.if_include_all_related_files:
        assert len(full_generation) == len(selected_gene)
        assert len(full_generation) == len(human_eval)
    assert len(selected_gene) == len(human_eval)
    if args.if_consider_M234 == 1:
        assert len(full_generation) == len(module2_rlt)
        assert len(full_generation) == len(module3_rlt)
        assert len(full_generation) == len(module4_rlt)
        assert len(full_generation) == len(module5_rlt)

    # threshold
    thres13, thres14, thres19, thres20 = threshold_storer(args)

    # get the index of selected_gene
    index_selected_gene = []
    for id_data in range(len(selected_gene)):
        fact, true_rule, tmp_sel_rule, idx = selected_gene[id_data]
        # assert idx == id_data
        tmp_sel_rule = tmp_sel_rule.strip('if ').strip('there exits ').strip('.').lower()
        # full_generation_this_idx
        if_found_idx = 0
        full_generation_this_idx = full_generation[idx]
        if args.if_consider_M234 == 1:
            m2_rlt_this_idx = module2_rlt[idx]
            if not len(full_generation_this_idx) == len(m2_rlt_this_idx):
                print("idx: ", idx)
                print("id_data: ", id_data)
                print("len(full_generation_this_idx): ", len(full_generation_this_idx))
                print("len(m2_rlt_this_idx): ", len(m2_rlt_this_idx))
                raise Exception
        for id_gene, tmp_gene in enumerate(full_generation_this_idx):
            if tmp_sel_rule in tmp_gene.lower():
                index_selected_gene.append(id_gene)
                if_found_idx += 1
                break
        if not if_found_idx != 0:
            print("idx: ", idx)
            print("id_data: ", id_data)
            print("tmp_sel_rule: ", tmp_sel_rule)
            print("full_generation_this_idx: ", full_generation_this_idx)
            raise Exception
    assert len(selected_gene) == len(index_selected_gene)

    if args.if_consider_M234 == 1:
        # get the prediction of m2/m3/m4 on selected_gene
        m2_rlt_selected, m3_rlt_selected, m4_rlt_selected, m5_rlt_selected = [], [], [], []
        for id_data in range(len(selected_gene)):
            fact, true_rule, tmp_sel_rule, idx = selected_gene[id_data]
            # print("id_data: {}; idx: {}; len(module2_rlt): {}; len(index_selected_gene): {}; index_selected_gene[id_data]: {}; len(module2_rlt[idx]): {}".format(id_data, idx, len(module2_rlt), len(index_selected_gene), index_selected_gene[id_data], len(module2_rlt[idx])))
            m2_rlt_selected.append(module2_rlt[idx][index_selected_gene[id_data]])
            m3_rlt_selected.append(module3_rlt[idx][index_selected_gene[id_data]])
            m4_rlt_selected.append(module4_rlt[idx][index_selected_gene[id_data]])
            m5_rlt_selected.append(module5_rlt[idx][index_selected_gene[id_data]])

    ## true precision -- human evaluation
    # get the full mean of human evaluation
    ave_human_eval_m1, ave_human_eval_m12, ave_human_eval_m13, ave_human_eval_m14, ave_human_eval_m15, ave_human_eval_m12345 = [], [], [], [], [], []
    ave_consistent_m1, ave_consistent_m12, ave_consistent_m13, ave_consistent_m14, ave_consistent_m15, ave_consistent_m12345 = [], [], [], [], [], []
    ave_commonsense_m1, ave_commonsense_m12, ave_commonsense_m13, ave_commonsense_m14, ave_commonsense_m15, ave_commonsense_m12345 = [], [], [], [], [], []
    ave_general_m1, ave_general_m12, ave_general_m13, ave_general_m14, ave_general_m15, ave_general_m12345 = [], [], [], [], [], []
    ave_trivial_m1, ave_trivial_m12, ave_trivial_m13, ave_trivial_m14, ave_trivial_m15, ave_trivial_m12345 = [], [], [], [], [], []
    ave_quality_m1, ave_quality_m12, ave_quality_m13, ave_quality_m14, ave_quality_m15, ave_quality_m12345 = [], [], [], [], [], []

    # list_cnt_modules: [#M1 w/o M5, #M1 w/o M5, #M1 w/ M2, #M1 w/ M3, #M1 w/ M4, #M1 w/ M5, #M1 w/ M2345]
    list_cnt_modules = [0, 0, 0, 0, 0, 0, 0]
    for id_data in range(len(human_eval)):
        if_general, if_consistent, if_commonsense, if_trivial, if_natural_language = human_eval[id_data]
        if_general, if_consistent, if_commonsense, if_trivial, if_natural_language = int(if_general), int(if_consistent), int(if_commonsense), int(if_trivial), int(if_natural_language)
        if_general_for_correctness = if_general / 2
        if_consistent_for_correctness = if_consistent / 2
        if_commonsense_for_correctness = if_commonsense / 2
        if_trivial_for_correctness = if_trivial
        if_natural_language_for_correctness = if_natural_language / 5
        # ave_human_eval = if_general_for_correctness * if_consistent_for_correctness * if_commonsense_for_correctness * if_trivial_for_correctness * if_natural_language_for_correctness
        ave_human_eval = if_general_for_correctness * if_consistent_for_correctness * if_commonsense_for_correctness * if_trivial_for_correctness
        if args.if_only_evaluate_on_nontrivial_data == 1:
            if_counted = True if if_trivial > 0 else False
        else:
            if_counted = True
        if if_counted:
            # we use a binary eval for ave_human_eval
            if ave_human_eval > 0:
                # ave_human_eval_discrete = 1
                ave_human_eval = 1
            else:
                # ave_human_eval_discrete = 0
                assert ave_human_eval == 0
            ## if binary eval for if_general, if_consistent, and if_commonsense
            if if_general > 0:
                if_general = 1
            if if_consistent > 0:
                if_consistent = 1
            if if_commonsense > 0:
                if_commonsense = 1
            ## if do not pay attention to partially true data
            # if if_general == 1:
            #     if_general = 0
            # if if_consistent == 1:
            #     if_consistent = 0
            # if if_commonsense == 1:
            #     if_commonsense = 0
            ## if_general, if_consistent, if_commonsense are used in a 3-point scale from 1~3, not from 0~2; if_trivial are used in a 2-point scale from 0~1
            # if_general += 1
            # if_consistent += 1
            # if_commonsense += 1
            # if_trivial += 1
            list_cnt_modules[0] += 1

            ave_human_eval_m1.append(ave_human_eval)
            ave_consistent_m1.append(if_consistent)
            ave_commonsense_m1.append(if_commonsense)
            ave_general_m1.append(if_general)
            ave_trivial_m1.append(if_trivial)
            ave_quality_m1.append(if_natural_language)

            list_cnt_modules[1] += 1
            if args.if_consider_M234 == 1:
                if m2_rlt_selected[id_data] > thres13:
                    ave_human_eval_m12.append(ave_human_eval)
                    ave_consistent_m12.append(if_consistent)
                    ave_commonsense_m12.append(if_commonsense)
                    ave_general_m12.append(if_general)
                    ave_trivial_m12.append(if_trivial)
                    ave_quality_m12.append(if_natural_language)
                    list_cnt_modules[2] += 1

                if m3_rlt_selected[id_data] > thres14:
                    ave_human_eval_m13.append(ave_human_eval)
                    ave_consistent_m13.append(if_consistent)
                    ave_commonsense_m13.append(if_commonsense)
                    ave_general_m13.append(if_general)
                    ave_trivial_m13.append(if_trivial)
                    ave_quality_m13.append(if_natural_language)
                    list_cnt_modules[3] += 1

                if m4_rlt_selected[id_data] > thres19:
                    ave_human_eval_m14.append(ave_human_eval)
                    ave_consistent_m14.append(if_consistent)
                    ave_commonsense_m14.append(if_commonsense)
                    ave_general_m14.append(if_general)
                    ave_trivial_m14.append(if_trivial)
                    ave_quality_m14.append(if_natural_language)
                    list_cnt_modules[4] += 1

                if m5_rlt_selected[id_data] > thres20:
                    ave_human_eval_m15.append(ave_human_eval)
                    ave_consistent_m15.append(if_consistent)
                    ave_commonsense_m15.append(if_commonsense)
                    ave_general_m15.append(if_general)
                    ave_trivial_m15.append(if_trivial)
                    ave_quality_m15.append(if_natural_language)
                    list_cnt_modules[5] += 1

                if m2_rlt_selected[id_data] > thres13 and m3_rlt_selected[id_data] > thres14 and m4_rlt_selected[id_data] > thres19 and m5_rlt_selected[id_data] > thres20:
                    ave_human_eval_m12345.append(ave_human_eval)
                    ave_consistent_m12345.append(if_consistent)
                    ave_commonsense_m12345.append(if_commonsense)
                    ave_general_m12345.append(if_general)
                    ave_trivial_m12345.append(if_trivial)
                    ave_quality_m12345.append(if_natural_language)
                    list_cnt_modules[6] += 1
                    # print("selected_gene[id_data][2]: ", selected_gene[id_data][2])


    human_eval_precision_m1 = np.mean(ave_human_eval_m1)
    human_eval_precision_m12 = np.mean(ave_human_eval_m12)
    human_eval_precision_m13 = np.mean(ave_human_eval_m13)
    human_eval_precision_m14 = np.mean(ave_human_eval_m14)
    human_eval_precision_m15 = np.mean(ave_human_eval_m15)
    human_eval_precision_m12345 = np.mean(ave_human_eval_m12345)


    print("=============== COUNTS ===============")
    print("count M1: {:.0f}".format(list_cnt_modules[0]))
    print("count M1: {:.0f}".format(list_cnt_modules[1]))
    print("count M1 + M2: {:.0f}".format(list_cnt_modules[2]))
    print("count M1 + M3: {:.0f}".format(list_cnt_modules[3]))
    print("count M1 + M4: {:.0f}".format(list_cnt_modules[4]))
    print("count M1 + M5: {:.0f}".format(list_cnt_modules[5]))
    print("count M1 + M2345: {:.0f}".format(list_cnt_modules[6]))

    print("=============== PRECISION ===============")
    print("human_eval_precision_m1: {:.3f}".format(human_eval_precision_m1))
    print("human_eval_precision_m12: {:.3f}".format(human_eval_precision_m12))
    print("human_eval_precision_m13: {:.3f}".format(human_eval_precision_m13))
    print("human_eval_precision_m14: {:.3f}".format(human_eval_precision_m14))
    print("human_eval_precision_m15: {:.3f}".format(human_eval_precision_m15))
    print("human_eval_precision_m12345: {:.3f}".format(human_eval_precision_m12345))

    ## true recall -- human evaluation
    num_true_rule_m1 = np.sum(ave_human_eval_m1)
    num_true_rule_m12 = np.sum(ave_human_eval_m12)
    num_true_rule_m13 = np.sum(ave_human_eval_m13)
    num_true_rule_m14 = np.sum(ave_human_eval_m14)
    num_true_rule_m15 = np.sum(ave_human_eval_m15)
    num_true_rule_m12345 = np.sum(ave_human_eval_m12345)

    # print("num_true_rule_m1: ", num_true_rule_m1)

    human_eval_recall_m1 = num_true_rule_m1/num_true_rule_m1
    human_eval_recall_m12 = num_true_rule_m12/num_true_rule_m1
    human_eval_recall_m13 = num_true_rule_m13/num_true_rule_m1
    human_eval_recall_m14 = num_true_rule_m14/num_true_rule_m1
    human_eval_recall_m15 = num_true_rule_m15/num_true_rule_m1
    human_eval_recall_m12345 = num_true_rule_m12345/num_true_rule_m1

    print("=============== RECALL ===============")
    print("human_eval_recall_m1: {:.3f}".format(human_eval_recall_m1))
    print("human_eval_recall_m12: {:.3f}".format(human_eval_recall_m12))
    print("human_eval_recall_m13: {:.3f}".format(human_eval_recall_m13))
    print("human_eval_recall_m14: {:.3f}".format(human_eval_recall_m14))
    print("human_eval_recall_m15: {:.3f}".format(human_eval_recall_m15))
    print("human_eval_recall_m12345: {:.3f}".format(human_eval_recall_m12345))

    def get_f1(precision, recall):
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    human_eval_f1_m1 = get_f1(human_eval_precision_m1, human_eval_recall_m1)
    human_eval_f1_m12 = get_f1(human_eval_precision_m12, human_eval_recall_m12)
    human_eval_f1_m13 = get_f1(human_eval_precision_m13, human_eval_recall_m13)
    human_eval_f1_m14 = get_f1(human_eval_precision_m14, human_eval_recall_m14)
    human_eval_f1_m15 = get_f1(human_eval_precision_m15, human_eval_recall_m15)
    human_eval_f1_m12345 = get_f1(human_eval_precision_m12345, human_eval_recall_m12345)

    print("=============== F1 ===============")
    print("human_eval_f1_m1: {:.3f}".format(human_eval_f1_m1))
    print("human_eval_f1_m12: {:.3f}".format(human_eval_f1_m12))
    print("human_eval_f1_m13: {:.3f}".format(human_eval_f1_m13))
    print("human_eval_f1_m14: {:.3f}".format(human_eval_f1_m14))
    print("human_eval_f1_m15: {:.3f}".format(human_eval_f1_m15))
    print("human_eval_f1_m12345: {:.3f}".format(human_eval_f1_m12345))


    human_eval_consistent_m1 = np.mean(ave_consistent_m1)
    human_eval_consistent_m12 = np.mean(ave_consistent_m12)
    human_eval_consistent_m13 = np.mean(ave_consistent_m13)
    human_eval_consistent_m14 = np.mean(ave_consistent_m14)
    human_eval_consistent_m15 = np.mean(ave_consistent_m15)
    human_eval_consistent_m12345 = np.mean(ave_consistent_m12345)

    print("=============== Consistent ===============")
    print("human_eval_consistent_m1: {:.3f}".format(human_eval_consistent_m1))
    print("human_eval_consistent_m12: {:.3f}".format(human_eval_consistent_m12))
    print("human_eval_consistent_m13: {:.3f}".format(human_eval_consistent_m13))
    print("human_eval_consistent_m14: {:.3f}".format(human_eval_consistent_m14))
    print("human_eval_consistent_m15: {:.3f}".format(human_eval_consistent_m15))
    print("human_eval_consistent_m12345: {:.3f}".format(human_eval_consistent_m12345))


    human_eval_commonsense_m1 = np.mean(ave_commonsense_m1)
    human_eval_commonsense_m12 = np.mean(ave_commonsense_m12)
    human_eval_commonsense_m13 = np.mean(ave_commonsense_m13)
    human_eval_commonsense_m14 = np.mean(ave_commonsense_m14)
    human_eval_commonsense_m15 = np.mean(ave_commonsense_m15)
    human_eval_commonsense_m12345 = np.mean(ave_commonsense_m12345)

    print("=============== Commonsense ===============")
    print("human_eval_commonsense_m1: {:.3f}".format(human_eval_commonsense_m1))
    print("human_eval_commonsense_m12: {:.3f}".format(human_eval_commonsense_m12))
    print("human_eval_commonsense_m13: {:.3f}".format(human_eval_commonsense_m13))
    print("human_eval_commonsense_m14: {:.3f}".format(human_eval_commonsense_m14))
    print("human_eval_commonsense_m15: {:.3f}".format(human_eval_commonsense_m15))
    print("human_eval_commonsense_m12345: {:.3f}".format(human_eval_commonsense_m12345))


    human_eval_general_m1 = np.mean(ave_general_m1)
    human_eval_general_m12 = np.mean(ave_general_m12)
    human_eval_general_m13 = np.mean(ave_general_m13)
    human_eval_general_m14 = np.mean(ave_general_m14)
    human_eval_general_m15 = np.mean(ave_general_m15)
    human_eval_general_m12345 = np.mean(ave_general_m12345)

    print("=============== General ===============")
    print("human_eval_general_m1: {:.3f}".format(human_eval_general_m1))
    print("human_eval_general_m12: {:.3f}".format(human_eval_general_m12))
    print("human_eval_general_m13: {:.3f}".format(human_eval_general_m13))
    print("human_eval_general_m14: {:.3f}".format(human_eval_general_m14))
    print("human_eval_general_m15: {:.3f}".format(human_eval_general_m15))
    print("human_eval_general_m12345: {:.3f}".format(human_eval_general_m12345))


    human_eval_trivial_m1 = np.mean(ave_trivial_m1)
    human_eval_trivial_m12 = np.mean(ave_trivial_m12)
    human_eval_trivial_m13 = np.mean(ave_trivial_m13)
    human_eval_trivial_m14 = np.mean(ave_trivial_m14)
    human_eval_trivial_m15 = np.mean(ave_trivial_m15)
    human_eval_trivial_m12345 = np.mean(ave_trivial_m12345)

    print("=============== Trivial ===============")
    print("human_eval_trivial_m1: {:.3f}".format(human_eval_trivial_m1))
    print("human_eval_trivial_m12: {:.3f}".format(human_eval_trivial_m12))
    print("human_eval_trivial_m13: {:.3f}".format(human_eval_trivial_m13))
    print("human_eval_trivial_m14: {:.3f}".format(human_eval_trivial_m14))
    print("human_eval_trivial_m15: {:.3f}".format(human_eval_trivial_m15))
    print("human_eval_trivial_m12345: {:.3f}".format(human_eval_trivial_m12345))


    human_eval_quality_m1 = np.mean(ave_quality_m1)
    human_eval_quality_m12 = np.mean(ave_quality_m12)
    human_eval_quality_m13 = np.mean(ave_quality_m13)
    human_eval_quality_m14 = np.mean(ave_quality_m14)
    human_eval_quality_m15 = np.mean(ave_quality_m15)
    human_eval_quality_m12345 = np.mean(ave_quality_m12345)

    print("=============== Quality ===============")
    print("human_eval_quality_m1: {:.3f}".format(human_eval_quality_m1))
    print("human_eval_quality_m12: {:.3f}".format(human_eval_quality_m12))
    print("human_eval_quality_m13: {:.3f}".format(human_eval_quality_m13))
    print("human_eval_quality_m14: {:.3f}".format(human_eval_quality_m14))
    print("human_eval_quality_m15: {:.3f}".format(human_eval_quality_m15))
    print("human_eval_quality_m12345: {:.3f}".format(human_eval_quality_m12345))
























if __name__ == '__main__':
    main()
