import argparse, os, math
import torch
import numpy as np
import nltk
from nltk import bleu
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from utils import load_data_Hypothetical_Induction_Module123, whether_not_included_in_in_context_demonstrations_in_rule_proposer, threshold_storer
from recall_star_calculator import get_wrecall_simple
# TreebankWordDetokenizer: reverse word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def remove_stop_words_nltk(e2):
    stop_words = set(stopwords.words('english'))
    e2_tokens = word_tokenize(e2)
    e2_tokens_NoStopWords = []
    for tmp_i, tmp_word in enumerate(e2_tokens):
        if tmp_word not in stop_words:
            e2_tokens_NoStopWords.append(tmp_word)

    e2_NoStopWords = TreebankWordDetokenizer().detokenize(e2_tokens_NoStopWords)
    return e2_NoStopWords


# cur_list: a list of list
def sum_an_element_from_listoflist(cur_list, id_element):
    ttl_value = 0
    for elements in cur_list:
        ttl_value += elements[id_element]

    return ttl_value


# FUNCTION:
#   a child function in while loop in get_bleu_green_analysis;
#   we need to note BLEU_rule_template and BLEU_rule_template_nofilter twice with the same code, so we abstract it as a function here
def BLEU_noter(id_key, key, id_rule, rule, tmp_bleu, dict_topicid2ruleFormid, generated_rule_notes, BLEU_full_each_key, BLEU_rule_template, BLEU_topics, BLEU_specific_general):
    BLEU_full_each_key.append([id_key, key, id_rule, tmp_bleu])
    # # for analysis
    cur_notes = generated_rule_notes[key][rule]
    for tmp_note in cur_notes:
        tmp_rule_template_id, tmp_topic_id, tmp_specific_general_fact_id, tmp_long_short_facts_id, tmp_cnt_facts_id, tmp_full_missing_facts_id = tmp_note
        tmp_ruleform_id = dict_topicid2ruleFormid[tmp_topic_id]
        # BLEU_rule_template
        if tmp_rule_template_id not in BLEU_rule_template:
            BLEU_rule_template[tmp_rule_template_id] = [[id_key, key, id_rule, tmp_bleu]]
        else:
            BLEU_rule_template[tmp_rule_template_id].append([id_key, key, id_rule, tmp_bleu])
        # BLEU_topics
        if tmp_topic_id not in BLEU_topics:
            BLEU_topics[tmp_topic_id] = [[id_key, key, id_rule, tmp_bleu]]
        else:
            BLEU_topics[tmp_topic_id].append([id_key, key, id_rule, tmp_bleu])
        # BLEU_specific_general
        if tmp_ruleform_id not in BLEU_specific_general:
            BLEU_specific_general[tmp_ruleform_id] = {tmp_specific_general_fact_id: [[id_key, key, id_rule, tmp_bleu]]}
        elif tmp_specific_general_fact_id not in BLEU_specific_general[tmp_ruleform_id]:
            BLEU_specific_general[tmp_ruleform_id][tmp_specific_general_fact_id] = [[id_key, key, id_rule, tmp_bleu]]
        else:
            BLEU_specific_general[tmp_ruleform_id][tmp_specific_general_fact_id].append([id_key, key, id_rule, tmp_bleu])
    return BLEU_full_each_key, BLEU_rule_template, BLEU_topics, BLEU_specific_general


def get_bleu_green_analysis(args, dataset_selection, bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt=None, module3_rlt=None, module4_rlt=None, module5_rlt=None, thres13=None, thres14=None, thres19=None, thres20=None):
    # assert dataset_selection == 12 or dataset_selection == 13 or dataset_selection == 14 or dataset_selection == 19 or dataset_selection == 1314 or dataset_selection == 1419 or dataset_selection == 1319 or dataset_selection == 131419
    if args.which_metric == 0:
        def score(hyp, refs):
            # print(hyp)
            # print(refs)
            n = bleu_n
            weights = [1/n] * n
            return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)
    elif args.which_metric == 1:
        def score(hyp, ref):
            return single_meteor_score(hyp, ref)
    else:
        raise NotImplementError
    # _nofilter: do not filter with M2/3/4/5; used to calculate recall
    # BLEU_rule_template: {'template_id': [bleu0, bleu1, ...]}
    # BLEU_rule_template = {}
    BLEU_rule_template = {0: [], 1: [], 2: [], 3: []}
    BLEU_rule_template_nofilter = {0: [], 1: [], 2: [], 3: []}
    # BLEU_topics: {'template_id': [bleu0, bleu1, ...]}
    BLEU_topics = {}
    BLEU_topics_nofilter = {}
    # BLEU_specific_general: {'ruleform_id': {'0': [bleu0, bleu1, ...]}}
    BLEU_specific_general = {}
    BLEU_specific_general_nofilter = {}
    # generated_rule_notes: {'0': {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}
    cnt = 0
    cnt_skipper = 0
    # here key in generated_rule corresponds to the datalineid in test_datasets
    # for key in generated_rule:
    for id_key, key in enumerate(sorted(generated_rule.keys())):
        golden_rules = dict_id2trueRule_test[key]
        BLEU_full_each_key, BLEU_full_each_key_nofilter = [], []
        gene_rules = generated_rule[key]
        # if_should_keep
        if len(golden_rules) == 1:
            if_should_keep = whether_not_included_in_in_context_demonstrations_in_rule_proposer(golden_rules[0])
        else:
            for tmp_id in range(len(golden_rules)):
                if_should_keep = whether_not_included_in_in_context_demonstrations_in_rule_proposer(golden_rules[tmp_id])
                if if_should_keep == 0:
                    break
        if not if_should_keep:
            cnt_skipper += 1
            print("golden_rules: ", golden_rules)
        elif args.if_consider_M234:
            m2_rlt_for_gene_rules = module2_rlt[key]
            m3_rlt_for_gene_rules = module3_rlt[key]
            m4_rlt_for_gene_rules = module4_rlt[key]
            m5_rlt_for_gene_rules = module5_rlt[key]
            assert len(gene_rules) == len(m2_rlt_for_gene_rules)
            assert len(gene_rules) == len(m3_rlt_for_gene_rules)
            assert len(gene_rules) == len(m4_rlt_for_gene_rules)
            assert len(gene_rules) == len(m5_rlt_for_gene_rules)
        for id_rule, rule in enumerate(gene_rules):
            # print("gene_rule: ", rule)
            # args.min_length_rule_to_be_considered
            if len(rule) < args.min_length_rule_to_be_considered:
                print("#############rule less than 45")
                continue
            # if_keep_in_record
            if_keep_in_record = True
            if dataset_selection == 12:
                pass
            elif args.if_consider_M234 == 1:
                if '13' in str(dataset_selection):
                    if float(m2_rlt_for_gene_rules[id_rule]) <= thres13:
                        if_keep_in_record = False
                if '14' in str(dataset_selection):
                    if float(m3_rlt_for_gene_rules[id_rule]) <= thres14:
                        if_keep_in_record = False
                if '19' in str(dataset_selection):
                    if float(m4_rlt_for_gene_rules[id_rule]) <= thres19:
                        if_keep_in_record = False
                if '20' in str(dataset_selection):
                    if float(m5_rlt_for_gene_rules[id_rule]) <= thres20:
                        if_keep_in_record = False
            else:
                raise NotImplementError
            # begin calculate bleu
            if rule.lower().strip()[0:2] == golden_rules[0].lower().strip()[0:2]:
            # if rule.lower().strip()[0:2] == golden_rules[0].lower().strip()[0:2] and rule.lower().strip()[0:2] == 'if':
                if args.if_remove_stop_words == 1:
                    for_score_rule = remove_stop_words_nltk(rule)
                    for_score_golden_rules = remove_stop_words_nltk(golden_rules[0])
                    for_score_golden_rules = [for_score_golden_rules]
                else:
                    for_score_rule = rule
                    for_score_golden_rules = golden_rules
                if args.which_metric == 0:
                    tmp_bleu = score(for_score_rule.lower().strip().strip('.').split(), [t.lower().strip().strip('.').split() for t in for_score_golden_rules])
                elif args.which_metric == 1:
                    assert len(golden_rules) == 1
                    tmp_bleu = score(for_score_rule.lower().strip().strip('.').split(), for_score_golden_rules[0].lower().strip().strip('.').split())
                else:
                    raise NotImplementError
                ## note BLEU_xxx for calculating recall (without filtering, note all)
                BLEU_full_each_key_nofilter, BLEU_rule_template_nofilter, BLEU_topics_nofilter, BLEU_specific_general_nofilter = BLEU_noter(id_key, key, id_rule, rule, tmp_bleu, dict_topicid2ruleFormid, generated_rule_notes, BLEU_full_each_key_nofilter, BLEU_rule_template_nofilter, BLEU_topics_nofilter, BLEU_specific_general_nofilter)
                ## note BLEU_xxx for calculating BLEU (with filtering, only note those with if_keep_in_record == True)
                if if_keep_in_record:
                    BLEU_full_each_key, BLEU_rule_template, BLEU_topics, BLEU_specific_general = BLEU_noter(id_key, key, id_rule, rule, tmp_bleu, dict_topicid2ruleFormid, generated_rule_notes, BLEU_full_each_key, BLEU_rule_template, BLEU_topics, BLEU_specific_general)
    print("cnt_test_data_that_in_icd: ", cnt_skipper)
    print("len(BLEU_topics): ", sum([len(BLEU_topics[i]) for i in range(len(BLEU_topics))]))

    # # wrecall
    # if args.if_consider_M234:
    #     wrecall_full = get_wrecall_simple(args, dataset_selection, BLEU_full_each_key_nofilter, len(BLEU_full_each_key_nofilter), module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
    #     print("wrecall_full: ", wrecall_full)

    print("===========BLEU_rule_form")
    ttl_bleu_score_FOL_math = [[], []]
    ttl_bleu_score_FOL_math_nofilter = [[], []]
    cnt_bleu_score_FOL_math = [0, 0]
    # ttl_bleu_score_FOL_math, cnt_bleu_score_FOL_math
    for key in BLEU_topics:
        idx_ruleform = dict_topicid2ruleFormid[key]
        ## ttl_bleu_score_FOL_math
        # ttl_bleu_score_FOL_math[idx_ruleform] += sum(BLEU_topics[key])
        for value in BLEU_topics[key]:
            ttl_bleu_score_FOL_math[idx_ruleform].append(value)
        ## cnt_bleu_score_FOL_math
        # += sum_an_element_from_listoflist(BLEU_topics[key], 3)
        cnt_bleu_score_FOL_math[idx_ruleform] += len(BLEU_topics[key])
        ## ttl_bleu_score_FOL_math_nofilter
        for value in BLEU_topics_nofilter[key]:
            ttl_bleu_score_FOL_math_nofilter[idx_ruleform].append(value)

    print("Overall bleu: {:.4f}".format((sum_an_element_from_listoflist(ttl_bleu_score_FOL_math[0], 3) + sum_an_element_from_listoflist(ttl_bleu_score_FOL_math[1], 3)) / sum(cnt_bleu_score_FOL_math)))
    for idx_ruleform, ttl_bleu_ruleform in enumerate(ttl_bleu_score_FOL_math):
        if cnt_bleu_score_FOL_math[idx_ruleform] > 0:
            mean_bleu_for_key = sum_an_element_from_listoflist(ttl_bleu_ruleform, 3) / cnt_bleu_score_FOL_math[idx_ruleform]
            wrecall_for_key = get_wrecall_simple(args, dataset_selection, ttl_bleu_score_FOL_math_nofilter[idx_ruleform], len(ttl_bleu_score_FOL_math_nofilter[idx_ruleform]), module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
            # print("len(ttl_bleu_score_FOL_math_nofilter[idx_ruleform]): ", len(ttl_bleu_score_FOL_math_nofilter[idx_ruleform]))
        else:
            mean_bleu_for_key = 0
            wrecall_for_key = 0
        cur_green = math.sqrt(mean_bleu_for_key * 100 * wrecall_for_key)
        print("rule form: {}; bleu: {:.4f}; wrecall: {:.3f}; green: {:.2f}".format(idx_ruleform, mean_bleu_for_key, wrecall_for_key, cur_green))

    print("===========BLEU_rule_template")
    for key in BLEU_rule_template:
        if len(BLEU_rule_template[key]) > 0:
            # mean_bleu_for_key = sum(BLEU_rule_template[key]) / len(BLEU_rule_template[key])
            mean_bleu_for_key = sum_an_element_from_listoflist(BLEU_rule_template[key], 3) / len(BLEU_rule_template[key])
            wrecall_for_key = get_wrecall_simple(args, dataset_selection, BLEU_rule_template_nofilter[key], len(BLEU_rule_template_nofilter[key]), module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
            cur_green = math.sqrt(mean_bleu_for_key * 100 * wrecall_for_key)
            print("key: {}; bleu: {:.4f}; wrecall: {:.3f}; green: {:.2f}".format(dict_id2ruleTemplate[key], mean_bleu_for_key, wrecall_for_key, cur_green))
        else:
            print("key: {}; there's no such rules left!".format(dict_id2ruleTemplate[key]))
    # mean_bleu_for_if
    if len(BLEU_rule_template[0]) + len(BLEU_rule_template[1]) + len(BLEU_rule_template[3]) > 0:
        # mean_bleu_for_if = (sum(BLEU_rule_template[0]) + sum(BLEU_rule_template[1]) + sum(BLEU_rule_template[3])) / (len(BLEU_rule_template[0]) + len(BLEU_rule_template[1]) + len(BLEU_rule_template[3]))
        mean_bleu_for_if = (sum_an_element_from_listoflist(BLEU_rule_template[0], 3) + sum_an_element_from_listoflist(BLEU_rule_template[1], 3) + sum_an_element_from_listoflist(BLEU_rule_template[3], 3)) / (len(BLEU_rule_template[0]) + len(BLEU_rule_template[1]) + len(BLEU_rule_template[3]))
        print("mean_bleu_for_if: {:.4f}".format(mean_bleu_for_if))
    else:
        # print("mean_bleu_for_if: {:.4f}".format(mean_bleu_for_if))
        print("mean_bleu_for_if: there's no if rules left!")
    # mean_bleu_for_there
    if len(BLEU_rule_template[2]) > 0:
        # mean_bleu_for_there = sum(BLEU_rule_template[2]) / len(BLEU_rule_template[2])
        mean_bleu_for_there = sum_an_element_from_listoflist(BLEU_rule_template[2], 3) / len(BLEU_rule_template[2])
        # wrecall_for_key = get_wrecall_simple(dataset_selection, BLEU_rule_template[2], len(BLEU_rule_template[2]), module2_rlt, module3_rlt, module4_rlt, thres13, thres14, thres19)
        # cur_green = math.sqrt(mean_bleu_for_there * 100 * wrecall_for_key)
        print("mean_bleu_for_there: {:.4f}".format(mean_bleu_for_there))
    else:
        print("mean_bleu_for_there: there's no there rules left!")

    print("===========BLEU_topics")
    for key in BLEU_topics:
        # mean_bleu_for_key = sum(BLEU_topics[key]) / len(BLEU_topics[key])
        mean_bleu_for_key = sum_an_element_from_listoflist(BLEU_topics[key], 3) / len(BLEU_topics[key])
        wrecall_for_key = get_wrecall_simple(args, dataset_selection, BLEU_topics_nofilter[key], len(BLEU_topics_nofilter[key]), module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
        green = math.sqrt(mean_bleu_for_key * 100 * wrecall_for_key)
        print("key: {}; bleu: {:.4f}; wrecall: {:.3f}; green: {:.2f}".format(dict_id2topic[key], mean_bleu_for_key, wrecall_for_key, green))

    print("===========BLEU_specific_general")
    # print("BLEU_specific_general: ", BLEU_specific_general)
    # print("BLEU_specific_general.keys(): ", BLEU_specific_general.keys())
    for idx_ruleform in BLEU_specific_general:
        # print("BLEU_specific_general[idx_ruleform].keys(): ", BLEU_specific_general[idx_ruleform].keys())
        for idx_specific_general in BLEU_specific_general[idx_ruleform]:
            # mean_bleu_for_key = sum(BLEU_specific_general[idx_ruleform][idx_specific_general]) / len(BLEU_specific_general[idx_ruleform][idx_specific_general])
            mean_bleu_for_key = sum_an_element_from_listoflist(BLEU_specific_general[idx_ruleform][idx_specific_general], 3) / len(BLEU_specific_general[idx_ruleform][idx_specific_general])
            wrecall_for_key = get_wrecall_simple(args, dataset_selection, BLEU_specific_general_nofilter[idx_ruleform][idx_specific_general], len(BLEU_specific_general_nofilter[idx_ruleform][idx_specific_general]), module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
            cur_green = math.sqrt(mean_bleu_for_key * 100 * wrecall_for_key)
            print("rule form: {}; if_specific_general: {:.4f}; bleu: {:.4f}; wrecall: {:.3f}; green: {:.2f}".format(idx_ruleform, idx_specific_general, mean_bleu_for_key, wrecall_for_key, cur_green))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_selection", type=float, default=12, help="0~4: standard ParaRules Mod0~4; 5: raw inductive reasoning dataset (contain bug); 6: inductive reasoning dataset with no synonym; 7: inductive reasoning dataset with half synonym; 8: inductive reasoning dataset with full synonym; 9: Module 1 (generate rules that can explain the given event); 9.5: Module 1 with retrieval; 10: Module 2 (predict whether the rule can exolain/casual the given event); 11: Module 3 (predict whether the rule is possible to happen or has already happened); 12: Rule Proposer; 13: Deduction Consistency Evaluator; 14: Indiscriminate Comfirmation Handler")
    parser.add_argument("--root_data_dir", type=str, default="~/openWorld_Analysis_Inductive_Reasoning_PLM/Data/", help="data dir for current dataset")
    parser.add_argument("--num_gene_times", type=int, default=1, help="call generate() num_gene_times times for each input sentence; basically num_gene_times has the same target with num_return_sequences, but can be implemented in a GPU-restriced way.; only be used when args.dataset_selection == 12")
    parser.add_argument("--bleu_n", type=int, default=4)
    # for load_data_Hypothetical_Induction_Module123, although they will not affect results of this .py file
    # parser.add_argument("--if_specific_or_general_facts", type=int, default=0, help="when 0, only use specific facts to induce rules and get results; when 1, only use general facts to induce rules and get results")
    parser.add_argument("--if_use_deer_train_data_for_test", type=int, default=0, help="should be used when: 1. only used in --do_test but not --do_train; 2. only used when dataset_selection == 12; FUNCTION: rule proposer do test on deer train data (for annotation of train set of deerlet data)")
    # /home/v-zonyang/Checkpoints/baseline_template_with_random_filling/
    # gptj_analysis_100test_newdata_newprompt
    # gptj_analysis_100test_newdata_newprompt_shortfact_1fact_fullfact
    # gptj_analysis_100test_newdata_newprompt_longfact_1fact_fullfact
    # gptj_analysis_100test_newdata_newprompt_shortfact_3fact_missingfact
    # gptj_analysis_100test_newdata_newprompt_2
    # gptj_analysis_100test_newdata_newprompt_longfact_1fact_fullfact_2
    # gptj_analysis_100test_newdata_newprompt_10
    # gptj_analysis_100test_newdata_newprompt_shortfact_2fact_fullfact_10
    # gptj_analysis_100test_newdata_newprompt_longfact_1fact_fullfact_10
    # gptj_analysis_100test_newdata_newprompt_shortfact_3fact_missingfact_10
    # ./Checkpoints/gptj_analysis_100test_newdata_newprompt_Gptneo125M
    # gptj_analysis_100test_newdata_newprompt_Gptneo2.7B_numgene10
    # gptj_analysis_100test_newdata_newprompt_Gptneox20B_numgene10_R2
    # gptj_analysis_100test_newdata_newprompt_Gptneo2.7B_numgene10_R3
    # gptj_analysis_100test_newdata_newprompt_Gptneo1.3B_numgene10
    # ======================
    # baseline_template_with_random_filling
    # baseline_template_with_random_filling_short_2_full
    # baseline_template_with_random_filling_short_1_full
    # baseline_template_with_random_filling_long_1_full
    # baseline_template_with_random_filling_short_3_missing
    # ======================
    # gptj_analysis_100test_newdata_newprompt_Gptneo125M_numgene10
    # gptj_analysis_100test_newdata_newprompt_Gptneo1.3B_numgene10_R3
    # gptj_analysis_100test_newdata_newprompt_Gptneo2.7B_numgene10_R3
    # gptj_analysis_100test_newdata_newprompt_10
    # gptj_analysis_100test_newdata_newprompt_Gptneox20B_numgene10_R3_capitalyesno1
    # gptj_analysis_100test_newdata_newprompt_Gptneo2.7B_numgene10_R3_capitalyesno1
    # ======================
    # gptj_analysis_100test_newdata_newprompt_shortfact_2fact_fullfact_10
    # gptj_analysis_100test_newdata_newprompt_shortfact_1fact_fullfact_10
    # gptj_analysis_100test_newdata_newprompt_longfact_1fact_fullfact_10
    # gptj_analysis_100test_newdata_newprompt_shortfact_3fact_missingfact_10
    # ======================
    # ./Checkpoints_try/gptj_analysis_100test_newdata_newprompt_10/
    # /export/home2/zonglin001/Checkpoints/CoLM/try12_llama/
    # /export/home2/zonglin001/Checkpoints/CoLM/try12_gpt2/
    # /export/home2/zonglin001/Checkpoints/CoLM/try12_vicunallama/
    # ======================
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene_2fact
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene_1fact
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene_missingfacts
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene_1fact_long
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptneox20B_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptneo2.7B_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptneo1.3B_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptneo125M_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_gptj_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_vicunallama_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_llama_12_5gene
    # /export/home2/zonglin001/Checkpoints/CoLM/new_data_llama_12_5gene_capitalYesNo
    parser.add_argument("--output_dir", default="/export/home2/zonglin001/Checkpoints/CoLM/new_data_llama_12_5gene_capitalYesNo/", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--generator_model_type", type=str, default="llama",
                        help="model type: gptneo125M/gptneo1.3B/gptneo2.7B/gptj/gptneox20B/llama/vicunallama")
    parser.add_argument("--if_long_or_short_facts", type=int, default=1, help="when 0, use long facts to induce rules; when 1, use short facts to induce rules")
    parser.add_argument("--cnt_facts_as_input", type=int, default=3, help="can be 1/2/3, indicates how many facts to use to induce rules")
    parser.add_argument("--if_full_or_missing_facts", type=int, default=0, help="when 0, use full facts; when 1, only use part of the fact to induce rules")
    parser.add_argument("--setting_selection_M1_forM2M3", type=int, default=1, help="used to identify which generated rules set to filter, useful when dataset_selection==13/14/15/16/17/18; current choices are 0/1")
    parser.add_argument("--setting_selection", type=int, default=2, help="0: zero-shot setting; 1: few-shot setting; 2: few-shot + chain of thought setting; 3: finetuning setting")
    parser.add_argument("--if_already_fintuned_for_test", type=int, default=0, help="always 0, unless when using finetuned checkpoint to only test")
    # ======================
    parser.add_argument("--if_consider_M234", type=int, default=1, help="For our framework, if_consider_M234 should be 1, while for baseline methods if_consider_M234 should be 0 to avoid read M2 and M3 and M4 and M5 result files")
    # precision metric
    parser.add_argument("--which_metric", type=int, default=1, help="0: use bleu; 1: use meteor; 2: use delta-bleu (not supported now)")
    parser.add_argument("--if_remove_stop_words", type=int, default=0, help="whether remove stop words before calculating BLEU -- 0: not remove; 1: remove")
    parser.add_argument("--min_length_rule_to_be_considered", type=int, default=45, help="the min length of generated rule to be collected for human annotation; in the first 5 train files (train_human_eval_rlt_M1setting_1_0/1/2/3/4.pt) and first 2 test files (human_eval_rlt_M1setting_1_0/1.pt), the value of this hyperparameter is 0, while for others should be 45; this should be 0 for checkpoint gptj_analysis_100test_newdata_newprompt but 45 for gptj_analysis_100test_newdata_newprompt_10")
    # recall metric
    parser.add_argument("--recall_method", type=int, default=0, help="0: the weights for recall are from [45, 35, ..., -45]; 1: the weights for recall are from [1.0, 0.9, ..., 0.1]")
    args = parser.parse_args()

    assert args.if_consider_M234 == 0 or args.if_consider_M234 == 1
    assert args.setting_selection == 0 or args.setting_selection == 1 or args.setting_selection == 2 or args.setting_selection == 3
    assert args.setting_selection_M1_forM2M3 == 0 or args.setting_selection_M1_forM2M3 == 1
    assert args.if_already_fintuned_for_test == 0 or args.if_already_fintuned_for_test == 1
    assert args.which_metric == 0 or args.which_metric == 1
    assert args.if_remove_stop_words == 0 or args.if_remove_stop_words == 1

    if args.if_consider_M234:
        thres13, thres14, thres19, thres20 = threshold_storer(args)
    else:
        thres13, thres14, thres19, thres20 = 0.5, 0.5, 0.5, 0.5


    # dict_id2trueRule_test: {0: [golden_rules]}
    test_datasets, test_datasets_notes, dict_id2trueRule_test = load_data_Hypothetical_Induction_Module123(args, 'test', if_save_dict_files=False, banned_rule_type='none')
    # generated_rule: {0: [rule1, rule2, ...]}
    # # to fit random fill baseline
    if 'baseline_template_with_random_filling' in args.output_dir:
        f_generated_rule = os.path.join(args.output_dir, 'rule_proposer_generated_rules.pt')
        f_generated_rule_notes = os.path.join(args.output_dir, 'ttl_dict_rule2notes.pt')
    else:
        f_generated_rule = os.path.join(args.output_dir, 'rule_proposer_generated_rules_{}_{}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test))
        f_generated_rule_notes = os.path.join(args.output_dir, 'ttl_dict_rule2notes_{}_{}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test))
    assert os.path.exists(f_generated_rule) and os.path.exists(f_generated_rule_notes)
    generated_rule = torch.load(f_generated_rule)
    generated_rule_notes = torch.load(f_generated_rule_notes)
    # generated_rule = torch.load(os.path.join(args.output_dir, 'rule_proposer_generated_rules_'+str(args.setting_selection_M1_forM2M3)+'.pt'))
    # generated_rule_notes = torch.load(os.path.join(args.output_dir, 'ttl_dict_rule2notes_'+str(args.setting_selection_M1_forM2M3)+'.pt'))
    # generated_rule_notes: {'0': {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}

    dict_id2ruleTemplate = torch.load(os.path.join(args.output_dir, 'dict_id2ruleTemplate_' + 'test' + '.pt'))
    dict_id2topic = torch.load(os.path.join(args.output_dir, 'dict_id2topic_test.pt'))

    # 0~5: First order logic rules; 6~7: mathematical rules
    dict_topicid2ruleFormid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1}
    if args.if_consider_M234:
        if args.if_already_fintuned_for_test == 1:
            module2_rlt = torch.load(os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            module3_rlt = torch.load(os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            module4_rlt = torch.load(os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            module5_rlt = torch.load(os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
        else:
            module2_path = os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3))
            if os.path.exists(module2_path):
                module2_path = os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3))
                module3_path = os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3))
                module4_path = os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3))
                module5_path = os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3))
            else:
                module2_path = os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test))
                module3_path = os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test))
                module4_path = os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test))
                module5_path = os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test))
            module2_rlt = torch.load(module2_path)
            module3_rlt = torch.load(module3_path)
            module4_rlt = torch.load(module4_path)
            module5_rlt = torch.load(module5_path)
    else:
        module2_rlt, module3_rlt, module4_rlt, module5_rlt = None, None, None, None

    assert len(generated_rule) == len(dict_id2trueRule_test)
    assert len(generated_rule) == len(generated_rule_notes)
    if args.if_consider_M234:
        assert len(generated_rule) == len(module2_rlt)
        assert len(generated_rule) == len(module3_rlt)
        assert len(generated_rule) == len(module4_rlt)
        assert len(generated_rule) == len(module5_rlt)

    ## BLEU for full generated rules
    print("=======================================No filter=============================================")

    get_bleu_green_analysis(args, 12, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
    # M2, M3, M2&M3
    if args.if_consider_M234:
        print("=======================================M2=============================================")
        get_bleu_green_analysis(args, 13, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)

        print("=======================================M3=============================================")
        get_bleu_green_analysis(args, 14, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)

        print("=======================================M4=============================================")
        get_bleu_green_analysis(args, 19, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)

        print("=======================================M5=============================================")
        get_bleu_green_analysis(args, 20, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)

        # print("=======================================M23=============================================")
        # get_bleu_green_analysis(args, 1314, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
        #
        # print("=======================================M24=============================================")
        # get_bleu_green_analysis(args, 1319, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)
        #
        # print("=======================================M34=============================================")
        # get_bleu_green_analysis(args, 1419, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)

        # print("=======================================M234=============================================")
        # get_bleu_green_analysis(args, 131419, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)

        print("=======================================M2345=============================================")
        get_bleu_green_analysis(args, 13141920, args.bleu_n, generated_rule, generated_rule_notes, dict_id2trueRule_test, dict_topicid2ruleFormid, dict_id2topic, dict_id2ruleTemplate, module2_rlt, module3_rlt, module4_rlt, module5_rlt, thres13, thres14, thres19, thres20)





if __name__ == '__main__':
    main()
