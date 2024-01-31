import numpy as np
import argparse, os
import torch
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
from scipy import stats
# remove stop words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# TreebankWordDetokenizer: reverse word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# word to original form
from nltk.stem.snowball import SnowballStemmer
# meteor
from nltk.translate.meteor_score import single_meteor_score
# import sacrebleu

def remove_stop_words_nltk(e2):
    stop_words = set(stopwords.words('english'))
    e2_tokens = word_tokenize(e2)
    e2_tokens_NoStopWords = []
    for tmp_i, tmp_word in enumerate(e2_tokens):
        if tmp_word not in stop_words:
            e2_tokens_NoStopWords.append(tmp_word)

    e2_NoStopWords = TreebankWordDetokenizer().detokenize(e2_tokens_NoStopWords)
    return e2_NoStopWords

# allowed_existing_annotation_files_test/train: ["", "", ...]; a list of allowable --if_overlook_existing_annotations to be considered
def get_and_concat_selected_gene_and_human_eval_from_files(args, allowed_existing_annotation_files_test="ALL", allowed_existing_annotation_files_train="ALL"):
    assert args.if_also_consider_train_val_annotation == 0 or args.if_also_consider_train_val_annotation == 1
    assert args.setting_selection_M1_forM2M3 == 0 or args.setting_selection_M1_forM2M3 == 1 or args.setting_selection_M1_forM2M3 == 2
    # find existing selected_gene and human_eval files:
    all_files = os.listdir(args.root_data_dir)
    all_files.sort(reverse=True)
    all_f_selected_generation = []
    all_f_human_eval = []

    for file in all_files:
        if file.startswith("train_selection_generation_for_huaman_eval_M1setting_"+str(args.setting_selection_M1_forM2M3)) and args.if_also_consider_train_val_annotation == 1:
            file_existing_annotation = file.replace("train_selection_generation_for_huaman_eval_M1setting_"+str(args.setting_selection_M1_forM2M3)+"_", "")[0:2].strip("._")
            if allowed_existing_annotation_files_train == "ALL" or file_existing_annotation in allowed_existing_annotation_files_train:
                all_f_selected_generation.append(file)
        elif file.startswith("selection_generation_for_huaman_eval_M1setting_"+str(args.setting_selection_M1_forM2M3)):
            file_existing_annotation = file.replace("selection_generation_for_huaman_eval_M1setting_"+str(args.setting_selection_M1_forM2M3)+"_", "")[0:2].strip("._")
            if allowed_existing_annotation_files_test == "ALL" or file_existing_annotation in allowed_existing_annotation_files_test:
                all_f_selected_generation.append(file)
        else:
            pass

        if file.startswith("train_human_eval_rlt_M1setting_"+str(args.setting_selection_M1_forM2M3)) and args.if_also_consider_train_val_annotation == 1:
            file_existing_annotation = file.replace("train_human_eval_rlt_M1setting_"+str(args.setting_selection_M1_forM2M3)+"_", "")[0:2].strip("._")
            if allowed_existing_annotation_files_train == "ALL" or file_existing_annotation in allowed_existing_annotation_files_train:
                print("file: ", file)
                all_f_human_eval.append(file)
        elif file.startswith("human_eval_rlt_M1setting_"+str(args.setting_selection_M1_forM2M3)):
            file_existing_annotation = file.replace("human_eval_rlt_M1setting_"+str(args.setting_selection_M1_forM2M3)+"_", "")[0:2].strip("._")
            if allowed_existing_annotation_files_test == "ALL" or file_existing_annotation in allowed_existing_annotation_files_test:
                all_f_human_eval.append(file)
        else:
            pass

    print("len(all_f_selected_generation): {}; len(all_f_human_eval): {}".format(len(all_f_selected_generation), len(all_f_human_eval)))
    assert len(all_f_selected_generation) == len(all_f_human_eval)

    selected_gene, human_eval = [], []
    cnt_found_hum_ev = 0
    for f_sel_gen in all_f_selected_generation:
        sel_gen = torch.load(os.path.join(args.root_data_dir, f_sel_gen))
        # selected_gene
        for tmp_id_sel_gen in range(len(sel_gen)):
            selected_gene.append(sel_gen[tmp_id_sel_gen])
        # find corresponding human evaluation
        f_sel_gen_latter_part = f_sel_gen[-7:]
        f_sel_gen_former_part = f_sel_gen[:5]
        # print("f_sel_gen_former_part: ", f_sel_gen_former_part)
        for f_hum_ev in all_f_human_eval:
            f_hum_ev_latter_part = f_hum_ev[-7:]
            f_hum_ev_former_part = f_hum_ev[:5]
            if_selected = False
            if f_sel_gen_former_part == 'train':
                if f_hum_ev_former_part == 'train':
                    if f_hum_ev_latter_part == f_sel_gen_latter_part:
                        if_selected = True
            else:
                if f_hum_ev_former_part != 'train':
                    if f_hum_ev_latter_part == f_sel_gen_latter_part:
                        if_selected = True
            if if_selected:
                cnt_found_hum_ev += 1
                hum_ev = torch.load(os.path.join(args.root_data_dir, f_hum_ev))
                # print("len(hum_ev): ", len(hum_ev))
                # human_eval
                for tmp_id_hum_ev in range(len(hum_ev)):
                    human_eval.append(hum_ev[tmp_id_hum_ev])
                break

    # print("cnt_found_hum_ev: {}; len(selected_gene): {}; len(human_eval): {}".format(cnt_found_hum_ev, len(selected_gene), len(human_eval)))
    assert cnt_found_hum_ev == len(all_f_human_eval)
    assert len(selected_gene) == len(human_eval)
    print("len(selected_gene): {}; len(human_eval): {}".format(len(selected_gene), len(human_eval)))
    return selected_gene, human_eval

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", default="./Checkpoints/gptj_analysis_100test_newdata_newprompt/", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--root_data_dir", type=str, default="./Data/DEERLET/", help="data dir for current dataset")
    # =======================
    parser.add_argument("--if_also_consider_train_val_annotation", type=int, default=1, help="0: only calculate correlation based on test annotations; 1: calculate correlation based on both train and val and test annotations.")
    parser.add_argument("--if_include_all_related_files", type=int, default=1, help="if 0: automatically collect all selected pred_rule and human annotation files for calculation; if 1: use --setting_selection and --if_overlook_existing_annotations to specify single file for calculation.")
    # added on 2022/12/15
    parser.add_argument("--setting_selection_M1_forM2M3", type=int, default=1, help="only works when --if_include_all_related_files is 0; used to identify which generated rules set to filter, useful when dataset_selection==13/14/15/16/17/18; current choices are 0/1")
    parser.add_argument("--setting_selection", type=int, default=1, help="only works when --if_include_all_related_files is 0; 0: zero-shot setting; 1: few-shot setting; 2: few-shot + chain of thought setting; 3: finetuning setting")
    parser.add_argument("--if_overlook_existing_annotations", type=int, default=0, help="only works when --if_include_all_related_files is 0; the second number in the name of randomly selected generation files; for original definition please check randomly_select_generations_for_human_eval.py")
    # =======================
    parser.add_argument("--which_metric", type=int, default=0, help="0: use bleu; 1: use meteor; 2: use delta-bleu (not supported now)")
    parser.add_argument("--bleu_n", type=int, default=4)
    parser.add_argument("--if_remove_stop_words", type=int, default=0, help="whether remove stop words before calculating BLEU -- 0: not remove; 1: remove")
    parser.add_argument("--if_word_original_form", type=int, default=0, help="whether use original form of words to calcuate BLEU -- 0: not original form; 1: original form")
    parser.add_argument("--min_length_rule_to_be_considered", type=int, default=45, help="the min length of generated rule to be used for calculating validity")
    args = parser.parse_args()

    stemmer = SnowballStemmer("english")

    assert args.if_include_all_related_files == 0 or args.if_include_all_related_files == 1
    assert args.setting_selection == 0 or args.setting_selection == 1
    assert args.if_remove_stop_words == 0 or args.if_remove_stop_words == 1
    assert args.if_word_original_form == 0 or args.if_word_original_form == 1
    assert args.which_metric == 0 or args.which_metric == 1
    assert args.if_overlook_existing_annotations >= 0
    assert args.if_also_consider_train_val_annotation == 0 or args.if_also_consider_train_val_annotation == 1

    # selected_gene: [[fact, true_rule, selected_rule_in_same_template_with_true_rule, idx]]
    if args.if_include_all_related_files:
        # load selected_gene and human_eval from all existing files
        selected_gene, human_eval = get_and_concat_selected_gene_and_human_eval_from_files(args, allowed_existing_annotation_files_test=["2", "3", "4"], allowed_existing_annotation_files_train=["5", "6", "7", "8", "9", "10", "11"])
        # selected_gene, human_eval = get_and_concat_selected_gene_and_human_eval_from_files(args, allowed_existing_annotation_files_test=["2", "3"], allowed_existing_annotation_files_train=["5", "6", "7", "8", "9", "10"])
        # selected_gene, human_eval = get_and_concat_selected_gene_and_human_eval_from_files(args, allowed_existing_annotation_files_test=["0", "1"], allowed_existing_annotation_files_train=["0", "1", "2", "3", "4"])
        # selected_gene, human_eval = get_and_concat_selected_gene_and_human_eval_from_files(args)
    else:
        # load selected_gene and human_eval from specific existing file
        f_selected_generation = os.path.join(args.root_data_dir, "selection_generation_for_huaman_eval_M1setting_{}_{}.pt".format(args.setting_selection, args.if_overlook_existing_annotations))
        selected_gene = torch.load(f_selected_generation)
        # human_eval: [[2,2,2,1,5], ...]
        f_human_eval = os.path.join(args.root_data_dir, "human_eval_rlt_M1setting_{}_{}.pt".format(args.setting_selection, args.if_overlook_existing_annotations))
        human_eval = torch.load(f_human_eval)

    # len should be the same
    assert len(selected_gene) == len(human_eval)

    n = args.bleu_n
    weights = [1/n] * n
    def score(hyp, refs):
        # print(hyp)
        # print(refs)
        return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

    # get bleu_collection and ave_human_eval_collection
    bleu_collection = []
    ave_human_eval_collection = []
    for id_data in range(len(human_eval)):
        # selected_gene and bleu
        fact, true_rule, pred_rule, idx = selected_gene[id_data]
        if len(pred_rule.strip()) <= args.min_length_rule_to_be_considered:
            continue
        # assert id_data == idx
        true_rule = true_rule.lower().strip('.').strip()
        pred_rule = pred_rule.lower().strip('.').strip()
        if args.if_remove_stop_words:
            true_rule = remove_stop_words_nltk(true_rule)
            pred_rule = remove_stop_words_nltk(pred_rule)
        if args.if_word_original_form:
            # true_rule
            true_rule = true_rule.split()
            for id_word, word in enumerate(true_rule):
                ori_word = stemmer.stem(word)
                true_rule[id_word] = ori_word
            true_rule = ' '.join(true_rule)
            # pred_rule
            pred_rule = pred_rule.split()
            for id_word, word in enumerate(pred_rule):
                ori_word = stemmer.stem(word)
                pred_rule[id_word] = ori_word
            pred_rule = ' '.join(pred_rule)

        if args.which_metric == 1:
            tmp_bleu = single_meteor_score(pred_rule.lower().strip().strip('.').split(), true_rule.lower().strip().strip('.').split())
        elif args.which_metric == 0:
            tmp_bleu = score(pred_rule.lower().strip().strip('.').split(), [t.lower().strip().strip('.').split() for t in [true_rule]])
        else:
            raise NotImplementError
        bleu_collection.append(tmp_bleu)
        # human_eval and ave_human_eval
        if_general, if_consistent, if_commonsense, if_trivial, if_natural_language = human_eval[id_data]
        if_general = int(if_general) / 2
        if_consistent = int(if_consistent) / 2
        if_commonsense = int(if_commonsense) / 2
        # if_general = np.maximum(int(if_general) - 1, 0) / 2
        # if_consistent = np.maximum(int(if_consistent) - 1, 0) / 2
        # if_commonsense = np.maximum(int(if_commonsense) - 1, 0) / 2
        if_trivial = int(if_trivial)
        if_natural_language = int(if_natural_language) / 5
        # ave_human_eval = np.mean([if_general, if_consistent, if_commonsense, if_trivial, if_natural_language])
        # ave_human_eval = if_general * if_consistent * if_commonsense * if_trivial * if_natural_language
        ave_human_eval = if_general * if_consistent * if_commonsense * if_trivial
        # ave_human_eval = if_general * if_consistent * if_commonsense
        ave_human_eval_collection.append(ave_human_eval)
    assert len(bleu_collection) == len(ave_human_eval_collection)
    if args.min_length_rule_to_be_considered == 0:
        assert len(bleu_collection) == len(selected_gene)

    mean_bleu = np.mean(bleu_collection)
    mean_ave_human_eval = np.mean(ave_human_eval_collection)

    covariance = 0
    cnt = 0
    for id_data in range(len(bleu_collection)):
        covariance += (bleu_collection[id_data] - mean_bleu) * (ave_human_eval_collection[id_data] - mean_ave_human_eval)
        cnt += 1
    covariance = covariance / (cnt - 1)
    print("covariance: ", covariance)
    std_bleu = np.std(bleu_collection)
    std_ave_human_eval = np.std(ave_human_eval_collection)
    correlation = covariance / (std_bleu * std_ave_human_eval)
    print("correlation: ", correlation)
    ttest = stats.ttest_ind(bleu_collection, ave_human_eval_collection)
    print("ttest: ", ttest)


    # very roughly check recall rate
    print("len(bleu_collection): {}; len(ave_human_eval_collection): {}".format(len(bleu_collection), len(ave_human_eval_collection)))
    idx_biggest_bleu = np.argsort(bleu_collection)[::-1]
    ave_human_eval_collection_biggest_bleu_order = [ave_human_eval_collection[i] for i in idx_biggest_bleu]
    ave_human_eval_collection_biggest_bleu_order_int = []
    for i in ave_human_eval_collection_biggest_bleu_order:
        if i > 0:
            ave_human_eval_collection_biggest_bleu_order_int.append(1)
        else:
            ave_human_eval_collection_biggest_bleu_order_int.append(0)
    ave_human_eval_collection_biggest_bleu_order = ave_human_eval_collection_biggest_bleu_order_int
    # assert len(ave_human_eval_collection) % 10 == 0
    segment_len = int(len(ave_human_eval_collection) / 10)
    ave_human_eval_segment_bleu = []
    for i in range(10):
        segment_bleu = np.mean(ave_human_eval_collection_biggest_bleu_order[i*segment_len:(i+1)*segment_len])
        ave_human_eval_segment_bleu.append(segment_bleu)
    recall_rate_weight = np.array([45, 35, 25, 15, 5, -5, -15, -25, -35, -45])
    ave_human_eval_block_bleu = np.array(ave_human_eval_segment_bleu)
    ground_truth_recall_rate = (np.dot(ave_human_eval_block_bleu, recall_rate_weight) + 125) / 250
    print("ave_human_eval_block_bleu: ", ave_human_eval_block_bleu)
    print("ground_truth_recall_rate: ", ground_truth_recall_rate)













if __name__ == '__main__':
    main()
