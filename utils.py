import random
import numpy as np
import torch
import json
import os
import re
import copy
import csv
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import average_precision_score

# Q:
# device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For generating decoder_input_ids for bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# FUNCTION
#   Get rules (sentences that with rule patterns) from the output of batch_decode;
#   Currently only fit for 1. if-then rules and there exist rules
# INPUT
#   decode_results: {'0': #num_return_sequences[ans1 for query1, ans2 for query1, ...], ...}
#   dict_generation2notes (when dataset_selection == 12 and if_query == False): {generation0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}; otherwise dict_generation2notes is None
# OUTPUT
#   generated_sents_with_rule_pattern: {'0': [rule1, ...]}
#   dict_rule2notes (when dataset_selection == 12): {'0': {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}; otherwise dict_rule2notes is None
def get_rules_from_decoding(args, decode_results, dict_generation2notes=None):
    if args.dataset_selection == 12:
        assert dict_generation2notes != None
        # dict_rule2notes: {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], [(in case two or more facts induce the same rule)]]};
        dict_rule2notes = {}
    else:
        dict_rule2notes = None
    generated_sents_with_rule_pattern = {}
    for key in decode_results:
        generated_sents_with_rule_pattern[key] = []
        for answer in decode_results[key]:
            gene_sent_split = re.split('\n|\.', answer)
            for sent in gene_sent_split:
                # use .strip(':') in case of ": if", which is caused by summary_ids[:, :args.max_e1+args.max_r-2]
                sent = sent.lower().strip(':').strip()
                if sent.startswith("rule template:"):
                    sent = sent.split("rule template:")[1].strip()
                if sent.startswith("rule:"):
                    sent = sent.split("rule:")[1].strip()
                if 'if ' in sent or 'there exist' in sent:
                    # filter too short rule, which are most likely to be incomplete sentence, and therefore reduce the cost of M2/3/4/5
                    if len(sent) >= args.min_length_rule_to_be_considered:
                        if sent not in generated_sents_with_rule_pattern[key]:
                            generated_sents_with_rule_pattern[key].append(sent)
                        if args.dataset_selection == 12:
                            if key not in dict_rule2notes:
                                dict_rule2notes[key] = {}
                            if sent not in dict_rule2notes[key]:
                                dict_rule2notes[key][sent] = [dict_generation2notes[answer][0]]
                            else:
                                dict_rule2notes[key][sent].append(dict_generation2notes[answer][0])
                    else:
                        print("filtered rule -- len(sent): {}; sent: {}".format(len(sent), sent))
    return generated_sents_with_rule_pattern, dict_rule2notes



# FUNCTION
#   adjust (concat) generation to handle num_return_sequences for generate()
#   e.g. [0, 1, 2, 3] with num_return_sequences = 2 will turn to [[0, 1], [2, 3]]
# INPUT
#   raw_batch_generation: can be query or generation: [query1, query2, ...]
#   gene_data_id: numpy array, [batch_size]
#   if_query: when true, we do not repetitively add it to batch_generation
#   notes_about_fact=[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], to relate rules with its input fact information
# OUTPUT
#   if_query==False: batch_generation: {'0': #num_return_sequences[ans1 for query1, ans2 for query1, ...], ...}
#   if_query==True: batch_generation: {'0': #only1,since ansx should be equal to ans1[ans1 for query1], ...}
#   dict_generation2notes (when dataset_selection == 12 and if_query == False): {generation0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}; otherwise dict_generation2notes is None
def concat_generation_to_handle_num_return_sequences(args, raw_batch_generation, gene_data_id, if_query=False, notes_about_fact=None):
    assert len(raw_batch_generation) == len(gene_data_id) * args.num_return_sequences
    assert len(gene_data_id.shape) == 1
    if args.dataset_selection == 12 and if_query == False:
        assert len(notes_about_fact) == 6
        for i in range(len(notes_about_fact)):
            assert len(notes_about_fact[i]) * args.num_return_sequences == len(raw_batch_generation)
        rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids = notes_about_fact
        # dict_generation2notes: {'0': {generation0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}
        dict_generation2notes = {}
    else:
        dict_generation2notes = None

    batch_generation = {}
    for id_ans, ans in enumerate(raw_batch_generation):
        cur_id_in_batch_size = int(id_ans / args.num_return_sequences)
        data_id_for_ans = gene_data_id[cur_id_in_batch_size]
        if data_id_for_ans not in batch_generation:
            batch_generation[data_id_for_ans] = [ans]
        elif ans not in batch_generation[data_id_for_ans]:
            # it's also possible to have multiple query with different rule forms for a single data_id
            batch_generation[data_id_for_ans].append(ans)
        if args.dataset_selection == 12 and if_query == False:
            if ans not in dict_generation2notes:
                dict_generation2notes[ans] = [[rule_template_ids[cur_id_in_batch_size], topic_ids[cur_id_in_batch_size], specific_general_fact_ids[cur_id_in_batch_size], long_short_facts_ids[cur_id_in_batch_size], cnt_facts_ids[cur_id_in_batch_size], full_missing_facts_ids[cur_id_in_batch_size]]]
            else:
                dict_generation2notes[ans].append([rule_template_ids[cur_id_in_batch_size], topic_ids[cur_id_in_batch_size], specific_general_fact_ids[cur_id_in_batch_size], long_short_facts_ids[cur_id_in_batch_size], cnt_facts_ids[cur_id_in_batch_size], full_missing_facts_ids[cur_id_in_batch_size]])
    return batch_generation, dict_generation2notes



# INPUT
#   batch:
#   when args.dataset_selection != 12: [#batch_size(gene_input_ids, gene_attention_mask, gene_lm_labels, data_idx_ids)]
#   when args.dataset_selection == 12: [#batch_size(gene_input_ids, gene_attention_mask, gene_lm_labels, data_idx_ids, rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids)]
#   step: global step
# OUTPUT:
#   nll_loss, nll_loss, seq_logprobs, ttl_accuracy, righ_format_accuracy, wrong_format_rewritten_accuracy, right_format_proportion, yesNoRatio, [batch_generation_query, batch_generation_answer, batch_generation_rules, dict_rule2notes], f1_counter, ttl_true_label
# Q: smooth loss is overlooked temporaly
def batch_step(args, batch, model_generator, tokenizer_generator, step=0, data_type="test"):
    assert "bart" in args.generator_model_type or 'gpt2' in args.generator_model_type or 'gptj' in args.generator_model_type or 'gptneo' in args.generator_model_type or "t5" in args.generator_model_type or "llama" in args.generator_model_type
    ## prepare input data
    batch_gene = batch
    # parallelformers will put batch to gpus itself
    # Q:
    # if not args.generator_model_type == 'gptj' or args.if_two_gpus == 0:
    if True:
        batch_gene = [t.to(device1) for t in batch_gene]
    # [batch_size, input_len_gene]; gene_data_id: the id of groups of fact, each gene_data_id could have more than one rules
    if args.dataset_selection == 12:
        gene_input_id, gene_attention_mask, gene_lm_labels, gene_data_id, rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids = batch_gene
        rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids = rule_template_ids.cpu().numpy(), topic_ids.cpu().numpy(), specific_general_fact_ids.cpu().numpy(), long_short_facts_ids.cpu().numpy(), cnt_facts_ids.cpu().numpy(), full_missing_facts_ids.cpu().numpy()
        notes_about_fact=[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids]
    else:
        gene_input_id, gene_attention_mask, gene_lm_labels, gene_data_id = batch_gene
        notes_about_fact = None
    gene_data_id = gene_data_id.cpu().numpy()
    # print("gene_data_id: ", gene_data_id)
    # print("gene_data_id.shape: ", gene_data_id.shape)
    assert len(gene_input_id.size()) == 2
    batch_size, tgt_length = gene_input_id.size()[0], gene_input_id.size()[1]
    # 9/9.5/12 --> proposer by generation; 10/11/13/14 --> classifier
    if args.dataset_selection == 9 or args.dataset_selection == 9.5 or (args.dataset_selection == 12 and data_type == "test"):
        # only need ppl for Module 1 during training
        nll_loss, nll_loss, seq_logprobs = torch.tensor(0), torch.tensor(0), None
        ttl_accuracy, righ_format_accuracy, wrong_format_rewritten_accuracy, right_format_proportion, f1_counter = None, None, None, None, None
        yesNoRatio, ttl_true_label = [], []
        if args.generator_model_type == "gptneox20B":
            bad_words_ids = [[64], [876], [15362], [1713], [3234], [795], [4772], [21933], [37866]]
        elif "gpt" in args.generator_model_type:
            bad_words_ids = [[62], [834], [17569], [1427], [29343], [25947], [37405], [1427], [4808], [11593], [46444], [44435], [37405], [2602], [4841]]
        elif "t5" in args.generator_model_type:
            bad_words_ids = [[834]]
        elif "llama" in args.generator_model_type:
            bad_words_ids = [[903], [4770]]
        else:
            raise NotImplementError
        summary_ids = model_generator.generate(
                    gene_input_id[:, :args.max_e1+args.max_r],
                    attention_mask=gene_attention_mask[:, :args.max_e1+args.max_r], max_length=args.max_e1+args.max_r+args.max_e2,
                    do_sample=True,
                    top_p=0.90,
                    temperature=0.90,
                    bad_words_ids=bad_words_ids,
                    num_return_sequences=args.num_return_sequences
                    )
        # print("summary_ids.size(): ", summary_ids.size())
        # decode query and generation
        # Q: add '-1' to include 'if' in generation; modified: use -2 to include ': if' or 'there exists'
        raw_batch_generation_query = tokenizer_generator.batch_decode(summary_ids[:, :args.max_e1+args.max_r-2], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw_batch_generation_answer = tokenizer_generator.batch_decode(summary_ids[:, args.max_e1+args.max_r-2:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("raw_batch_generation_query: ", raw_batch_generation_query)
        batch_generation_query, _ = concat_generation_to_handle_num_return_sequences(args, raw_batch_generation_query, gene_data_id, if_query=True)
        # dict_generation2notes (when dataset_selection == 12 and if_query == False): {generation0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}; otherwise dict_generation2notes is None
        batch_generation_answer, dict_generation2notes = concat_generation_to_handle_num_return_sequences(args, raw_batch_generation_answer, gene_data_id, if_query=False, notes_about_fact=notes_about_fact)
        # batch_generation_rules: generated_sents_with_rule_pattern: {'0': [rule1, ...]}
        # dict_rule2notes (when dataset_selection == 12): {'0': {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}; otherwise dict_rule2notes is None
        batch_generation_rules, dict_rule2notes = get_rules_from_decoding(args, batch_generation_answer, dict_generation2notes)

        return nll_loss, nll_loss, seq_logprobs, ttl_accuracy, righ_format_accuracy, wrong_format_rewritten_accuracy, right_format_proportion, yesNoRatio, [batch_generation_query, batch_generation_answer, batch_generation_rules, dict_rule2notes], f1_counter, ttl_true_label
    elif (args.dataset_selection == 12 and (data_type == "train" or data_type == "val")) or args.dataset_selection == 10 or args.dataset_selection == 11 or args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
        if "bart" in args.generator_model_type:
            # decoder_input_ids = shift_tokens_right(gene_lm_labels, model_generator.config.pad_token_id, model_generator.config.decoder_start_token_id)
            # results = model_generator(gene_input_id, attention_mask=gene_attention_mask, decoder_input_ids=decoder_input_ids, labels=gene_lm_labels)
            raise NotImplementError
        elif "gpt2" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or "t5" in args.generator_model_type or "llama" in args.generator_model_type:
            results = model_generator(gene_input_id, attention_mask=gene_attention_mask, labels=gene_lm_labels)
        else:
            raise NotImplementError

        # logits/seq_logits: [batch_size, tgt_length, #vocab]
        nll_loss, logits = results[0], results[1]

        seq_logits = logits

        # smooth loss is overlooked
        # get seq_logprobs: [batch_size, tgt_length, #vocab]
        seq_logprobs = F.log_softmax(seq_logits, dim=-1)

        ## get accuracy
        correct_prediction, right_format_prediction = 0, 0
        rewritten_correct_prediction, wrong_format_cannot_rewritten_prediction = 0, 0
        accuracy = 0
        # For module 2 and module 3, we use batch_generation to note the predictions
        batch_generation = {}
        # yesNoRatio: #batch_size[]
        yesNoRatio = []
        # ttl_true_label: #batch_size[]
        ttl_true_label = []
        # f1_counter: [count of generate 'yes' and label is 'yes' #(yes, yes), #(yes, no), #(no, yes), #(no, no)]
        f1_counter = [0, 0, 0, 0]
        # generator_eos_id
        if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
            generator_eos_id = tokenizer_generator.encode(tokenizer_generator.eos_token)[0]
        elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
            generator_eos_id = tokenizer_generator.encode(tokenizer_generator.eos_token)[1]
        # get indices & decode & compare to true label
        value, indices = seq_logprobs.max(dim=-1)
        for id_example in range(batch_size):
            if "gpt2" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
                output = indices[id_example].tolist()[-(args.max_e2+1):]
            elif "bart" in args.generator_model_type or "t5" in args.generator_model_type:
                output = indices[id_example].tolist()
            else:
                raise NotImplementError
            try:
                eos_pos = output.index(generator_eos_id)
                output = tokenizer_generator.decode(output[:eos_pos])
            except:
                output = tokenizer_generator.decode(output)
            # true_label: here true_label should only be one token length
            true_label = batch[2][id_example][args.max_e1+args.max_r:args.max_e1+args.max_r+1].tolist()
            # print("true_label: ", true_label)
            true_label = tokenizer_generator.decode(true_label)
            if args.dataset_selection >= 15 and args.dataset_selection <= 18:
                assert true_label == 'yes' or true_label == 'no'
                if true_label == 'yes':
                    ttl_true_label.append(1)
                elif true_label == 'no':
                    ttl_true_label.append(0)
                else:
                    raise Exception("unexpected true_label: ", true_label)
            # rewritten_output: rewrited output to be aligned with 'yes' / 'no'
            rewritten_output = None
            if output == 'yes' or output == 'no':
                right_format_prediction += 1
            else:
                if 'gptneox' in args.generator_model_type:
                    # " yes"
                    probability_yes1 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 4754])
                    # "yes"
                    probability_yes2 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 9820])
                    # " no"
                    probability_no1 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 642])
                    # "no"
                    probability_no2 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 2369])
                    probability_yes = (probability_yes1 + probability_yes2) / 2
                    probability_no = (probability_no1 + probability_no2) / 2
                elif 'gptj' in args.generator_model_type or 'gptneo' in args.generator_model_type or 'gpt2' in args.generator_model_type:
                    # probability_true = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 2081])
                    # probability_false = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 3991])
                    # probability_yes, probability_no = probability_true, probability_false
                    # " yes"
                    probability_yes1 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 3763])
                    # "yes"
                    probability_yes2 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 8505])
                    # # " Yes"
                    # probability_yes3 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 3363])
                    # # "Yes"
                    # probability_yes4 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 5297])
                    # " no"
                    probability_no1 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 645])
                    # "no"
                    probability_no2 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 3919])
                    # # " No"
                    # probability_no3 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 1400])
                    # # "No"
                    # probability_no4 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 2949])
                    # probability_yes = (probability_yes1 + probability_yes2 + probability_yes3 + probability_yes4) / 4
                    # probability_no = (probability_no1 + probability_no2 + probability_no3 + probability_no4) / 4
                    probability_yes = (probability_yes1 + probability_yes2) / 2
                    probability_no = (probability_no1 + probability_no2) / 2
                    # capital Yes and capital No
                elif 't5' in args.generator_model_type:
                    # " yes" / "yes"
                    probability_yes1 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 4273])
                    # " no" / "no"
                    probability_no1 = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 150])
                    probability_yes = probability_yes1
                    probability_no = probability_no1
                elif "llama" in args.generator_model_type:
                    # # " yes "
                    # probability_yes = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 4874])
                    # # " no "
                    # probability_no = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 694])
                    # " Yes "
                    probability_yes = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 3869])
                    # " No "
                    probability_no = torch.exp(seq_logprobs[id_example, args.max_e1+args.max_r-1, 1939])
                else:
                    raise NotImplementError
                # print("probability_yes: {}, probability_no: {}".format(probability_yes, probability_no))
                # yesNoRatio
                cur_yesNoRatio = probability_yes / (probability_yes + probability_no)
                cur_yesNoRatio = cur_yesNoRatio.detach().cpu()
                yesNoRatio.append(cur_yesNoRatio)
                # rewritten_output
                if probability_yes > probability_no:
                    rewritten_output = 'yes'
                else:
                    rewritten_output = 'no'
                if gene_data_id[id_example] not in batch_generation:
                    # batch_generation[gene_data_id[id_example]] = [rewritten_output]
                    batch_generation[gene_data_id[id_example]] = [cur_yesNoRatio.numpy()]
                else:
                    # batch_generation[gene_data_id[id_example]].append(rewritten_output)
                    batch_generation[gene_data_id[id_example]].append(cur_yesNoRatio.numpy())

            if true_label == output:
                correct_prediction += 1
            elif true_label == rewritten_output:
                rewritten_correct_prediction += 1
            # count of f1_counter; final_prediction is output if output is 'yes' or 'no'; else final_prediction is rewritten_output
            final_prediction = None
            if output == 'yes' or output == 'no':
                final_prediction = output
            else:
                assert rewritten_output == 'yes' or rewritten_output == 'no'
                final_prediction = rewritten_output
            if args.dataset_selection >= 15 and args.dataset_selection <= 18:
                assert true_label == 'yes' or true_label == 'no'
            if true_label == 'yes':
                if final_prediction == 'yes':
                    f1_counter[0] += 1
                elif final_prediction == 'no':
                    f1_counter[2] += 1
            elif true_label == 'no':
                if final_prediction == 'yes':
                    f1_counter[1] += 1
                elif final_prediction == 'no':
                    f1_counter[3] += 1

            # print("true_label: {}, rewritten_output: {}".format(true_label, rewritten_output))
            if data_type == "test":
                print("step: {}, true_label: {}, rewritten_output: {}, cur_yesNoRatio: {:.4f}".format(step, true_label, rewritten_output, cur_yesNoRatio))

        ttl_accuracy = (correct_prediction + rewritten_correct_prediction) / batch_size
        righ_format_accuracy = correct_prediction / batch_size
        wrong_format_rewritten_accuracy = rewritten_correct_prediction / batch_size
        right_format_proportion = right_format_prediction / batch_size
        return nll_loss, nll_loss, seq_logprobs, ttl_accuracy, righ_format_accuracy, wrong_format_rewritten_accuracy, right_format_proportion, yesNoRatio, batch_generation, f1_counter, ttl_true_label
    else:
        raise NotImplementError


# FUNCTION
#   concat dict_single to dict_sum;
# INPUT
#   dict_sum/dict_single: {'0': [sent1, sent2, ...], ...}
def concat_dicts(dict_single, dict_sum):
    for data_id, filtered_rules in dict_single.items():
        if data_id not in dict_sum:
            dict_sum[data_id] = filtered_rules
        else:
            for rule in filtered_rules:
                if rule not in dict_sum[data_id]:
                    dict_sum[data_id].append(rule)
    return dict_sum


# INPUT:
#   dict_single/dict_sum: {'0': {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}
# OUTPUT:
#   dict_sum: {'0': {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], ...], ...}, ...}
def concat_dicts_factNoter(dict_single, dict_sum):
    for data_id in dict_single:
        if data_id not in dict_sum:
            dict_sum[data_id] = {}
        for rule in dict_single[data_id]:
            if rule not in dict_sum[data_id]:
                dict_sum[data_id][rule] = dict_single[data_id][rule]
            else:
                # append no matter there already exists the same note
                for note in dict_single[data_id][rule]:
                    dict_sum[data_id][rule].append(note)
    return dict_sum



    return dict_sum



# data_type: "val" or "test"
# INPUT
#   dict_id2trueRule: {"0": [rule1, rule2, ...], ...}; used for save Module1_results.csv
#   pre_best_eval_acc: only save val results (when dataset_selection == 15/16/17/18) when it achieves the best val performance
# OUTPUT:
#   eval_loss, accuracy, right_format_proportion
def evaluate(args, model_generator, tokenizer_generator, eval_or_test_dataloader, data_type, dict_id2trueRule, pre_best_eval_acc=0):
    assert data_type == "val" or data_type == "test"
    if data_type == "val":
        print('INFO: begin evaluating...')
    elif data_type == "test":
        print('INFO: begin testing...')
    num_batches = len(eval_or_test_dataloader)
    print("num_batches: ", num_batches)

    eval_loss = 0
    nb_eval_steps = 0
    num_displays = 1
    if args.dataset_selection == 9 or args.dataset_selection == 9.5 or (args.dataset_selection == 12 and data_type == 'test'):
        # We need to collect generation for Module 1 during eval/test time
        ttl_batch_generation_query, ttl_batch_generation_generation, ttl_batch_generation_rule, ttl_dict_rule2notes = {}, {}, {}, {}
    elif args.dataset_selection == 10 or args.dataset_selection == 11 or (args.dataset_selection == 12 and data_type == 'val') or args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
        ttl_classification_result = {}
        ttl_correct_prediction, righ_format_correct_prediction = 0, 0
        rewritten_correct_prediction, right_format_prediction = 0, 0
        ttl_yesNoRatio = []
        ttl_true_labels = []
        # [#(generated yes, label yes), #(generated yes, label no), #(generated no, label yes), #(generated no, label no)]
        ttl_f1_counter = [0, 0, 0, 0]
    # display_batch_indices
    display_batch_indices = list(range(num_batches))
    random.shuffle(display_batch_indices)
    display_batch_indices = display_batch_indices[:num_displays]

    # eos_token
    if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
        eos_token = tokenizer_generator.encode(tokenizer_generator.eos_token)[0]
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        eos_token = tokenizer_generator.encode(tokenizer_generator.eos_token)[1]

    for step, batch in enumerate(eval_or_test_dataloader):
        batch_size = batch[0].size()[0]
        input_ids = batch[0]
        with torch.no_grad():
            loss, nll_loss, seq_logprobs, batch_accuracy, batch_righ_format_accuracy, batch_rewritten_accuracy, batch_right_format_proportion, yesNoRatio, batch_generation, f1_counter, true_labels = batch_step(args, batch, model_generator, tokenizer_generator, step, data_type)

            eval_loss += loss * batch_size
            nb_eval_steps += batch_size
            if args.dataset_selection == 9 or args.dataset_selection == 9.5 or (args.dataset_selection == 12 and data_type == "test"):
                # batch_generation[2]: {'0': [rule1, ...]}; key represents data_idx_ids
                # ttl_batch_generation_query += batch_generation[0]
                # ttl_batch_generation_generation += batch_generation[1]
                # ttl_batch_generation_rule += batch_generation[2]
                ttl_batch_generation_query = concat_dicts(batch_generation[0], ttl_batch_generation_query)
                ttl_batch_generation_generation = concat_dicts(batch_generation[1], ttl_batch_generation_generation)
                ttl_batch_generation_rule = concat_dicts(batch_generation[2], ttl_batch_generation_rule)
                ttl_dict_rule2notes = concat_dicts_factNoter(batch_generation[3], ttl_dict_rule2notes)
                # batch_generation[3]: dict_rule2notes (when dataset_selection == 12): {rule0: [[rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids], [(in case two or more facts induce the same rule)]], ...}
                # print("step: {}, generated_rules: {}".format(step, batch_generation[2]))
            if args.dataset_selection == 10 or args.dataset_selection == 11 or (args.dataset_selection == 12 and data_type == "val") or args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
                for key in batch_generation:
                    if key not in ttl_classification_result:
                        ttl_classification_result[key] = batch_generation[key]
                    else:
                        ttl_classification_result[key] += batch_generation[key]
                ttl_correct_prediction += batch_accuracy * batch_size
                righ_format_correct_prediction += batch_righ_format_accuracy * batch_size
                rewritten_correct_prediction += batch_rewritten_accuracy * batch_size
                right_format_prediction += batch_right_format_proportion * batch_size
                ttl_yesNoRatio += yesNoRatio
                ttl_true_labels += true_labels
                if step in display_batch_indices:
                    value, indices = seq_logprobs.max(dim=-1)
                    sample_index = random.randint(0, batch_size - 1)
                # ttl_f1_counter
                for tmp_i in range(len(ttl_f1_counter)):
                    ttl_f1_counter[tmp_i] += f1_counter[tmp_i]
    eval_loss = eval_loss / nb_eval_steps
    print('eval_loss:{}'.format(eval_loss))
    # print and save results
    if args.dataset_selection == 9 or args.dataset_selection == 9.5 or (args.dataset_selection == 12 and data_type == 'test'):
        assert args.if_already_fintuned_for_test == 0
        accuracy, f1, righ_format_correct_proportion, rewritten_correct_proportion, right_format_proportion, ttl_yesNoRatio, averaged_precision = None, None, None, None, None, None, None
        assert len(ttl_batch_generation_query) == len(ttl_batch_generation_generation)
        assert len(ttl_batch_generation_query) == len(ttl_batch_generation_rule)
        assert len(ttl_dict_rule2notes) == len(ttl_batch_generation_rule)
        if data_type == 'test':
            print("Here comes the generation:")
            for key in ttl_batch_generation_generation:
                if key == 0:
                    print("====================================================================================")
                    print("key: {}, Query:\n {}".format(key, ttl_batch_generation_query[key]))
                    print("-----------------------")
                    print("Generation:\n {}".format(ttl_batch_generation_generation[key]))
                    print("Rules:\n", ttl_batch_generation_rule[key])
                    print("Notes:\n", ttl_dict_rule2notes[key])
            # during test time and when args.dataset_selection == 12, save the generated rules for further processing
            # torch.save(ttl_batch_generation_rule, os.path.join(args.output_dir, 'rule_proposer_generated_rules_'+str(args.setting_selection)+'.pt'))
            # torch.save(ttl_dict_rule2notes, os.path.join(args.output_dir, 'ttl_dict_rule2notes_'+str(args.setting_selection)+'.pt'))
            torch.save(ttl_batch_generation_rule, os.path.join(args.output_dir, 'rule_proposer_generated_rules_{:.0f}_{:.0f}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test)))
            torch.save(ttl_dict_rule2notes, os.path.join(args.output_dir, 'ttl_dict_rule2notes_{:.0f}_{:.0f}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test)))
            with open(os.path.join(args.output_dir, 'Module1_results_{:.0f}_{:.0f}.csv'.format(args.setting_selection, args.if_use_deer_train_data_for_test)), 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["data_id", "query", "generated_rule", "golden_rule", "full_generation", "fact notes"])
                for key in ttl_batch_generation_query:
                    writer.writerow([key, ttl_batch_generation_query[key], ttl_batch_generation_rule[key], dict_id2trueRule[key], ttl_batch_generation_generation[key], ttl_dict_rule2notes[key]])
        return eval_loss.item(), accuracy, f1, righ_format_correct_proportion, rewritten_correct_proportion, right_format_proportion, ttl_yesNoRatio, averaged_precision
    elif args.dataset_selection == 10 or args.dataset_selection == 11 or (args.dataset_selection == 12 and data_type == 'val') or args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
        accuracy = ttl_correct_prediction / nb_eval_steps
        righ_format_correct_proportion = righ_format_correct_prediction / nb_eval_steps
        rewritten_correct_proportion = rewritten_correct_prediction / nb_eval_steps
        right_format_proportion = right_format_prediction / nb_eval_steps
        # get f1 and averaged_precision
        if args.dataset_selection >= 15 and args.dataset_selection <= 18:
            # F1 score
            print("ttl_f1_counter: ", ttl_f1_counter)
            if (ttl_f1_counter[0] + ttl_f1_counter[1]) > 0:
                precision = ttl_f1_counter[0] / (ttl_f1_counter[0] + ttl_f1_counter[1])
            else:
                print("ttl_f1_counter[0] + ttl_f1_counter[1] == 0: ", ttl_f1_counter[0], ttl_f1_counter[1])
                precision = 0
            if (ttl_f1_counter[0] + ttl_f1_counter[2]) > 0:
                recall = ttl_f1_counter[0] / (ttl_f1_counter[0] + ttl_f1_counter[2])
            else:
                print("ttl_f1_counter[0] + ttl_f1_counter[2] == 0: ", ttl_f1_counter[0], ttl_f1_counter[2])
                recall = 0
            if precision + recall > 0:
                f1 = 2*precision*recall / (precision + recall)
            else:
                print("precision: {}; recall: {}".format(precision, recall))
                f1 = 0
            # averaged_precision
            assert len(ttl_true_labels) == len(ttl_yesNoRatio)
            averaged_precision = average_precision_score(ttl_true_labels, ttl_yesNoRatio)
        else:
            f1, averaged_precision = None, None
        print('accuracy: {}, f1: {}, righ_format_correct_proportion: {}, rewritten_correct_proportion:{}, right_format_proportion: {}, nb_eval_steps: {}'.format(accuracy, f1, righ_format_correct_proportion, rewritten_correct_proportion, right_format_proportion, nb_eval_steps))
        # display ttl_yesNoRatio (only in test time do we need to display and save the results)
        if data_type == 'test':
            if args.dataset_selection == 10:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_casual_relation : {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 11:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted facuality: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 13:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_deductive_consistent: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 14:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_satisfy_commonsense: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 15:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_more_general: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 16:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_consistent: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 17:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_satisfy_commonsense: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 18:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_not_trivial: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 19:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_more_general: {:.2f}%".format(id_event+1, rato*100))
            elif args.dataset_selection == 20:
                for id_event, rato in enumerate(ttl_yesNoRatio):
                    print("id_event: {}, predicted if_not_trivial: {:.2f}%".format(id_event+1, rato*100))
            # save ttl_classification_result
            if args.dataset_selection == 13:
                # torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module2_classification_results_'+str(args.setting_selection)+'_'+str(args.setting_selection_M1_forM2M3)+'.pt'))
                torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module2_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            elif args.dataset_selection == 14:
                # torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module3_classification_results_'+str(args.setting_selection)+'_'+str(args.setting_selection_M1_forM2M3)+'.pt'))
                torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module3_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            elif args.dataset_selection == 19:
                # torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module4_classification_results_'+str(args.setting_selection)+'_'+str(args.setting_selection_M1_forM2M3)+'.pt'))
                torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module4_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            elif args.dataset_selection == 20:
                torch.save(ttl_classification_result, os.path.join(args.output_dir, 'module5_classification_results_{:.0f}_{:.0f}_{:.0f}.pt'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)))
            elif args.dataset_selection >= 15 and args.dataset_selection <= 18:
                # assert args.if_already_fintuned_for_test == 0
                assert args.if_already_fintuned_for_test == 0 or (args.setting_selection == 3 and args.if_use_deerlet_val_train_for_test == 1)
                # torch.save(ttl_classification_result, os.path.join(args.output_dir, 'M2M3_{:.0f}_{:.0f}_{:.0f}_results.pt'.format(args.dataset_selection, args.setting_selection, args.setting_selection_M1_forM2M3)))
                torch.save(ttl_classification_result, os.path.join(args.output_dir, 'M2M3_{:.0f}_{:.0f}_{:.0f}_results.pt'.format(args.dataset_selection, args.setting_selection, args.if_use_deerlet_val_train_for_test)))
        elif data_type == 'val':
            if args.dataset_selection >= 15 and args.dataset_selection <= 18:
                # assert args.if_already_fintuned_for_test == 0
                assert args.if_already_fintuned_for_test == 0 or (args.setting_selection == 3 and args.if_use_deerlet_val_train_for_test == 1)
                cur_eval_ppl = np.exp(eval_loss.item()) if eval_loss < 300 else np.inf
                # only save val results when reaches the best val performance (using acc as early stop metric)
                if accuracy > pre_best_eval_acc:
                # assert f1 != None
                # if f1 < pre_best_f1:
                    # if_use_deerlet_val_train_for_test = 1 since its val
                    if_use_deerlet_val_train_for_test = 1
                    torch.save(ttl_classification_result, os.path.join(args.output_dir, 'M2M3_{:.0f}_{:.0f}_{:.0f}_results.pt'.format(args.dataset_selection, args.setting_selection, if_use_deerlet_val_train_for_test)))
            else:
                raise NotImplementError
        else:
            raise Exception
        return eval_loss.item(), accuracy, f1, righ_format_correct_proportion, rewritten_correct_proportion, right_format_proportion, ttl_yesNoRatio, averaged_precision
    else:
        raise NotImplementError


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_special_tokens(tokenizer):
    # print("vocab size:", len(tokenizer))
    # print("\nspecial tokens:", tokenizer.special_tokens_map)
    # add special tokens
    # if not tokenizer.cls_token:
    #     tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    # if not tokenizer.sep_token:
    #     tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    # if not tokenizer.pad_token:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# FUNCTION
#   1. change nan to ""
#   2. replace "[x]" with ""
def clean_xlsx_files(df):
    ndf = copy.deepcopy(df)
    for id_row, row in ndf.iterrows():
        # print("row: ", row)
        # print("row.keys(): ", row.keys())
        for row_key in row.keys():
            value = row[row_key]
            if str(value) != "0.0" and str(value) != "1.0":
                if not str(value).isnumeric():
                    if pd.isnull(value):
                        ndf.loc[id_row, row_key] = ""
                        value = ""
                    # print("value: ", value)
                    value = re.sub('\[\d*\]', '', value)
                    ndf.loc[id_row, row_key] = re.sub(u'\\xa0', u' ', value)
    return ndf


# INPUT:
#   data_type: 'train' / 'val' / 'test'
# OUTPUT:
#   full_datasets: [#size(event, reason)]
def load_data_Hypothetical_Induction_only_Module23(args, data_type, allowed_existing_annotation_files_val="ALL", allowed_existing_annotation_files_test="ALL", allowed_existing_annotation_files_train="ALL"):
    assert data_type == 'train' or data_type == 'val' or data_type == 'test'
    def load_from_raw_deerlet_data(args, data_type):
        if 'gptneo2.7B' in args.output_dir:
            deerlet_dir = './Data/DEERLET/gptneo2.7B/'
        else:
            deerlet_dir = './Data/DEERLET/'
        assert data_type == 'train' or data_type == 'val' or data_type == 'test'
        if data_type == 'train':
            # Q: specify M1_setting as 1; since the DEERLET data are generated under M1setting 1
            # selected_generation_file_beginning = "train_selection_generation_for_huaman_eval_M1setting_"
            # human_eval_file_beginning = "train_human_eval_rlt_M1setting_"
            selected_generation_file_beginning = "train_selection_generation_for_huaman_eval_M1setting_1_"
            human_eval_file_beginning = "train_human_eval_rlt_M1setting_1_"
            allowed_existing_annotation_files = allowed_existing_annotation_files_train
        elif data_type == 'test':
            # Q: specify M1_setting as 1; since the DEERLET data are generated under M1setting 1
            # selected_generation_file_beginning = "selection_generation_for_huaman_eval_M1setting_"
            # human_eval_file_beginning = "human_eval_rlt_M1setting_"
            selected_generation_file_beginning = "selection_generation_for_huaman_eval_M1setting_1_"
            human_eval_file_beginning = "human_eval_rlt_M1setting_1_"
            allowed_existing_annotation_files = allowed_existing_annotation_files_test
        elif data_type == 'val':
            selected_generation_file_beginning = "selection_generation_for_huaman_eval_M1setting_1_"
            human_eval_file_beginning = "human_eval_rlt_M1setting_1_"
            allowed_existing_annotation_files = allowed_existing_annotation_files_val
        else:
            raise Exception("Not supported data_type: ", data_type)

        data_file = os.listdir(deerlet_dir)
        data_file.sort(reverse=True)
        all_selected_generations, all_human_evals = [], []
        for file in data_file:
            if file.startswith(selected_generation_file_beginning):
                file_existing_annotation = file.replace(selected_generation_file_beginning, "")[0:2].strip("._")
                # only use allowed annotation files
                if allowed_existing_annotation_files == "ALL" or file_existing_annotation in allowed_existing_annotation_files:
                    f_cur_selected_generations = os.path.join(deerlet_dir, file)
                    cur_selected_generations = torch.load(f_cur_selected_generations)
                    all_selected_generations += cur_selected_generations
                    if_found_corresponding_human_eval_file = False
                    for tmp_file in data_file:
                        # if tmp_file.startswith(human_eval_file_beginning) and (tmp_file[-7:] == file[-7:]):
                        if tmp_file.startswith(human_eval_file_beginning):
                            human_eval_file_existing_annotation = tmp_file.replace(human_eval_file_beginning, "")[0:2].strip("._")
                            if human_eval_file_existing_annotation == file_existing_annotation:
                                f_cur_human_evals = os.path.join(deerlet_dir, tmp_file)
                                cur_human_evals = torch.load(f_cur_human_evals)
                                all_human_evals += cur_human_evals
                                if_found_corresponding_human_eval_file = True
                                break
                    assert if_found_corresponding_human_eval_file == True
        assert len(all_selected_generations) == len(all_human_evals)
        if data_type == 'train':
            if 'gptneo2.7B' in args.output_dir:
                raise NotImplementError
            else:
                assert len(all_selected_generations) == 365 or len(all_selected_generations) == 546
        elif data_type == 'val':
            if 'gptneo2.7B' in args.output_dir:
                assert len(all_selected_generations) == 100
                all_selected_generations = all_selected_generations
                all_human_evals = all_human_evals
            else:
                # assert len(all_selected_generations) == 200
                # all_selected_generations = all_selected_generations[:50]
                # all_human_evals = all_human_evals[:50]
                assert len(all_selected_generations) == 100
                all_selected_generations = all_selected_generations
                all_human_evals = all_human_evals
        elif data_type == 'test':
            if 'gptneo2.7B' in args.output_dir:
                raise NotImplementError
            else:
                assert len(all_selected_generations) == 200
                # all_selected_generations = all_selected_generations[50:]
                # all_human_evals = all_human_evals[50:]
                all_selected_generations = all_selected_generations
                all_human_evals = all_human_evals
        else:
            raise Exception
        assert len(all_selected_generations) == len(all_human_evals)
        return all_selected_generations, all_human_evals

    assert args.dataset_selection >= 15 and args.dataset_selection <= 18
    assert data_type == 'train' or data_type == 'val' or data_type == 'test'
    full_datasets = []
    dict_id2trueRule = None
    # all_selected_generations: [[fact, true_rule, selected_rule_in_same_template_with_true_rule, idx], ...]
    # all_human_evals: [[if_general, if_consistent, if_fits_commonsense, if_trivial, if_natural_language], ...]
    all_selected_generations, all_human_evals = load_from_raw_deerlet_data(args, data_type)
    assert len(all_selected_generations) == len(all_human_evals)
    print("len(all_selected_generations): ", len(all_selected_generations))
    if data_type == 'train':
        if 'gptneo2.7B' in args.output_dir:
            raise NotImplementError
        else:
            assert len(all_selected_generations) == 365 or len(all_selected_generations) == 546
    elif data_type == 'val':
        if 'gptneo2.7B' in args.output_dir:
            assert len(all_selected_generations) == 100
        else:
            # assert len(all_selected_generations) == 50
            assert len(all_selected_generations) == 100
    elif data_type == 'test':
        if 'gptneo2.7B' in args.output_dir:
            raise NotImplementError
        else:
            # assert len(all_selected_generations) == 150
            assert len(all_selected_generations) == 200
    else:
        raise Exception
    facts, golden_rule, generated_rule = [], [], []
    whether_general_enough, whether_consistent, whether_commonsense, whether_not_trivial, whether_fluent = [], [], [], [], []
    for id_gene in range(len(all_selected_generations)):
        cur_fact, cur_true_rule, selected_rule_in_same_template_with_true_rule, idx = all_selected_generations[id_gene]
        cur_if_general, cur_if_consistent, cur_if_commonsense, cur_if_trivial, cur_if_fluent = all_human_evals[id_gene]
        cur_if_general, cur_if_consistent, cur_if_commonsense, cur_if_trivial, cur_if_fluent = int(cur_if_general), int(cur_if_consistent), int(cur_if_commonsense), int(cur_if_trivial), int(cur_if_fluent)
        cur_if_general = 1 if cur_if_general > 0 else 0
        cur_if_consistent = 1 if cur_if_consistent > 0 else 0
        cur_if_commonsense = 1 if cur_if_commonsense > 0 else 0
        assert cur_if_trivial == 1 or cur_if_trivial == 0
        cur_if_fluent = 1 if cur_if_fluent >= 3 else 0
        facts.append(cur_fact)
        golden_rule.append(cur_true_rule)
        generated_rule.append(selected_rule_in_same_template_with_true_rule)
        whether_general_enough.append(cur_if_general)
        whether_consistent.append(cur_if_consistent)
        whether_commonsense.append(cur_if_commonsense)
        whether_not_trivial.append(cur_if_trivial)
        whether_fluent.append(cur_if_fluent)

    for id_line in range(len(facts)):
        # find appropriate label according to dataset_selection
        if args.dataset_selection == 15:
            label = whether_general_enough[id_line]
        elif args.dataset_selection == 16:
            label = whether_consistent[id_line]
        elif args.dataset_selection == 17:
            label = whether_commonsense[id_line]
        elif args.dataset_selection == 18:
            label = whether_not_trivial[id_line]
        else:
            raise Exception
        # verify label and translate label to 'yes' or 'no'
        assert label == 1 or label == 0
        label = 'yes' if label == 1 else 'no'
        # label = 'no' if label == 1 else 'yes'
        # this_fact
        this_fact = facts[id_line].strip()
        if this_fact[-1] != '.':
            this_fact = this_fact + '.' + '\n'
        # this_generated_rule
        this_generated_rule = generated_rule[id_line].strip()
        if this_generated_rule[-1] != '.':
            this_generated_rule = this_generated_rule + '.'
        full_datasets.append((this_fact, this_generated_rule, label, id_line))
    print("len(full_datasets): ", len(full_datasets))
    if data_type == 'train':
        if 'gptneo2.7B' in args.output_dir:
            raise NotImplementError
        else:
            assert len(full_datasets) == 365 or len(full_datasets) == 546
    elif data_type == 'val':
        if 'gptneo2.7B' in args.output_dir:
            assert len(full_datasets) == 100
        else:
            # assert len(full_datasets) == 50
            assert len(full_datasets) == 100
    elif data_type == 'test':
        if 'gptneo2.7B' in args.output_dir:
            raise NotImplementError
        else:
            # assert len(full_datasets) == 150
            assert len(full_datasets) == 200
    else:
        raise Exception
    return full_datasets, dict_id2trueRule


# FUNCTION: get concat facts according to hyperparameters
# INPUT:
#   df: output of pd.read_excel() function
#   id_line: id_line of rule_type/topic/fact11 to get concat_fact
# OUTPUT:
#   concat_fact: concated fact using the initial setup
def get_concat_facts_according_to_initial_setup(args, df, id_line):
    rule_type = df['rule type'].tolist()
    topic = df['topic'].tolist()
    fact11 = df['fact 1.1'].tolist()
    fact12 = df['fact 1.2'].tolist()
    fact21 = df['fact 2.1'].tolist()
    fact22 = df['fact 2.2'].tolist()
    fact31 = df['fact 3.1'].tolist()
    fact32 = df['fact 3.2'].tolist()
    sfact11 = df['short fact 1.1'].tolist()
    sfact12 = df['short fact 1.2'].tolist()
    sfact21 = df['short fact 2.1'].tolist()
    sfact22 = df['short fact 2.2'].tolist()
    sfact31 = df['short fact 3.1'].tolist()
    sfact32 = df['short fact 3.2'].tolist()
    true_ruleTemplate = df['rule template'].tolist()
    true_rule = df['rule'].tolist()
    # if_long_or_short_facts
    if args.if_long_or_short_facts == 0:
        chosen_fact11, chosen_fact12, chosen_fact21, chosen_fact22, chosen_fact31, chosen_fact32 = fact11[id_line], fact12[id_line], fact21[id_line], fact22[id_line], fact31[id_line], fact32[id_line]
    elif args.if_long_or_short_facts == 1:
        chosen_fact11, chosen_fact12, chosen_fact21, chosen_fact22, chosen_fact31, chosen_fact32 = sfact11[id_line], sfact12[id_line], sfact21[id_line], sfact22[id_line], sfact31[id_line], sfact32[id_line]
    else:
        raise NotImplementError
    # if_full_or_missing_facts: if any of input to fact_to_missing_fact() is not '', the output of fact_to_missing_fact() will not be all ''
    if args.if_full_or_missing_facts == 1:
        def fact_to_missing_fact(chosen_fact1, chosen_fact2):
            assert isinstance(chosen_fact1, str) and isinstance(chosen_fact2, str)
            if len(chosen_fact1) > 0 and len(chosen_fact2) > 0:
                prob_rand = np.random.rand()
                if prob_rand < 0.5:
                    return chosen_fact1
                else:
                    return chosen_fact2
            elif len(chosen_fact1) + len(chosen_fact2) > 0:
                full_facts = '\n'.join([chosen_fact1, chosen_fact2])
                full_facts_split = re.split('\. |\n', full_facts)
                assert len(full_facts_split) >= 2
                reserved_fact_sent = full_facts_split[0]
                cnt_while_reserved_fact = 0
                while reserved_fact_sent.strip() == "":
                    cnt_while_reserved_fact += 1
                    reserved_fact_sent = full_facts_split[cnt_while_reserved_fact]
                assert reserved_fact_sent.strip() != ""
                len_full_facts_split = len(full_facts_split)
                half_len_full_facts_split = int(0.5*len_full_facts_split)
                if half_len_full_facts_split == 0:
                    missing_facts_split = reserved_fact_sent
                else:
                    prob_rand = np.random.rand()
                    if prob_rand > 0.5:
                        missing_facts_split = full_facts_split[half_len_full_facts_split:]
                    else:
                        missing_facts_split = full_facts_split[:half_len_full_facts_split]
                    missing_facts_split.append('')
                assert len(missing_facts_split) >= 2
                for id_sent, sent in enumerate(missing_facts_split):
                    missing_facts_split[id_sent] = missing_facts_split[id_sent].strip()
                missing_facts = '. '.join(missing_facts_split)
                missing_facts = missing_facts.replace('..', '.').replace('. .', '.')
                return missing_facts
            else:
                assert chosen_fact1.strip() == '' and chosen_fact2.strip() == ''
                missing_facts = ''
                return missing_facts
        chosen_fact1 = fact_to_missing_fact(chosen_fact11, chosen_fact12)
        chosen_fact2 = fact_to_missing_fact(chosen_fact21, chosen_fact22)
        chosen_fact3 = fact_to_missing_fact(chosen_fact31, chosen_fact32)
    else:
        chosen_fact1 = '\n'.join([chosen_fact11, chosen_fact12])
        chosen_fact2 = '\n'.join([chosen_fact21, chosen_fact22])
        chosen_fact3 = '\n'.join([chosen_fact31, chosen_fact32])
    # cnt_facts_as_input
    all_fact_for_this_line_id = [chosen_fact1, chosen_fact2, chosen_fact3]

    rand_idx_for_facts = []
    for id_cnt_fact in range(args.cnt_facts_as_input):
        cur_rand_idx = np.random.randint(0, 3)
        cnt_while_loop = 0
        # selected fact should not be ""; no repetition is allowed in selected fact
        while all_fact_for_this_line_id[cur_rand_idx].strip() == "" or cur_rand_idx in rand_idx_for_facts:
            cur_rand_idx = np.random.randint(0, 3)
            cnt_while_loop += 1
            if cnt_while_loop > 200:
                raise Exception("Not enough facts to select: {}".format(args.cnt_facts_as_input))
        rand_idx_for_facts.append(cur_rand_idx)
    # raw_list_to_cancat = all_fact_for_this_line_id[:args.cnt_facts_as_input]
    raw_list_to_cancat = [all_fact_for_this_line_id[rand_idx_for_facts[i]] for i in range(len(rand_idx_for_facts))]
    assert len(rand_idx_for_facts) == len(raw_list_to_cancat)
    # concat_fact
    concat_fact = '\n'.join(raw_list_to_cancat)
    return concat_fact


# INPUT:
#   data_type: 'train' / 'val' / 'test'
# OUTPUT:
#   full_datasets: [#size(e1, rel, e2, id_line)]
#   full_datasets_notes: [#size(rule template, topic, specific/general fact, long/short fact, number of fact concated, full/missing fact)]
def load_data_Hypothetical_Induction_Module123(args, data_type, if_save_dict_files=False, if_use_old_data=False, banned_rule_type="none", if_true_rule_without_prefix=False):
    assert args.dataset_selection == 12 or args.dataset_selection == 13 or args.dataset_selection == 14 or args.dataset_selection == 19 or args.dataset_selection == 20
    assert args.if_use_deer_train_data_for_test == 0 or args.if_use_deer_train_data_for_test == 1
    assert data_type == 'train' or data_type == 'val' or data_type == 'test'
    banned_rule_type = banned_rule_type.lower()
    full_datasets = []
    full_datasets_notes = []
    # factrule_property_noter: note the rule type & topic & general/specifc fact & long/short fact & number of fact concacted & full/missing fact;
    #   len(factrule_property_noter) == len(full_datasets)
    #   factrule_property_noter[i]: [rule type(rule template), topic, 1/0 for general/specific fact, 1/0 for long/short fact, 1/2/3 for number of fact concated, 1/0 for full/missing fact]
    factrule_property_noter = []
    # load facts, rule template, and ground truth rule
    if if_use_old_data:
        df = pd.read_excel(os.path.join(args.root_data_dir, 'Old_data', 'Hypothetical_Induction_'+data_type+'.xlsx'))
    else:
        # Q: use 78train instead of 73train; in the futher 100train should be used; 2022/12/8 added
        df = pd.read_excel(os.path.join(args.root_data_dir, 'DEER', 'Hypothetical_Induction_'+data_type+'.xlsx'))
        # df = pd.read_excel(os.path.join(args.root_data_dir, 'DEER', 'Hypothetical_Induction_'+data_type+'_enlarged_78_ready.xlsx'))
    if_rule_null = df['rule'].isnull()
    if_fact11_null = df['fact 1.1'].isnull()
    # print("if_rule_null: ", if_rule_null)
    df = clean_xlsx_files(df)
    rule_type = df['rule type'].tolist()
    topic = df['topic'].tolist()
    if not if_use_old_data:
        if_specific_or_general = df['specific facts (0) / general facts (1)'].tolist()
    fact11 = df['fact 1.1'].tolist()
    fact12 = df['fact 1.2'].tolist()
    fact21 = df['fact 2.1'].tolist()
    fact22 = df['fact 2.2'].tolist()
    fact31 = df['fact 3.1'].tolist()
    fact32 = df['fact 3.2'].tolist()
    sfact11 = df['short fact 1.1'].tolist()
    sfact12 = df['short fact 1.2'].tolist()
    sfact21 = df['short fact 2.1'].tolist()
    sfact22 = df['short fact 2.2'].tolist()
    sfact31 = df['short fact 3.1'].tolist()
    sfact32 = df['short fact 3.2'].tolist()
    true_ruleTemplate = df['rule template'].tolist()
    true_rule = df['rule'].tolist()

    if args.dataset_selection == 12:
        dict_id2trueRule = {}
        dict_ruleTemplate2id = {}
        dict_id2ruleTemplate = {}
        dict_topic2id = {}
        dict_id2topic = {}
        dict_key2idDataset = {}
        dict_idDataset2key = {}
        for id_line in range(len(fact11)):
            # Q: not include math data for now
            if banned_rule_type not in rule_type[id_line].lower():
                if if_rule_null[id_line] == False and if_fact11_null[id_line] == False:
                    # here only use the shorten facts
                    # if id_line == 0:
                        # print("[sfact11[id_line], sfact12[id_line], sfact21[id_line], sfact22[id_line], sfact31[id_line], sfact32[id_line]]: ", [sfact11[id_line], sfact12[id_line], sfact21[id_line], sfact22[id_line], sfact31[id_line], sfact32[id_line]])
                    concat_fact = get_concat_facts_according_to_initial_setup(args, df, id_line)
                    # rule_without_prefix: true rules that does not contain "If " or "There exist/s/ed " in the begining of the sentence
                    rule_without_prefix = true_rule[id_line].strip().strip("If ").strip("There existed ").strip("There exists ").strip("There exist ").strip()
                    dict_id2trueRule[id_line] = [true_rule[id_line]]
                    # dict_ruleTemplate2id / dict_id2ruleTemplate
                    if true_ruleTemplate[id_line] not in dict_ruleTemplate2id:
                        assert len(dict_ruleTemplate2id) == len(dict_id2ruleTemplate)
                        tmp_prev_len_dict = len(dict_ruleTemplate2id)
                        dict_ruleTemplate2id[true_ruleTemplate[id_line]] = tmp_prev_len_dict
                        dict_id2ruleTemplate[tmp_prev_len_dict] = true_ruleTemplate[id_line]
                    # dict_topic2id / dict_id2topic
                    if topic[id_line] not in dict_topic2id:
                        assert len(dict_topic2id) == len(dict_id2topic)
                        tmp_prev_len_dict = len(dict_topic2id)
                        dict_topic2id[topic[id_line]] = tmp_prev_len_dict
                        dict_id2topic[tmp_prev_len_dict] = topic[id_line]
                    if id_line not in dict_key2idDataset:
                        # "- 1" since the index starts at 0; +1 means the next data
                        cur_id_of_dataset_for_data_of_this_key = len(full_datasets) - 1 + 1
                        dict_key2idDataset[id_line] = cur_id_of_dataset_for_data_of_this_key
                        assert cur_id_of_dataset_for_data_of_this_key not in dict_idDataset2key
                        dict_idDataset2key[cur_id_of_dataset_for_data_of_this_key] = id_line
                    # repeat each input fact num_gene_times times for more generation
                    for i in range(args.num_gene_times):
                        if if_true_rule_without_prefix:
                            full_datasets.append((concat_fact, true_ruleTemplate[id_line], rule_without_prefix, id_line))
                        else:
                            full_datasets.append((concat_fact, true_ruleTemplate[id_line], true_rule[id_line], id_line))
                        if if_use_old_data:
                            full_datasets_notes.append((dict_ruleTemplate2id[true_ruleTemplate[id_line]], dict_topic2id[topic[id_line]], args.if_long_or_short_facts, args.cnt_facts_as_input, args.if_full_or_missing_facts))
                        else:
                            full_datasets_notes.append((dict_ruleTemplate2id[true_ruleTemplate[id_line]], dict_topic2id[topic[id_line]], if_specific_or_general[id_line], args.if_long_or_short_facts, args.cnt_facts_as_input, args.if_full_or_missing_facts))
        assert len(full_datasets) == len(full_datasets_notes)
        # save dicts to decode
        if if_save_dict_files:
            # in this case, load as 'train' and save as 'test'
            if args.if_use_deer_train_data_for_test and data_type == 'train' and os.path.exists(os.path.join(args.output_dir, "dict_ruleTemplate2id_" + data_type + ".pt")):
                data_type = 'test'
            torch.save(dict_ruleTemplate2id, os.path.join(args.output_dir, "dict_ruleTemplate2id_" + data_type + ".pt"))
            torch.save(dict_id2ruleTemplate, os.path.join(args.output_dir, "dict_id2ruleTemplate_" + data_type + ".pt"))
            torch.save(dict_topic2id, os.path.join(args.output_dir, "dict_topic2id_" + data_type + ".pt"))
            torch.save(dict_id2topic, os.path.join(args.output_dir, "dict_id2topic_" + data_type + ".pt"))
            torch.save(dict_key2idDataset, os.path.join(args.output_dir, "dict_key2idDataset_" + data_type + ".pt"))
            torch.save(dict_idDataset2key, os.path.join(args.output_dir, "dict_idDataset2key_" + data_type + ".pt"))
    elif args.dataset_selection == 13 or args.dataset_selection == 19:
        full_datasets_notes, dict_id2trueRule = None, None
        # # to fits for old code
        # if os.path.exists(f_generated_rule):
        #     generated_rule = torch.load(f_generated_rule)
        # else:
        f_generated_rule = os.path.join(args.output_dir, 'rule_proposer_generated_rules_{:.0f}_{:.0f}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test))
        # assert os.path.exists(f_generated_rule)
        if not os.path.exists(f_generated_rule):
            print("f_generated_rule: ", f_generated_rule)
            assert os.path.exists(f_generated_rule)
        generated_rule = torch.load(f_generated_rule)
        # len_valid_fact11 = sum([0 if fact11[i].strip() == "" else 1 for i in range(len(fact11))])
        # Q: do not count in math datas, since we haven't had enough math data yet
        len_valid_fact11 = sum([0 if fact11[i].strip() == "" or banned_rule_type in rule_type[i].lower() else 1 for i in range(len(fact11))])
        if not len(generated_rule) == len_valid_fact11:
            print("len(generated_rule): ", len(generated_rule))
            print("len(fact11): ", len(fact11))
            print("generated_rule: ", generated_rule)
            print("fact11: ", fact11)
            raise Exception
        for id_line in range(len(fact11)):
            # Q: do not count in math datas, since we haven't had enough math data yet
            if banned_rule_type not in rule_type[id_line].lower():
                if fact11[id_line].strip() != "":
                    if if_rule_null[id_line] == False and if_fact11_null[id_line] == False:
                        for id_rule in range(len(generated_rule[id_line])):
                            # here only use the shorten facts
                            concat_fact = '\n'.join([sfact11[id_line], sfact12[id_line], sfact21[id_line], sfact22[id_line], sfact31[id_line], sfact32[id_line]])
                            this_generated_rule = generated_rule[id_line][id_rule].strip()
                            if this_generated_rule[-1] != '.':
                                this_generated_rule = this_generated_rule + '.'
                            full_datasets.append((concat_fact, this_generated_rule, true_rule[id_line], id_line))
    elif args.dataset_selection == 14 or args.dataset_selection == 20:
        full_datasets_notes, dict_id2trueRule = None, None
        # # to fits for old code
        # if os.path.exists(f_generated_rule):
        #     generated_rule = torch.load(f_generated_rule)
        # else:
        f_generated_rule = os.path.join(args.output_dir, 'rule_proposer_generated_rules_{:.0f}_{:.0f}.pt'.format(args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test))
        assert os.path.exists(f_generated_rule)
        generated_rule = torch.load(f_generated_rule)
        # module2_rlt = torch.load(os.path.join(args.output_dir, 'module2_classification_results.pt'))
        # len_valid_fact11 = sum([0 if fact11[i].strip() == "" else 1 for i in range(len(fact11))])
        # Q: do not count in math datas, since we haven't had enough math data yet
        len_valid_fact11 = sum([0 if fact11[i].strip() == "" or banned_rule_type in rule_type[i].lower() else 1 for i in range(len(fact11))])
        assert len(generated_rule) == len_valid_fact11
        for id_line in range(len(fact11)):
            # Q: do not count in math datas, since we haven't had enough math data yet
            if banned_rule_type not in rule_type[id_line].lower():
                if fact11[id_line].strip() != "":
                    if if_rule_null[id_line] == False and if_fact11_null[id_line] == False:
                        for id_rule in range(len(generated_rule[id_line])):
                            # here do not use facts
                            this_generated_rule = generated_rule[id_line][id_rule].strip()
                            if this_generated_rule[-1] != '.':
                                this_generated_rule = this_generated_rule + '.'
                            full_datasets.append(("", this_generated_rule, true_rule[id_line], id_line))
    print("Len(full_datasets): ", len(full_datasets))
    return full_datasets, full_datasets_notes, dict_id2trueRule


# INPUT:
#   generated_rule: a string (both capitalized or not are ok; both with '.' in the end or not are ok)
# OUTPUT:
#   1/0, 1 means not included and can be used for evaluation, and 0 means included
def whether_not_included_in_in_context_demonstrations_in_rule_proposer(generated_rule):
    ## for Module 1
    # if rules
    rule0 = "If an animal eats meat, then it probably has large and sharp teeth.".strip().strip('.').lower()
    rule1 = "If an animal is nourished by the milks of females and is vertebrate, then it is mammal.".strip().strip('.').lower()
    rule2 = "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time.".strip().strip('.').lower()
    rule3 = "If a carnivore is frightened or hungry, it might attack.".strip().strip('.').lower()
    rule4 = "If a plant has pests, then it probably can't grow healthily.".strip().strip('.').lower()
    rule5 = "There exists a plant that is heavier than the heaviest animal in the world.".strip().strip('.').lower()
    rule6 = "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic.".strip().strip('.').lower()
    rule7 = "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant.".strip().strip('.').lower()
    rule8 = "There exists plant that can live in a desert.".strip().strip('.').lower()
    rule9 = "If two plates collide, then a plateau might form.".strip().strip('.').lower()
    rule10 = "If a place is on the land and far away from oceans, then it probably have few precipitation.".strip().strip('.').lower()
    rule11 = "There existed a continental crust of earth that covers an area of about 100,000,000 km2, about one-fifth of the Earth's surface.".strip().strip('.').lower()
    rule12 = "If a place is mountainous or has large rivers, then it probably can help troops to better defend from their enemies.".strip().strip('.').lower()
    rule13 = "If a planet orbits around a star, then the mass of the planet is less than the mass of the star.".strip().strip('.').lower()
    rule14 = "There exists a star named Sun in the Solar System, which is by far the most important source of energy for life on Earth.".strip().strip('.').lower()
    rule15 = "If a star produce more energy or has more mass, then it might has higher surface temperature.".strip().strip('.').lower()
    rule16 = "If a man is arrogant, then he probably does not have a precise understanding about the world and might encounter with failure.".strip().strip('.').lower()
    rule17 = "If a gene can't help its owner to better adapt to the world or can't deal with the new environment, then it might be eliminated by the law of nature.".strip().strip('.').lower()
    rule18 = "If a person can constantly learn new knoweldge and think deeply over the knowledge, then he probably will less likely to be confused and have proper understanding over the world.".strip().strip('.').lower()
    rule19 = "If a solid is pure and crystalline, then it has a characteristic melting point.".strip().strip('.').lower()
    # in_context_demonstrations_rule_collections
    in_context_demonstrations_rule_collections = [rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19]

    ## generated_rule
    generated_rule = generated_rule.strip().strip('.').lower()

    if generated_rule in in_context_demonstrations_rule_collections:
        return 0
    else:
        return 1




# FUNCTION
#   find encoded prompt (including in-context demonstrations) for given ruleform and dataset_selection and setting_selection
# INPUT
#   ruleform: must be in ["ifthen", "thereexists", "ifandthen", "iforthen"]
def get_encoded_input_all_prompts_besides_e1rel_for_given_ruleform(args, tokenizer_generator, ruleform, data_type):
    assert ruleform in ["ifthen", "thereexists", "ifandthen", "iforthen"]
    assert data_type in ["train", "val", "test"]
    # if ruleform in ["ifandthen", "iforthen"]:
    #     print("Warning: not support ifandthen and iforthen for now, change ruleform to ifthen")
    #     ruleform = "ifthen"
    # assert ruleform in ["ifthen", "thereexists"]
    if args.if_capital_yesno == 1:
        string_yes = "Yes"
        string_no = "No"
    elif args.if_capital_yesno == 0:
        string_yes = "yes"
        string_no = "no"
    else:
        raise Exception("undefined args.if_capital_yesno", args.if_capital_yesno)
    if args.dataset_selection == 12:
        prompt_before_e1 = "Please consider the following facts and the given rule template, try to generate a rule that satisfies the rule template and the given facts. Do not include '__' in generation. \nFact:\n "
        prompt_after_e1_before_r = " \nRule template: "
        # Q: include 'if'
        if ruleform == "ifthen" or ruleform == "ifandthen" or ruleform == "iforthen":
            # prompt_after_r = "\nRule: if"
            prompt_after_r = "\nRule: If"
        elif ruleform == "thereexists":
            # prompt_after_r = "\nRule: there exists"
            prompt_after_r = "\nRule: There exist"
        else:
            raise NotImplementError
        if args.setting_selection == 1:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. \nA baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks.\nPolar Bears’ large canine teeth were highly valued as talismans. \nMature polar bears tend to eat only the calorie-rich skin and blubber of the seal, which are highly digestible, whereas younger bears consume the protein-rich red meat.\nWolf’s teeth are heavy and large, making them better suited to crushing bone than those of other canids. \nWolves can digest their meal in a few hours and can feed several times in one day, making quick use of large quantities of meat.\n" + prompt_after_e1_before_r + "if __, then __." + "\nRule: " + "If an animal eats meat, then it probably has large and sharp teeth.\n"
                in_context_demonstration_2 = prompt_before_e1 + "Cucurbita can be susceptible to the pest Bemisia argentifolii.\nSoybean plants are vulnerable to a wide range of bacterial diseases, fungal diseases, viral diseases and parasites.\nAs a fast-growing plant, diseases are not generally a problem with radishes, but some insect pests can be a nuisance.\n" + prompt_after_e1_before_r + "if __, then __." + "\nRule: " + "If a plant has pests, then it probably can't grow healthily.\n"
                in_context_demonstration_3 = prompt_before_e1 + "The collision began in the Upper Cretaceous period about 70 million years ago, when the north-moving Indo-Australian Plate collided with the Eurasian Plate. The Indo-Australian plate continues to be driven horizontally below the Tibetan Plateau, which forces the plateau to move upwards.\nThe Chota Nagpur Plateau is a continental plateau—an extensive area of land thrust above the general land. The plateau has been formed by continental uplift from forces acting deep inside the earth, caused by the collision between the Deccan Plate and the Eurasian continent. \nDuring the Pennsylvanian period the Ozark Plateau was uplifted as a result of the Ouachita orogeny. During the late Paleozoic the deep ocean basin that existed in central and southern Arkansas was lifted when South America collided with North America, creating the folded Ouachita Mountains and uplifting the Ozark plateau to the north.\n" + prompt_after_e1_before_r + "if __, then __." + "\nRule: " + "If two plates collide, then a plateau might form.\n"
                in_context_demonstration_4 = prompt_before_e1 + "With an average orbital speed of 9.68 km/s, it takes Saturn 10,759 Earth days to finish one revolution around the Sun.\nThe Sun is 12 times larger than Saturn. 1,600 Saturn-sized planets could fit inside the Sun.\nMercury is the smallest planet in the Solar System and the closest to the Sun. Its orbit around the Sun takes 87.97 Earth days, the shortest of all the Sun's planets.\nThis is widely presumed to be what we have in the now internationally accepted figure of 1: 6023 600 for the mass of Mercury compared with that of the Sun.\nEarth orbits the Sun at an average distance of about 150 million km every 365.2564 mean solar days, or one sidereal year.\nThe mass of the sun is 1.989 x 1030 kilograms, about 333,000 times the mass of the Earth.\n" + prompt_after_e1_before_r + "if __, then __." + "\nRule: " + "If a planet orbits around a star, then the mass of the planet is less than the mass of the star.\n"
                in_context_demonstration_5 = prompt_before_e1 + "'There are two circumstances that lead to arrogance: one is when you’re wrong and you can’t face it; the other is when you’re right and nobody else can face it.' – Criss Jami \n 'The only thing more dangerous than ignorance is arrogance. ' - Albert Einstein. \n'When men are most sure and arrogant they are commonly most mistaken, given views to passion without proper deliberation which alone can secure them from the grossest absurdities.' - David Hume." + prompt_after_e1_before_r + "if __, then __." + "\nRule: " + "If a man is arrogant, then he probably does not have a precise understanding about the world and might encounter with failure.\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + "The largest known clonal flowering plant, and indeed largest plant and organism, is a grove of male Aspen in Utah, nicknamed Pando. The grove is connected by a single root system, and each stem above the ground is genetically identical. It is estimated to weigh approximately 6,000,000 kg, and covers 43.6 ha. \nThe blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons, it is the largest animal known to have ever existed. \n" + prompt_after_e1_before_r + "There exists __, which __." + "\nRule: " + "There exists a plant that is heavier than the heaviest animal in the world.\n"
                in_context_demonstration_2 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons.\n" + prompt_after_e1_before_r + "There exists __, which __." + "\nRule: " + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. \n"
                in_context_demonstration_3 = prompt_before_e1 + "Cactus are found throughout the desert regions and usually bloom in late March through May. \nSynonymous with the Mojave Desert is the Joshua tree. \nQuiver tree flourishes in desert and semi-desert areas and is found most easily in parts of South Africa and Namibia." + prompt_after_e1_before_r + "There exists __, which __." + "\nRule: " + "There exists plant that can live in a desert.\n"
                in_context_demonstration_4 = prompt_before_e1 + "Eventually, Gondwana became the largest piece of continental crust of the Paleozoic Era, covering an area of about 100,000,000 km2, about one-fifth of the Earth's surface. \n" + prompt_after_e1_before_r + "There exists __, which __." + "\nRule: " + "There existed a continental crust of earth that covers an area of about 100,000,000 km2, about one-fifth of the Earth's surface.\n"
                in_context_demonstration_5 = prompt_before_e1 + "The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy mainly as visible light, ultraviolet light, and infrared radiation. It is by far the most important source of energy for life on Earth. \n" + prompt_after_e1_before_r + "There exists __, which __." + "\nRule: " + "There exists a star named Sun in the Solar System, which is by far the most important source of energy for life on Earth. \n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + "TThe blue whale is a marine mammal and a baleen whale. \nEstimates suggest that because calves require 2–4 kilograms milk per kg of mass gain, female blue whales likely produce 220 kilograms of milk per day. The blue whale, which uses baleen to filter its prey from ocean water and can reach lengths of over 100 feet, is the largest vertebrate animal that has ever lived.\n" + prompt_after_e1_before_r + "if __ and __, then __." + "\nRule: " + "If an animal is nourished by the milks of females and is vertebrate, then it is mammal.\n"
                in_context_demonstration_2 = prompt_before_e1 + "While annuals live for only one season, they tend to have a long bloom season. They are usually bright and showy, used by gardeners to add burst of bright color to their flower beds and container gardens.\n" + prompt_after_e1_before_r + "if __ and __, then __." + "\nRule: " + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant.\n"
                in_context_demonstration_3 = prompt_before_e1 + "Why does the inland regions not get much rain? Because they are very far from the rain factories - the oceans. Now, most inland regions are very far from the oceans, with almost no proper water source capable of producing clouds. \n" + prompt_after_e1_before_r + "if __ and __, then __." + "\nRule: " + "If a place is on the land and far away from oceans, then it probably have few precipitation.\n"
                in_context_demonstration_4 = prompt_before_e1 + "To learn without thinking is blindness, to think without learning is idleness. -- <The Analects of Confucius> \n" + prompt_after_e1_before_r + "if __ and __, then __." + "\nRule: " + "If a person can constantly learn new knoweldge and think deeply over the knowledge, then he probably will less likely to be confused and have proper understanding over the world.\n"
                in_context_demonstration_5 = prompt_before_e1 + "Common ice is a crystalline material wherein the molecules are regularly arranged in a hexagonal lattice, whereas amorphous ice has a lack of long-range order in its molecular arrangement. Under a pressure of one standard atmosphere, the melting point of pure ice is the same as the ice point, that is, 0°C.\n" + prompt_after_e1_before_r + "if __ and __, then __." + "\nRule: " + "If a solid is pure and crystalline, then it has a characteristic melting point.\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + "Snakes are elongated, limbless, carnivorous reptiles of the suborder Serpentes. Snakes typically react to handling with fear or, if they are calm and relaxed, curiosity. Frightened snakes typically do not move very much, or they retreat to defensive postures. Curious snakes are not in a state of acute fear, and crawl around slowly and deliberately, flicking their forked tongues frequently.\n" + prompt_after_e1_before_r + "if __ or __, then __." + "\nRule: " + "If a carnivore is frightened or hungry, it might attack. \n"
                in_context_demonstration_2 = prompt_before_e1 + "All parts of hippomane mancinella, including the fruit, contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis. Standing beneath the tree during rain will cause blistering of the skin from even slight contact with this liquid. \n" + prompt_after_e1_before_r + "if __ or __, then __." + "\nRule: " + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic.\n"
                in_context_demonstration_3 = prompt_before_e1 + "General Mascarenhas de Moraes had established his forward headquarters in the town of Porretta Terme, which was in front of the mountains under German control. German artillery positions were considered privileged, subjecting the Allies to constant vigilance, hindering any progress towards Bologna and Po Valley.\n" + prompt_after_e1_before_r + "if __ or __, then __." + "\nRule: " + "If a place is mountainous or has large rivers, then it probably can help troops to better defend from their enemies. \n"
                in_context_demonstration_4 = prompt_before_e1 + "The surface temperature of a main sequence star is determined by the rate of energy production of its core and by its radius, and is often estimated from the star's color index.\n" + prompt_after_e1_before_r + "if __ or __, then __." + "\nRule: " + "If a star produce more energy or has more mass, then it might has higher surface temperature. \n"
                in_context_demonstration_5 = prompt_before_e1 + "Giraffes, lizards, and many other known species adapted to their environments through genetic changes to their skeletons. This form of natural selection meant that members of the population who didn't develop and present these skeletal changes died out. \n" + prompt_after_e1_before_r + "if __ or __, then __." + "\nRule: " + "If a gene can't help its owner to better adapt to the world or can't deal with the new environment, then it might be eliminated by the law of nature. \n"
            else:
                raise NotImplementError
            if args.generator_model_type == 'gptj' or 'gptneo' in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5
            # llama is using one less demonstration
            elif "llama" in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4
            elif 'gpt2' in args.generator_model_type:
                # raise Exception("input length of gpt2 model is not enough for 5 in_context_demonstrations.")
                print("Warning: input length of gpt2 model is not enough for 5 in_context_demonstrations.")
                # in_context_demonstrations = in_context_demonstration_1
                in_context_demonstrations = ""
            elif 't5' in args.generator_model_type:
                raise Exception("t5 do not have enought input length for in-context demonstrations.")
            else:
                raise NotImplementError
            # in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2
            # when do_train, we train/val/test the module without in-context demonstrations
            if args.do_train:
                prompt_before_e1 = prompt_before_e1
            else:
                prompt_before_e1 = in_context_demonstrations + prompt_before_e1
        elif args.setting_selection == 0:
            pass
        else:
            raise NotImplementError
    elif args.dataset_selection == 13 or args.dataset_selection == 16:
        # prompt_before_e1 = "Please consider the following facts and a related rule, try to classify whether the rule is deductively consistent with the facts (deductively consistent means highly relevant and not contradictory). \nFact:\n "
        prompt_before_e1 = "Please consider the following facts and a related rule, try to classify whether the rule is highly relevant and not contradictory to the facts. \nFact:\n "
        prompt_after_e1_before_r = "\nRule: "
        # prompt_after_r = "\nIs the given rule deductively consistent with the given facts? "
        prompt_after_r = "\nIs the given rule highly relevant and not contradictory to the given facts? "
        if args.setting_selection == 1:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm.\n A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks.\nPolar Bears’ large canine teeth were highly valued as talismans. \nMature polar bears tend to eat only the calorie-rich skin and blubber of the seal, which are highly digestible, whereas younger bears consume the protein-rich red meat.\nWolf’s teeth are heavy and large, making them better suited to crushing bone than those of other canids. \nWolves can digest their meal in a few hours and can feed several times in one day, making quick use of large quantities of meat." + prompt_after_e1_before_r + "If an animal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm.\n A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an mammal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably has small teeth." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably start to eat meat when it is very young." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal can fly, then it probably has a pair of wings." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably start to eat meat at the age of eight years." + prompt_after_r + string_no + ".\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists a currently living land animal, which is heavier than the heaviest marine animals of all time. " + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists blue whale, which is a marine mammal and a baleen whale. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists blue whale, which is one of the largest known land animals of all time. " + prompt_after_r + string_no + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists black hole, which is large and dense." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists , which is ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists shark, which is a kind of fish and breathe through swimming." + prompt_after_r + string_no + ".\n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Estimates suggest that because calves require 2–4 kilograms milk per kg of mass gain, female blue whales likely produce 220 kilograms of milk per day. The blue whale, which uses baleen to filter its prey from ocean water and can reach lengths of over 100 feet, is the largest vertebrate animal that has ever lived. \nZebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is a mammal. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is a mammal. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is a reptile. " + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is a mammal, then it is vertebrate. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant. " + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If  and , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal produces milk and does not have a backbone, then it is a vertebrate." + prompt_after_r + string_no + ".\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis. \nAnemone nemorosa contains chemicals that are toxic to animals including humans. All parts of the plant contain protoanemonin, which can cause severe skin and gastrointestinal irritation, bitter taste and burning in the mouth and throat. \nAll parts of rhododendron tomentosum contain poisonous terpenes that affect the central nervous system. They emit strong smell to attract bees and other pollinating insects.\n" + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is not toxic. " + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant has a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a place is mountainous or has large rivers, then it probably can help troops to better defend from their enemies. " + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant does not have a milky sap or does not produce allergic dermatitis, then it is toxic." + prompt_after_r + string_no + ".\n"
            else:
                raise NotImplementError
        elif args.setting_selection == 2:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm.\n A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks.\nPolar Bears’ large canine teeth were highly valued as talismans. \nMature polar bears tend to eat only the calorie-rich skin and blubber of the seal, which are highly digestible, whereas younger bears consume the protein-rich red meat.\nWolf’s teeth are heavy and large, making them better suited to crushing bone than those of other canids. \nWolves can digest their meal in a few hours and can feed several times in one day, making quick use of large quantities of meat." + prompt_after_e1_before_r + "If an animal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_2 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm.\n A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an mammal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_3 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably has small teeth." + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
                in_context_demonstration_4 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably start to eat meat when it is very young." + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_5 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal can fly, then it probably has a pair of wings." + prompt_after_r + string_no + ", since the rule is not highly relevant to the given facts.\n"
                # in_context_demonstration_6 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If , then ." + prompt_after_r + string_no + ", since the rule is an incomplete sentence.\n"
                in_context_demonstration_6 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably start to eat meat at the age of eight years." + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_2 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists a currently living land animal, which is heavier than the heaviest marine animals of all time. " + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
                in_context_demonstration_3 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists blue whale, which is a marine mammal and a baleen whale. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_4 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists blue whale, which is one of the largest known land animals of all time. " + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
                in_context_demonstration_5 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists black hole, which is large and dense." + prompt_after_r + string_no + ", since the rule is not highly relevant to the given facts.\n"
                # in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists , which is ." + prompt_after_r + string_no + ", since the rule is an incomplete sentence.\n"
                in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists shark, which is a kind of fish and breathe through swimming." + prompt_after_r + string_no + ", since the rule is not highly relevant to the given facts.\n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Estimates suggest that because calves require 2–4 kilograms milk per kg of mass gain, female blue whales likely produce 220 kilograms of milk per day. The blue whale, which uses baleen to filter its prey from ocean water and can reach lengths of over 100 feet, is the largest vertebrate animal that has ever lived. \nZebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is a mammal. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_2 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is a mammal. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_3 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is a reptile. " + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
                in_context_demonstration_4 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is a mammal, then it is vertebrate. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_5 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant. " + prompt_after_r + string_no + ", since the rule is not highly relevant to the given facts.\n"
                # in_context_demonstration_6 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "if  and , then ." + prompt_after_r + string_no + ", since the rule is an incomplete sentence.\n"
                in_context_demonstration_6 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate.\n" + prompt_after_e1_before_r + "If an animal produces milk and does not have a backbone, then it is a vertebrate." + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis. \nAnemone nemorosa contains chemicals that are toxic to animals including humans. All parts of the plant contain protoanemonin, which can cause severe skin and gastrointestinal irritation, bitter taste and burning in the mouth and throat. \nAll parts of rhododendron tomentosum contain poisonous terpenes that affect the central nervous system. They emit strong smell to attract bees and other pollinating insects.\n" + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_2 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_3 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is not toxic. " + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
                in_context_demonstration_4 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant has a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ", since the rule is highly relevant and not contradictory to the given facts.\n"
                in_context_demonstration_5 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a place is mountainous or has large rivers, then it probably can help troops to better defend from their enemies. " + prompt_after_r + string_no + ", since the rule is not highly relevant to the given facts.\n"
                # in_context_demonstration_6 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If or , then. " + prompt_after_r + string_no + ", since the rule is an incomplete sentence.\n"
                in_context_demonstration_6 = prompt_before_e1 + "All parts of hippomane mancinella contain toxic phorbol esters typical of the Euphorbiaceae plant family. Contact with the milky white latex produces strong allergic dermatitis.\n" + prompt_after_e1_before_r + "If a plant does not have a milky sap or does not produce allergic dermatitis, then it is toxic." + prompt_after_r + string_no + ", since the rule is contradictory to the given facts.\n"
            else:
                raise NotImplementError
        if args.setting_selection == 1 or args.setting_selection == 2:
            if args.generator_model_type == 'gptj' or 'gptneo' in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
            elif "llama" in args.generator_model_type:
                # not adding in_context_demonstration_4, since the limit of input length, and the 4th are mostly "yes" (we want to give more negative examples)
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_5 + in_context_demonstration_6
            elif 'gpt2' in args.generator_model_type:
                # raise Exception("input length of gpt2 model is not enough for 5 in_context_demonstrations.")
                print("Warning: input length of gpt2 model is not enough for 5 in_context_demonstrations.")
                in_context_demonstrations = ""
            elif 't5' in args.generator_model_type:
                raise Exception("t5 do not have enought input length for in-context demonstrations.")
            else:
                raise NotImplementError
            # in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2
            # when do_train, we train/val/test the module without in-context demonstrations
            if args.do_train:
                prompt_before_e1 = prompt_before_e1
            else:
                prompt_before_e1 = in_context_demonstrations + prompt_before_e1
    elif args.dataset_selection == 14 or args.dataset_selection == 17:
        # maybe also prompts for decide whether the rule is a copy of fact
        # prompt_before_e1 = "Please consider the following rule, try to classify whether it fits commonsense and is not trivial (not trivial means the rule should be a complete sentence and the information in former subsentence and latter subsentence should not have the same meaning)."
        prompt_before_e1 = "Please consider the following rule, try to classify whether it fits commonsense."
        prompt_after_e1_before_r = "\nRule: "
        # prompt_after_r = "\nDoes the given rule fit commonsense and not trivial? "
        prompt_after_r = "\nDoes the given rule fit commonsense? "
        if args.setting_selection == 1:
            if ruleform == "ifthen":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If , then ." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "if is the quality of the ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has gills" + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal eats meat, then it probably has hard shells." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal has lung, then it has lung." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal has lung, then it is an animal." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal has lung, then it is probably terrestrial." + prompt_after_r + string_yes + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal has lung, then it probably uses its gill to breathe." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it eats meat." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal eats meat, then it is an animal." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has no teeth." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has weak and thin body." + prompt_after_r + string_no + ".\n"
            elif ruleform == "thereexists":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "There exists , which ." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which has blue color." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a fish but not mammal." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a blue whale." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a large land animal." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is one of the largest known land animals of all time. " + prompt_after_r + string_no + ".\n"
            elif ruleform == "ifandthen":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If and , then ." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the and ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the flower is an animal." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the flower will have red fruit." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might has colorful flower." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might have no flower." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant probably is a fern." + prompt_after_r + string_no + ".\n"
            elif ruleform == "iforthen":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the or ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it is delicious." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it is a mammal." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has strange smell." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is not toxic." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is favourate food of humans." + prompt_after_r + string_no + ".\n"
            else:
                raise NotImplementError
        elif args.setting_selection == 2:
            if ruleform == "ifthen":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If , then ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has gills" + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal eats meat, then it probably has hard shells." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it eats meat." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                # in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal eats meat, then it is an animal." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ", since the rule fits commensense and is not trivial.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has no teeth." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If an animal eats meat, then it probably has weak and thin body." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
            elif ruleform == "thereexists":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "There exists , which ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which has blue color." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a fish but not mammal." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a blue whale." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a large land animal." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ", since the rule fits commensense and is not trivial.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is one of the largest known land animals of all time. " + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
            elif ruleform == "ifandthen":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If and , then ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the and ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the flower is an animal." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the flower will have red fruit." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might has colorful flower." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant." + prompt_after_r + string_yes + ", since the rule fits commensense and is not trivial.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might have no flower." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant probably is a fern." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
            elif ruleform == "iforthen":
                # in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                # in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the or ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it is delicious." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it is a mammal." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has strange smell." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic." + prompt_after_r + string_yes + ", since the rule fits commensense and is not trivial.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is not toxic." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is favourate food of humans." + prompt_after_r + string_no + ", since the rule does not fit commonsense.\n"
            else:
                raise NotImplementError
        if args.setting_selection == 1 or args.setting_selection == 2:
            if args.generator_model_type == 'gptj' or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
            elif 'gpt2' in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
            elif 't5' in args.generator_model_type:
                raise Exception("t5 do not have enought input length for in-context demonstrations.")
            else:
                raise NotImplementError
            # when do_train, we train/val/test the module without in-context demonstrations
            if args.do_train:
                prompt_before_e1 = prompt_before_e1
            else:
                prompt_before_e1 = in_context_demonstrations + prompt_before_e1
    elif args.dataset_selection == 19 or args.dataset_selection == 15:
        prompt_before_e1 = "Please consider the following facts and a related rule, try to classify whether the rule is more general than given facts or no more general rule can be induced from the given facts. \nFact:\n "
        prompt_after_e1_before_r = "\nRule: "
        prompt_after_r = "\nIs the given rule more general than the given facts or no more general rule can be induced from the given facts? "
        if args.setting_selection == 1:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm.\n A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If a mammal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If a tiger eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal is a baby tiger, then its milk teeth break through at the age of about two weeks." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_5 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If  and , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it is tiger." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "'There are two circumstances that lead to arrogance: one is when you’re wrong and you can’t face it; the other is when you’re right and nobody else can face it.' – Criss Jami \n 'The only thing more dangerous than ignorance is arrogance. ' - Albert Einstein." + prompt_after_e1_before_r + "If a man is arrogant, then he probably does not have a precise understanding about the world and might encounter with failure." + prompt_after_r + string_yes + ".\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists blue whale, which is a marine mammal and a baleen whale. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "Argentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists argentinosaurus, which is with length estimates ranging from 30 to 39.7 metres. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists a marine animal which reaches a maximum confirmed length of 29.9 meters." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists a marine mammal which weighs up to 199 metric tons." + prompt_after_r + string_yes + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists , which is ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists blue whale, which can reach 29.9 meters long and 199 metric tons." + prompt_after_r + string_no + ".\n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is mammal." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If a horse is nourished by the milks of females and is vertebrate, then it is mammal." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If a zebra is nourished by the milks of females and is vertebrate, then it is mammal." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If a animal is zebra and is vertebrate, then it is mammal." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_5 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If and , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If an animal can produce milk and is vertebrate, then it is a zebra." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "To learn without thinking is blindness, to think without learning is idleness. -- <The Analects of Confucius>" + prompt_after_e1_before_r + "If a person can constantly learn new knoweldge and think deeply over the knowledge, then he probably will less likely to be confused and have proper understanding over the world." + prompt_after_r + string_yes + ".\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a carnivore is frightened or hungry, it might attack." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a animal is frightened or hungry, it might attack." + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a cat is frightened or hungry, it might attack." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a animal is a cat or is frightened, it might attack." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_5 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If an animal is frightened or hungry, then it is a cat." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + "The surface temperature of a main sequence star is determined by the rate of energy production of its core and by its radius, and is often estimated from the star's color index." + prompt_after_e1_before_r + "If a star produce more energy or has more mass, then it might has higher surface temperature." + prompt_after_r + string_yes + ".\n"
            else:
                raise NotImplementError
        elif args.setting_selection == 2:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm.\n A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it generalizes tiger to animal.\n"
                in_context_demonstration_2 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If a mammal eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it generalizes tiger to mammal.\n"
                in_context_demonstration_3 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If a tiger eats meat, then it probably has large and sharp teeth." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a tiger'.\n"
                in_context_demonstration_4 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "If an animal is a baby tiger, then its milk teeth break through at the age of about two weeks." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a baby tiger'.\n"
                in_context_demonstration_5 = prompt_before_e1 + "'There are two circumstances that lead to arrogance: one is when you’re wrong and you can’t face it; the other is when you’re right and nobody else can face it.' – Criss Jami \n 'The only thing more dangerous than ignorance is arrogance. ' - Albert Einstein. \n" + prompt_after_e1_before_r + "If a man is arrogant, then he probably does not have a precise understanding about the world and might encounter with failure." + prompt_after_r + string_yes + ", since it's very hard to induce rules which is more general than the given rule from given facts.\n"
                in_context_demonstration_6 = ""
                # in_context_demonstration_6 = prompt_before_e1 + "The tiger has fairly stout teeth; its somewhat curved canines are the longest among living felids with a crown height of up to 90 mm. A baby tiger’s milk teeth break through at the age of about two weeks. They start to eat meat at the age of eight weeks." + prompt_after_e1_before_r + "if  and , then " + prompt_after_r + string_no + ", since the rule is not a complete sentence.\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons. \nArgentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it compares the given facts and reaches to a new general conclusion.\n"
                in_context_demonstration_2 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists blue whale, which is a marine mammal and a baleen whale. " + prompt_after_r + string_yes + ", since it's very hard to induce rules which is more general than the given rule from given facts.\n"
                in_context_demonstration_3 = prompt_before_e1 + "Argentinosaurus is one of the largest known land animals of all time, perhaps the largest, with length estimates ranging from 30 to 39.7 metres and weight estimates from 50 to 100 tons." + prompt_after_e1_before_r + "There exists argentinosaurus, which is with length estimates ranging from 30 to 39.7 metres. " + prompt_after_r + string_yes + ", since it's very hard to induce rules which is more general than the given rule from given facts.\n"
                in_context_demonstration_4 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists a marine animal which reaches a maximum confirmed length of 29.9 meters." + prompt_after_r + string_yes + ", since the rule is more general than the given facts.\n"
                in_context_demonstration_5 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists a marine mammal which weighs up to 199 metric tons." + prompt_after_r + string_yes + ", since the rule is more general than the given facts.\n"
                # in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists , which is ." + prompt_after_r + string_no + ", since the rule is not a complete sentence.\n"
                in_context_demonstration_6 = prompt_before_e1 + "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of 29.9 meters and weighing up to 199 metric tons." + prompt_after_e1_before_r + "There exists blue whale, which can reach 29.9 meters long and 199 metric tons." + prompt_after_r + string_no + ", since it's very hard to induce rules which is more general than the given rule from given facts.\n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If an animal is nourished by the milks of females and is vertebrate, then it is mammal." + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it generalizes zebra to animal.\n"
                in_context_demonstration_2 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If a horse is nourished by the milks of females and is vertebrate, then it is mammal." + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it generalizes zebra to horse.\n"
                in_context_demonstration_3 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If a zebra is nourished by the milks of females and is vertebrate, then it is mammal." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a zebra'.\n"
                in_context_demonstration_4 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If a animal is zebra and is vertebrate, then it is mammal." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'is zebra'.\n"
                # in_context_demonstration_5 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If and , then ." + prompt_after_r + string_no + ", since the rule is not a complete sentence.\n"
                in_context_demonstration_5 = prompt_before_e1 + "Zebras' dazzling stripes make them among the most recognisable mammals. Plains zebra produce milk. The zebra has a backbone and hence is classified as a vertebrate. " + prompt_after_e1_before_r + "If an animal can produce milk and is vertebrate, then it is a zebra." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a zebra'.\n"
                in_context_demonstration_6 = prompt_before_e1 + "To learn without thinking is blindness, to think without learning is idleness. -- <The Analects of Confucius>" + prompt_after_e1_before_r + "If a person can constantly learn new knoweldge and think deeply over the knowledge, then he probably will less likely to be confused and have proper understanding over the world." + prompt_after_r + string_yes + ", since it's very hard to induce rules which is more general than the given rule from given facts.\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a carnivore is frightened or hungry, it might attack." + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it generalizes cat to carnivore.\n"
                in_context_demonstration_2 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a animal is frightened or hungry, it might attack." + prompt_after_r + string_yes + ", since the rule is more general than the given facts, as it generalizes cat to animal.\n"
                in_context_demonstration_3 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a cat is frightened or hungry, it might attack." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a cat'.\n"
                in_context_demonstration_4 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If a animal is a cat or is frightened, it might attack." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a cat'.\n"
                # in_context_demonstration_5 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ", since the rule is not a complete sentence.\n"
                in_context_demonstration_5 = prompt_before_e1 + "The cat is a domestic species of small carnivorous mammal. When frightened, some cats may show signs of agitation or aggression, such as dilated pupils, arched back, pilo-erection, and hissing." + prompt_after_e1_before_r + "If an animal is frightened or hungry, then it is a cat." + prompt_after_r + string_no + ", since the rule is as general than the given facts, as it directly quote 'a cat'.\n"
                in_context_demonstration_6 = prompt_before_e1 + "The surface temperature of a main sequence star is determined by the rate of energy production of its core and by its radius, and is often estimated from the star's color index." + prompt_after_e1_before_r + "If a star produce more energy or has more mass, then it might has higher surface temperature." + prompt_after_r + string_yes + ", since it's very hard to induce rules which is more general than the given rule from given facts.\n"
            else:
                raise NotImplementError
        if args.setting_selection == 1 or args.setting_selection == 2:
            if args.generator_model_type == 'gptj' or 'gptneo' in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
            elif "llama" in args.generator_model_type:
                # not adding in_context_demonstration_2, since the limit of input length, and the 4th are mostly "yes" (we want to give more negative examples)
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
            elif 'gpt2' in args.generator_model_type:
                # in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
                in_context_demonstrations = ""
                print("Warning: input length of gpt2 model is not enough for 5 in_context_demonstrations.")
            elif 't5' in args.generator_model_type:
                raise Exception("t5 do not have enought input length for in-context demonstrations.")
            else:
                raise NotImplementError
            # when do_train, we train/val/test the module without in-context demonstrations
            if args.do_train:
                prompt_before_e1 = prompt_before_e1
            else:
                prompt_before_e1 = in_context_demonstrations + prompt_before_e1
    elif args.dataset_selection == 20 or args.dataset_selection == 18:
        prompt_before_e1 = "Please consider the following rule, try to classify whether it is not trivial (not trivial means the rule should be a complete sentence and the information in former subsentence and latter subsentence should not have the same meaning)."
        prompt_after_e1_before_r = "\nRule: "
        prompt_after_r = "\nDoes the given rule not trivial? "
        if args.setting_selection == 1:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the ." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "if and " + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If *, then *." + prompt_after_r + string_no + ".\n"
                # in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal has lung, then it is probably terrestrial." + prompt_after_r + string_yes + ".\n"
                # in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "if an animal has lung, then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has a strange smell." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a place is closer to the equator, then it might near the equator." + prompt_after_r + string_no + ".\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "There exists , which ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a blue whale." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "There exists '', which ''." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "There exists <>, which is <>. " + prompt_after_r + string_no + ".\n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If and , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the and ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If * and *, then *." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might has flowers." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might has colorful flower." + prompt_after_r + string_no + ".\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the or ." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If * or *, then *." + prompt_after_r + string_no + ".\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ".\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has strange smell. " + prompt_after_r + string_no + ".\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has bitter taste. " + prompt_after_r + string_no + ".\n"
            else:
                raise NotImplementError
        elif args.setting_selection == 2:
            if ruleform == "ifthen":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If , then ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If *, then *." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ", since the rule is not trivial.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has a strange smell." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a place is closer to the equator, then it might near the equator." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
            elif ruleform == "thereexists":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "There exists , which ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "There exists blue whale, which is a blue whale." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "There exists '', which ''." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "There exists a currently living marine animal blue whale, which is heavier than the heaviest land animals of all time. " + prompt_after_r + string_yes + ", since the rule is not trivial.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "There exists <>, which is <>. " + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
            elif ruleform == "ifandthen":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If and , then ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the and ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If * and *, then *." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant holding the flower might be annual plant. " + prompt_after_r + string_yes + ", since the rule is not trivial.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might has flowers." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a flower has long bloom season and has flamboyant color, then the plant might has colorful flower." + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
            elif ruleform == "iforthen":
                in_context_demonstration_1 = prompt_before_e1 + prompt_after_e1_before_r + "If or , then ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_2 = prompt_before_e1 + prompt_after_e1_before_r + "If is the quality of the or ." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_3 = prompt_before_e1 + prompt_after_e1_before_r + "If * or *, then *." + prompt_after_r + string_no + ", since the rule is trivial as it is not a complete sentence.\n"
                in_context_demonstration_4 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably is toxic. " + prompt_after_r + string_yes + ", since the rule is not trivial.\n"
                in_context_demonstration_5 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has strange smell. " + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
                in_context_demonstration_6 = prompt_before_e1 + prompt_after_e1_before_r + "If a plant has a bitter taste, or a funny smell, or milky sap, then it probably has bitter taste. " + prompt_after_r + string_no + ", since the rule is trivial as the latter part of the rule sentence is a repetition of its former part.\n"
            else:
                raise NotImplementError
        if args.setting_selection == 1 or args.setting_selection == 2:
            if args.generator_model_type == 'gptj' or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
                in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
            elif 'gpt2' in args.generator_model_type:
                # in_context_demonstrations = in_context_demonstration_1 + in_context_demonstration_2 + in_context_demonstration_3 + in_context_demonstration_4 + in_context_demonstration_5 + in_context_demonstration_6
                in_context_demonstrations = ""
                print("Warning: input length of gpt2 model is not enough for 5 in_context_demonstrations.")
            elif 't5' in args.generator_model_type:
                raise Exception("t5 do not have enought input length for in-context demonstrations.")
            else:
                raise NotImplementError
            # when do_train, we train/val/test the module without in-context demonstrations
            if args.do_train:
                prompt_before_e1 = prompt_before_e1
            else:
                prompt_before_e1 = in_context_demonstrations + prompt_before_e1
    else:
        raise NotImplementError

    ## mask_value_on_lm_models & encoded_pad_token
    # Both Bert and GPT2 require here as -100
    mask_value_on_lm_models = -100
    if 'gpt2' in args.generator_model_type or 'gptj' in args.generator_model_type or 'gptneo' in args.generator_model_type:
        encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)
        encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)
        encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)
    elif 'bart' in args.generator_model_type:
        # keep the <sos> token
        # encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)[1:-1]
        encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)[:-1]
        encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)[1:-1]
        encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)[1:-1]
    elif 't5' in args.generator_model_type:
        encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)[:-1]
        encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)[:-1]
        encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)[:-1]
    elif "llama" in args.generator_model_type:
        # keep the <sos> token
        encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)[:]
        encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)[1:]
        encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)[1:]
    else:
        raise NotImplementError


    return encoded_prompt_before_e1, encoded_prompt_after_e1_before_r, encoded_prompt_after_r

# FUNCTION
#   get ruleform of the crowdsourced rule of the data instance in dataset
#   here we use original excel data to read the rule format. It is possible since we have its id in original excel data
# INPUT
#   target: encoded target of (e1, rel, target, id_in_ori_dataset)
#   id_in_ori_dataset: id_in_ori_dataset in (e1, rel, target, id_in_ori_dataset)
# OUTPUT
#   golded_ruleform: must be in ["ifthen", "thereexists", "ifandthen", "iforthen"]
def get_golden_ruleform(args, target, id_in_ori_dataset, tokenizer_generator, data_type):
    if args.dataset_selection == 12:
        df = pd.read_excel(os.path.join(args.root_data_dir, 'DEER', 'Hypothetical_Induction_'+data_type+'.xlsx'))
        df = clean_xlsx_files(df)
        this_true_ruleTemplate = df['rule template'].tolist()[id_in_ori_dataset]
        this_true_rule = df['rule'].tolist()[id_in_ori_dataset].lower().strip()
        this_true_rule_split = this_true_rule.split(",")
        decoded_target_rule = tokenizer_generator.decode(target).lower().strip()
        if not decoded_target_rule in this_true_rule:
            print("this_true_rule: ", this_true_rule)
            print("decoded_target_rule: ", decoded_target_rule)
            raise Exception
        # classify ruleformat
        if this_true_rule.startswith("there"):
            golded_ruleform = "thereexists"
        elif " and " in this_true_rule_split[0]:
            golded_ruleform = "ifandthen"
        elif " or " in this_true_rule_split[0]:
            golded_ruleform = "iforthen"
        elif "if " in this_true_rule:
            golded_ruleform = "ifthen"
        else:
            raise NotImplementError
    elif args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
        decoded_target_rule = tokenizer_generator.decode(target).lower().strip()
        if decoded_target_rule.startswith("if"):
            golded_ruleform = "ifthen"
            if "," in decoded_target_rule:
                decoded_target_rule = decoded_target_rule.split(",")
                former_decoded_target_rule = decoded_target_rule[0]
                if " and " in former_decoded_target_rule:
                    golded_ruleform = "ifandthen"
                elif " or " in former_decoded_target_rule:
                    golded_ruleform = "iforthen"
        elif decoded_target_rule.startswith("there"):
            golded_ruleform = "thereexists"
        elif "if" in decoded_target_rule:
            golded_ruleform = "ifthen"
            decoded_target_rule = decoded_target_rule.split("if")[1]
            if "," in decoded_target_rule:
                decoded_target_rule = decoded_target_rule.split(",")
                former_decoded_target_rule = decoded_target_rule[0]
                if " and " in former_decoded_target_rule:
                    golded_ruleform = "ifandthen"
                elif " or " in former_decoded_target_rule:
                    golded_ruleform = "iforthen"
        elif "there" in decoded_target_rule and "exist" in decoded_target_rule:
            golded_ruleform = "thereexists"
        else:
            # raise Exception("No rule can be found in decoded_target_rule: ", decoded_target_rule)
            print("Warning: no if or thereexists rule can be found in decoded_target_rule: ", decoded_target_rule)
            golded_ruleform = "ifthen"
    else:
        raise NotImplementError
    return golded_ruleform

# FUNCTION
#   no padding in the middle and in the front, only pad in the right, no labels added; during inference, need to remove the padding on the right side, and the batch size can only be 1;
#   I guess there's bug in GPT-j model so that padding will lead to bad results
#   Encode (train, val, test) sets in single function
# INPUT:
#   encoded_datasets: [#train[#size(facts, rule, label)], #eval[], #test[]]
#   data_type: 'train' or 'val' or 'test'; when data_type=='train', tensor_dataset only includes data whose rule format matches ground truth label; when data_type=='val' or 'test', tensor_dataset includes data that for each input fact all rule formats are covered.
#   data_notes: (train_datasets_notes, eval_datasets_notes, test_datasets_notes) from load_data_Hypothetical_Induction_Module123;
#   train_datasets_notes/eval_datasets_notes/test_datasets_notes: [#size(rule template, topic, specific/general fact, long/short fact, number of fact concated, full/missing fact)]
# OUTPUT:
#   tensor_datasets: [#train(gene_input_ids, gene_attention_mask, gene_lm_labels, data_idx_ids), #eval(), #test()]
# , data_type='test'
def preprocess_datasets_Module123_hypothetical_induction_leftPadding(args, encoded_datasets, tokenizer_generator, data_notes=None):
    assert args.dataset_selection >= 12 and args.dataset_selection <= 20
    assert "gpt2" in args.generator_model_type or "bart" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or 't5' in args.generator_model_type or "llama" in args.generator_model_type
    if args.dataset_selection == 12:
        assert len(data_notes) == 3
    tensor_datasets = []
    input_len = args.max_e1 + args.max_r + args.max_e2

    ## mask_value_on_lm_models & encoded_pad_token
    # Both Bert and GPT2 require here as -100
    mask_value_on_lm_models = -100
    if 'gpt2' in args.generator_model_type or 'gptj' in args.generator_model_type or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
        encoded_pad_token = tokenizer_generator.encode(tokenizer_generator.pad_token)[0]
        # encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)
        # encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)
        # encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)
        encoded_eos_token = tokenizer_generator.encode(tokenizer_generator.eos_token)
    elif 'bart' in args.generator_model_type:
        encoded_pad_token = tokenizer_generator.encode(tokenizer_generator.pad_token)[1]
        # encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)[1:-1]
        # encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)[1:-1]
        # encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)[1:-1]
        encoded_eos_token = tokenizer_generator.encode(tokenizer_generator.eos_token)[1:-1]
    elif 't5' in args.generator_model_type:
        encoded_pad_token = [tokenizer_generator.encode(tokenizer_generator.pad_token)[0]]
        # encoded_prompt_before_e1 = tokenizer_generator.encode(prompt_before_e1)[:-1]
        # encoded_prompt_after_e1_before_r = tokenizer_generator.encode(prompt_after_e1_before_r)[:-1]
        # encoded_prompt_after_r = tokenizer_generator.encode(prompt_after_r)[:-1]
        encoded_eos_token = [tokenizer_generator.encode(tokenizer_generator.eos_token)[0]]
    else:
        raise NotImplementError

    data_type_list = ["train", "val", "test"]
    for id_dataset, dataset in enumerate(encoded_datasets):
        # data_type related issues
        assert len(data_type_list) == len(encoded_datasets)
        data_type = data_type_list[id_dataset]
        assert data_type == 'train' or data_type == 'val' or data_type == 'test'
        # begin preprocess
        # rule_format_collection: used for getting proper ICD and prompts
        if (args.dataset_selection == 12 and data_type == 'train') or args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
            rule_format_collection = [None]
        elif args.dataset_selection == 12 and (data_type == 'val' or data_type == 'test'):
            rule_format_collection = ["ifthen", "thereexists", "ifandthen", "iforthen"]
            # rule_format_collection = ["ifthen", "thereexists"]
        else:
            raise NotImplementError
        dict_key2idDataset = torch.load(os.path.join(args.output_dir, 'dict_key2idDataset_' + data_type + '.pt'))
        dict_idDataset2key = torch.load(os.path.join(args.output_dir, 'dict_idDataset2key_' + data_type + '.pt'))

        # previous beginning of the for loop
        n_data = len(dataset)
        print("n_data: ", n_data)
        ## input_ids
        # input_ids = encoded_pad_token * np.full((n_data, input_len), fill_value=1, dtype=np.int64)
        input_ids = encoded_pad_token * np.full((n_data * len(rule_format_collection), input_len), fill_value=1, dtype=np.int64)
        data_idx_ids = []
        if args.dataset_selection == 12:
            # cur_data_notes: [#size(rule template, topic, specific/general fact, long/short fact, number of fact concated, full/missing fact)]
            cur_data_notes = data_notes[id_dataset]
            assert len(cur_data_notes) == len(dataset)
            rule_template_ids, topic_ids, specific_general_fact_ids, long_short_facts_ids, cnt_facts_ids, full_missing_facts_ids = [], [], [], [], [], []
        for idx_data, (e1, rel, target, id_in_ori_dataset) in enumerate(dataset):
            for id_ruleform, tmp_ruleform in enumerate(rule_format_collection):
                idx_for_input_ids = idx_data * len(rule_format_collection) + id_ruleform
                if tmp_ruleform == None:
                    if args.dataset_selection == 12 and data_type == 'train':
                        tmp_golded_ruleform = get_golden_ruleform(args, target, id_in_ori_dataset, tokenizer_generator, data_type)
                    elif args.dataset_selection >= 13 and args.dataset_selection <= 20:
                            tmp_golded_ruleform = get_golden_ruleform(args, rel, id_in_ori_dataset, tokenizer_generator, data_type)
                    else:
                        raise Exception
                    tmp_ruleform = tmp_golded_ruleform
                encoded_prompt_before_e1, encoded_prompt_after_e1_before_r, encoded_prompt_after_r = get_encoded_input_all_prompts_besides_e1rel_for_given_ruleform(args, tokenizer_generator, tmp_ruleform, data_type)

                # when == 14/17/18, we do not want to involve fact, but just to evaluate rule itself
                if args.dataset_selection == 14 or args.dataset_selection == 17 or args.dataset_selection == 20 or args.dataset_selection == 18:
                    e1_with_prompt = encoded_prompt_before_e1
                else:
                    e1_with_prompt = encoded_prompt_before_e1 + e1
                rel_with_prompt = encoded_prompt_after_e1_before_r + rel + encoded_prompt_after_r
                target_with_eos_token = target + encoded_eos_token
                data_idx_ids.append(id_in_ori_dataset)
                if args.dataset_selection == 12:
                    rule_template_ids.append(cur_data_notes[dict_key2idDataset[id_in_ori_dataset]][0])
                    topic_ids.append(cur_data_notes[dict_key2idDataset[id_in_ori_dataset]][1])
                    specific_general_fact_ids.append(cur_data_notes[dict_key2idDataset[id_in_ori_dataset]][2])
                    long_short_facts_ids.append(cur_data_notes[dict_key2idDataset[id_in_ori_dataset]][3])
                    cnt_facts_ids.append(cur_data_notes[dict_key2idDataset[id_in_ori_dataset]][4])
                    full_missing_facts_ids.append(cur_data_notes[dict_key2idDataset[id_in_ori_dataset]][5])
                if len(e1_with_prompt) > args.max_e1:
                    print('Warning: max_e1 is not enough', len(e1_with_prompt), args.max_e1)
                    # raise Exception('Warning: max_e1 is not enough', len(e1_with_prompt), args.max_e1)
                    print('Warning: max_e1 is not enough', len(e1_with_prompt), args.max_e1)
                    e1_with_prompt = e1_with_prompt[:args.max_e1]
                if len(rel_with_prompt) > args.max_r:
                    print('Warning: max_r is not enough', len(rel_with_prompt), args.max_r)
                    # raise Exception('Warning: max_r is not enough', len(rel_with_prompt), args.max_r)
                    print('Warning: max_r is not enough', len(rel_with_prompt), args.max_r)
                    rel_with_prompt = rel_with_prompt[:args.max_r]
                # N: here target_with_eos_token uses the limit of e2 (this limit is not necessary though)
                if len(target_with_eos_token) > args.max_e2:
                    print('Warning: max_e2 is not enough', len(target_with_eos_token), args.max_e2)
                    # raise Exception('Warning: max_e2 is not enough', len(target_with_eos_token), args.max_e2)
                    print('Warning: max_e2 is not enough', len(target_with_eos_token), args.max_e2)
                    target_with_eos_token = target_with_eos_token[:args.max_e2]
                # input_ids
                input_ids[idx_for_input_ids, (args.max_e1 + args.max_r - len(e1_with_prompt+rel_with_prompt)):(args.max_e1 + args.max_r)] = e1_with_prompt + rel_with_prompt
                # input_ids[i, :len(e1_with_prompt+rel_with_prompt)] = e1_with_prompt + rel_with_prompt
                # input_ids[i, args.max_e1:args.max_e1 + len(rel_with_prompt)] = rel_with_prompt
                input_ids[idx_for_input_ids, args.max_e1+args.max_r:args.max_e1+args.max_r+len(target_with_eos_token)] = target_with_eos_token
                if idx_for_input_ids == len(dataset)-1:
                    print("e1:", tokenizer_generator.decode(e1_with_prompt), "rel:", tokenizer_generator.decode(rel_with_prompt), "target:", tokenizer_generator.decode(target_with_eos_token))
                    # print("input_ids:", input_ids[i])

        # input_mask
        input_mask = (input_ids != encoded_pad_token)
        # print("input_mask:", input_mask[len(dataset)-1])
        # lm_labels; here we can overlook the lm_labels
        lm_labels = mask_value_on_lm_models * np.full((n_data, input_len * len(rule_format_collection)), fill_value=1, dtype=np.int64)
        lm_labels = np.copy(input_ids)
        lm_labels[input_ids == encoded_pad_token] = mask_value_on_lm_models
        lm_labels[:, :args.max_e1 + args.max_r] = mask_value_on_lm_models

        # QQ: to prevent label's effect on prediction
        input_ids[:, args.max_e1+args.max_r:] = encoded_pad_token
        input_mask[:, args.max_e1+args.max_r:] = 0

        if args.dataset_selection == 12:
            tensor_datasets.append((torch.tensor(input_ids), torch.tensor(input_mask).to(torch.float32), torch.tensor(lm_labels), torch.tensor(data_idx_ids), torch.tensor(rule_template_ids), torch.tensor(topic_ids), torch.tensor(specific_general_fact_ids), torch.tensor(long_short_facts_ids), torch.tensor(cnt_facts_ids), torch.tensor(full_missing_facts_ids)))
        else:
            tensor_datasets.append((torch.tensor(input_ids), torch.tensor(input_mask).to(torch.float32), torch.tensor(lm_labels), torch.tensor(data_idx_ids)))

    return tensor_datasets



def tokenize_and_encode(obj, tokenizer, model_type=None):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        # As dpr's tokenizers are the same as BertTokenizer
        if 'bert' in model_type or 'dpr' in model_type or 'bart' in model_type:
            # bert tokenizer will automatically add [CLS] at beginning and [SEP] at end; while gpt tokneizer don't
            # we will consider [CLS] and [SEP] seperately
            return tokenizer.encode(obj)[1:-1]
        elif 'gpt2' in model_type or 'gpt-j' in model_type or 'gpt-neo' in model_type:
            return tokenizer.encode(obj)
        elif 't5' in model_type:
            return tokenizer.encode(obj)[:-1]
        elif 'llama' in model_type or 'vicuna' in model_type:
            return tokenizer.encode(obj)[1:]
        else:
            raise Exception("Not supported model_type: ", model_type)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        return None
    return list(tokenize_and_encode(o, tokenizer, model_type=model_type) for o in obj)


# INPUT:
#   data_type: 'train' / 'val' / 'test'
# OUTPUT:
#   full_datasets: [#size(event, reason)]
def load_data_whyQA_Module123(args, data_type):
    assert args.dataset_selection == 9 or args.dataset_selection == 9.5 or args.dataset_selection == 10 or args.dataset_selection == 11
    assert data_type == 'train' or data_type == 'val' or data_type == 'test'
    full_datasets = []

    if args.dataset_selection == 9 or args.dataset_selection == 9.5:
        if args.dataset_selection == 9:
            prefix = 'Module1_'
        elif args.dataset_selection == 9.5:
            prefix = 'Module1_withRetrieval_'
        with open(os.path.join(args.root_data_dir, prefix+data_type+'.txt'), 'r') as f:
            lines = f.readlines()
            for id_line, line in enumerate(lines):
                line = line.strip('\n').split('[SEP]')
                assert len(line) == 3
                retrieved_doc, event, cause = line[0], line[1], line[2]
                full_datasets.append((retrieved_doc, event, cause, id_line))
    elif args.dataset_selection == 10:
        with open(os.path.join(args.root_data_dir, 'Module2_'+data_type+'.txt'), 'r') as f:
            lines = f.readlines()
            for id_line, line in enumerate(lines):
                # line = line.strip('\n').split('\t')
                # N: use [SEP] instead of '\t'
                line = line.strip('\n').split('[SEP]')
                assert len(line) == 3
                event, cause, label = line[0], line[1], line[2]
                full_datasets.append((event, cause, label, id_line))
    elif args.dataset_selection == 11:
        with open(os.path.join(args.root_data_dir, 'Module3_'+data_type+'.txt'), 'r') as f:
            lines = f.readlines()
            for id_line, line in enumerate(lines):
                line = line.strip('\n').split('[SEP]')
                assert len(line) == 3
                retrieved_doc, event, label = line[0], line[1], line[2]
                full_datasets.append((retrieved_doc, event, label, id_line))

    return full_datasets


# thresholds for M2/3/4/5 with new ICDs; could be used for bleu_green_calculator_analysis.py
def threshold_storer(args):
    assert args.generator_model_type == "gptneo125M" or args.generator_model_type == "gptneo1.3B" or args.generator_model_type == "gptneo2.7B" or args.generator_model_type == "gptj" or args.generator_model_type == "gptneox20B" or args.generator_model_type == "llama" or args.generator_model_type == "vicunallama"
    # maybe thresholds for M2/3/4/5 should be independent with M1
    # assert args.setting_selection_M1_forM2M3 == 0 or args.setting_selection_M1_forM2M3 == 1 or args.setting_selection_M1_forM2M3 == 2 or args.setting_selection_M1_forM2M3 == 3
    assert args.setting_selection == 0 or args.setting_selection == 1 or args.setting_selection == 2 or args.setting_selection == 3
    assert args.if_already_fintuned_for_test == 0 or args.if_already_fintuned_for_test == 1

    # random filing baseline
    if "baseline_template_with_random_filling" in args.output_dir:
        assert args.if_consider_M234 == 0
        return None, None, None, None



    # CoLM related baseline and models
    # Q: only temporaly thresholds
    if args.generator_model_type == "llama":
        # 'yes' / 'no'
        # thres13 = 0.61
        # thres14 = 0.55
        # thres19 = 0.64
        # thres20 = 0.66
        # 'Yes' / 'No'
        tmp_add = 0.05
        thres13 = 0.41 + tmp_add
        thres14 = 0.335
        thres19 = 0.435 + tmp_add + 0.05
        thres20 = 0.370 + tmp_add
    elif args.generator_model_type == "vicunallama":
        # thres13 = 0.61
        thres13 = 0.50
        thres14 = 0.55
        thres19 = 0.64
        thres20 = 0.54
    elif args.generator_model_type == "gptneo125M":
        if args.setting_selection == 2:
            thres13 = 0.795
            thres14 = 0.305
            thres19 = 0.88
            thres20 = 0.78
        else:
            raise NotImplementError
    elif args.generator_model_type == "gptneo1.3B":
        if args.setting_selection == 2:
            thres13 = 0.34
            thres14 = 0.145
            thres19 = 0.235
            thres20 = 0.15
        else:
            raise NotImplementError
    elif args.generator_model_type == "gptneo2.7B":
        if args.setting_selection == 2:
            # thres13 = 0.24
            thres13 = 0.20
            thres14 = 0.115
            thres19 = 0.425
            # thres20 = 0.295
            thres20 = 0.225
        else:
            raise NotImplementError
    elif args.generator_model_type == "gptj":
        if args.setting_selection == 2:
            thres13 = 0.47
            thres14 = 0.455
            thres19 = 0.48
            thres20 = 0.47
        elif args.setting_selection == 3 and args.if_already_fintuned_for_test == 1:
            # thres13 = 0.765
            # thres14 = 0.45
            # thres19 = 0.79
            # thres20 = 0.82

            thres13 = 0.405
            thres14 = 0.10
            thres19 = 0.79
            thres20 = 0.88
        else:
            raise NotImplementError
    elif args.generator_model_type == "gptneox20B":
        if args.setting_selection == 2:
            # original
            # thres13 = 0.43
            # thres14 = 0.645
            # thres19 = 0.44
            # thres20 = 0.54

            # thres13 = 0.53
            # thres14 = 0.745
            # thres19 = 0.54
            # thres20 = 0.64

            thres13 = 0.47
            thres14 = 0.745
            thres19 = 0.54
            thres20 = 0.62
        else:
            raise NotImplementError
    else:
        raise NotImplementError



    return thres13, thres14, thres19, thres20
