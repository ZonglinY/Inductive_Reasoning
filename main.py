import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import argparse, logging, sys, random, datetime, math, time, shutil, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "..")
# cached_path
from transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW,
                            get_linear_schedule_with_warmup)
# from transformers import (BartForConditionalGeneration, BartTokenizer, BartConfig)
from transformers import (T5ForConditionalGeneration, T5Tokenizer, T5Config)
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config)
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoXConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJConfig
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils import (set_seed, add_special_tokens, load_data_whyQA_Module123, load_data_Hypothetical_Induction_Module123, load_data_Hypothetical_Induction_only_Module23, tokenize_and_encode, preprocess_datasets_Module123_hypothetical_induction_leftPadding, batch_step, evaluate)

logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)
# device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_model_type", type=str, default="gpt2-lmhead",
                        help="model type: bart-base/t5-base/gpt2-lmhead/gptj/llama/vicunallama(named it vicunallama but not vicuna to reuse the hyperparameter of llama)/t5flan")
    parser.add_argument("--toy", action="store_true", help="test code")
    parser.add_argument("--do_train", action="store_true", help="do training")
    parser.add_argument("--do_test", action="store_true", help="do testing")
    # parser.add_argument("--do_eval", action="store_true", help="do evaluation in the end")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_dataset", type=str, nargs="+", default=["./Data/conceptnet/train100k_CN_sorted.txt"])
    parser.add_argument("--eval_dataset", type=str, nargs="+", default=["./Data/conceptnet/dev1_CN_sorted.txt"])
    parser.add_argument("--test_dataset", type=str, nargs="+", default=["./Data/conceptnet/test_CN_sorted.txt"])

    parser.add_argument("--max_e1", type=int, default=24)
    parser.add_argument("--max_r", type=int, default=10)
    parser.add_argument("--max_e2", type=int, default=36)

    parser.add_argument("--seed", type=int, default=123)
    # parser.add_argument("--no_pretrain", action="store_true", help="w/o pretrained parameters initialized")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dev_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument("--eval_per_steps", type=int, default=150)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.002)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    # added
    parser.add_argument("--root_data_dir", type=str, default="~/openWorld_Analysis_Inductive_Reasoning_PLM/Data/", help="data dir for current dataset")
    parser.add_argument("--shared_data_dir", type=str, default="~/Shared_data/", help="data dir for shared data between experiments (e.g. tensor of current dataset)")
    parser.add_argument("--dataset_selection", type=float, default=12, help="0~4: standard ParaRules Mod0~4; 5: raw inductive reasoning dataset (contain bug); 6: inductive reasoning dataset with no synonym; 7: inductive reasoning dataset with half synonym; 8: inductive reasoning dataset with full synonym; 9: Module 1 (generate rules that can explain the given event); 9.5: Module 1 with retrieval; 10: Module 2 (predict whether the rule can exolain/casual the given event); 11: Module 3 (predict whether the rule is possible to happen or has already happened); 12: Rule Proposer; 13: Deduction Consistency Evaluator, using input data generated by Rule Proposer; 14: Indiscriminate Comfirmation Handler, using input data generated by Rule Proposer; 15: M2&M3 data: whether the induced rule is more general than the given facts; 16: M2&M3 data: whether the induced rule is deductively consistent with the given facts; 17: M2&M3 data: whether the induced rule fits commonsense; 18: M2&M3 data: whether the induced rule is not trivial; 19: Module 4 (if general); 20: Module 5 (if trivial), use DEER data not DEERLET;")
    parser.add_argument("--smooth_score", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3, help='for early stopping')
    parser.add_argument("--if_train_from_scratch", type=int, default=0, help="0: do not train from scratch and train from initialized PLM; 1: train PLM from scratch")
    parser.add_argument("--num_return_sequences", type=int, default=2, help="num_return_sequences for generate()")
    parser.add_argument("--setting_selection", type=int, default=0, help="0: zero-shot setting; 1: few-shot setting; 2: few-shot + chain of thought setting; 3: finetuning setting")
    parser.add_argument("--num_gene_times", type=int, default=1, help="call generate() num_gene_times times for each input sentence; basically num_gene_times has the same target with num_return_sequences, but can be implemented in a GPU-restriced way.; only be used when args.dataset_selection == 12")
    # different config of facts
    parser.add_argument("--if_long_or_short_facts", type=int, default=1, help="when 0, use long facts to induce rules; when 1, use short facts to induce rules")
    parser.add_argument("--cnt_facts_as_input", type=int, default=3, help="can be 1/2/3, indicates how many facts to use to induce rules")
    # if_full_or_missing_facts not fully implemented
    parser.add_argument("--if_full_or_missing_facts", type=int, default=0, help="when 0, use full facts; when 1, only use part of the fact to induce rules")
    # parser.add_argument("--if_add_adversarial_examples_to_test_data", type=int, default=1, help="0: do not add adversarial examples to test data; 1: add adversarial examples to test data")
    parser.add_argument("--setting_selection_M1_forM2M3", type=int, default=1, help="used to identify which generated rules set to filter, useful when dataset_selection==13/14/15/16/17/18/19/20; current choices are 0/1")
    parser.add_argument("--if_use_deer_train_data_for_test", type=int, default=0, help="should be used when: 1. only used in --do_test but not --do_train; 2. only used when dataset_selection == 12; FUNCTION: rule proposer do test on deer train data (for annotation of train set of deerlet data)")
    parser.add_argument("--if_use_deerlet_val_train_for_test", type=int, default=0, help="only useful when dataset_selection == 15/16/17/18, and should not be used with --do_train; 0: use deerlet test for test; 1. use deerlet val for test; 2: use deerlet train for test")
    parser.add_argument("--if_already_fintuned_for_test", type=int, default=0, help="always 0, unless when using finetuned checkpoint to only test")
    parser.add_argument("--finetuned_checkpoint_dir", type=str, default="", help="always not used, unless when using finetuned checkpoint to only test")
    parser.add_argument("--if_capital_yesno", type=int, default=1, help="whether to use capitalized yes/no in in-context demonstrations; as a start, it is recommended to use captical for gptneo 125M/1.3B/6B; and use lower case for gpt neo 2.7B/20B.")
    parser.add_argument("--min_length_rule_to_be_considered", type=int, default=45, help="the min length of generated rule to be collected for human annotation; in the first 5 train files (train_human_eval_rlt_M1setting_1_0/1/2/3/4.pt) and first 2 test files (human_eval_rlt_M1setting_1_0/1.pt), the value of this hyperparameter is 0, while for others should be 45; this should be 0 for checkpoint gptj_analysis_100test_newdata_newprompt but 45 for gptj_analysis_100test_newdata_newprompt_10")
    args = parser.parse_args()

    # # prevent from using wrong match output_dir
    # if 'gpt2' in args.generator_model_type:
    #     assert 'gpt2' in args.output_dir
    # elif 'bart' in args.generator_model_type:
    #     assert 'bart' in args.output_dir
    assert args.if_train_from_scratch == 0 or args.if_train_from_scratch == 1
    assert args.num_gene_times >= 1
    # assert args.if_add_adversarial_examples_to_test_data == 0 or args.if_add_adversarial_examples_to_test_data == 1
    # different config of facts
    assert args.if_long_or_short_facts == 0 or args.if_long_or_short_facts == 1
    assert args.cnt_facts_as_input == 1 or args.cnt_facts_as_input == 2 or args.cnt_facts_as_input == 3
    assert args.if_full_or_missing_facts == 0 or args.if_full_or_missing_facts == 1
    assert args.if_use_deer_train_data_for_test == 0 or args.if_use_deer_train_data_for_test == 1
    if args.dataset_selection == 12 and args.do_train:
        # since we do not have extra data for in-context demonstrations
        assert args.setting_selection_M1_forM2M3 == 0 and args.setting_selection == 0
        assert args.num_gene_times == 1
    if args.if_use_deer_train_data_for_test == 1:
        assert args.dataset_selection == 12
        assert args.do_test
        assert not args.do_train
    assert args.if_use_deerlet_val_train_for_test == 0 or args.if_use_deerlet_val_train_for_test == 1 or args.if_use_deerlet_val_train_for_test == 2
    if args.if_use_deerlet_val_train_for_test > 0:
        assert not args.do_train
    assert args.if_already_fintuned_for_test == 0 or args.if_already_fintuned_for_test == 1
    if args.if_already_fintuned_for_test == 1:
        assert not args.do_train
        assert args.setting_selection == 0 or args.setting_selection == 3
        # assert args.setting_selection_M1_forM2M3 == 0
        assert (args.dataset_selection == 13 or args.dataset_selection == 14 or args.dataset_selection == 19 or args.dataset_selection == 20) or (args.dataset_selection == 15 or args.dataset_selection == 16 or args.dataset_selection == 17 or args.dataset_selection == 18)
    assert args.if_capital_yesno == 1



    # special configuration for different dataset_selection
    if not (args.dataset_selection == 9 or args.dataset_selection == 9.5 or args.dataset_selection == 10 or args.dataset_selection == 11 or (args.dataset_selection >= 12 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20):
        raise Exception("This code currently only support open world setting experiments.")
    # setting_selection restrictions
    if args.setting_selection == 0:
        pass
    elif args.setting_selection == 1:
        pass
    elif args.setting_selection == 2:
        if args.dataset_selection == 12:
            raise NotImplementError
    elif args.setting_selection == 3:
        if args.dataset_selection == 12:
            raise NotImplementError
    else:
        raise NotImplementError
    # generator_model_type restrictions
    assert args.generator_model_type == "gpt2-lmhead" or args.generator_model_type == "gptj" or args.generator_model_type == "gptneo125M" or args.generator_model_type == "gptneo1.3B" or args.generator_model_type == "gptneo2.7B" or args.generator_model_type == "gptneox20B" or "t5" in args.generator_model_type or "llama" in args.generator_model_type or "vicuna" in args.generator_model_type
    # elif not args.dataset_selection == 9:
    #     raise NotImplementError("Current code only develop Module 1 function")
    if args.dataset_selection == 9 or args.dataset_selection == 9.5 or args.dataset_selection == 10 or args.dataset_selection == 11:
        args.root_data_dir = "~/openWorld_Analysis_Inductive_Reasoning_PLM/Data/whyQA/"
        # max length of event/observation
        args.max_e1 = 700
        # max length of rules
        args.max_r = 150
        # max length of possible cause
        args.max_e2 = 150
    elif args.dataset_selection >= 12 and args.dataset_selection <= 20:
        args.root_data_dir = "~/openWorld_Analysis_Inductive_Reasoning_PLM/Data/"
        if args.generator_model_type == "gpt2-lmhead":
            if args.setting_selection == 0 or args.setting_selection == 3:
                # max length of facts
                args.max_e1 = 650
                # max length of rule template
                args.max_r = 200
                # max length of rule
                args.max_e2 = 150
            elif args.setting_selection == 1 or args.setting_selection == 2:
                # max length of facts
                args.max_e1 = 750
                # max length of rule template
                args.max_r = 180
                # max length of rule
                args.max_e2 = 70
            else:
                raise NotImplementError
        elif args.generator_model_type == 'gptj' or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
            # max length of facts
            if args.setting_selection == 0 or args.setting_selection == 3:
                args.max_e1 = 700
            elif args.setting_selection == 1 or args.setting_selection == 2:
                if args.dataset_selection == 12:
                    args.max_e1 = 1750
                elif args.dataset_selection == 13 or args.dataset_selection == 16:
                    args.max_e1 = 1600
                # list 15 seperately to prevent not enough GPU memory error
                elif args.dataset_selection == 19 or args.dataset_selection == 15:
                    args.max_e1 = 1600
                elif args.dataset_selection == 14 or args.dataset_selection == 17 or args.dataset_selection == 18 or args.dataset_selection == 20:
                    args.max_e1 = 1100
                else:
                    raise NotImplementError
            else:
                raise NotImplementError
            # max length of rule template
            args.max_r = 180
            # max length of rule
            args.max_e2 = 90
        elif "t5" in args.generator_model_type:
            if args.setting_selection == 0 or args.setting_selection == 3:
                args.max_e1 = 200
            else:
                raise NotImplementError
            # max length of rule template
            args.max_r = 180
            # max length of rule
            args.max_e2 = 90
        else:
            raise NotImplementError
    else:
        raise NotImplementError

    # when use gptj, if we only have 2 gpus, then the batch_size can only be 1
    if "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or 't5-11B' in args.generator_model_type or "llama" in args.generator_model_type:
        # args.train_batch_size = 1
        # args.dev_batch_size = 1
        # args.test_batch_size = 1
        if args.setting_selection == 0 or args.setting_selection == 3:
            if args.do_train:
                args.num_return_sequences = 1
            else:
                args.num_return_sequences = 3
        elif args.setting_selection == 1 or args.setting_selection == 2:
            args.num_return_sequences = 1
        else:
            raise NotImplementError


    # set random seeds & check availability of GPUs
    set_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device1, n_gpu))
    # print("n_gpu: ", n_gpu)
    assert n_gpu >= 1

    # important paths
    path_tensorboard = os.path.join(args.output_dir, args.output_dir.split('/')[-1])
    path_if_finished_training = os.path.join(args.output_dir, 'training_finished.pt')
    path_generator_final = os.path.join(args.output_dir, 'generator_final_{:.0f}_{:.0f}.pt'.format(args.dataset_selection, args.setting_selection_M1_forM2M3))
    # Q: might need to change path_tensor_dataset for different data pre-processing
    path_tensor_dataset = os.path.join(args.shared_data_dir, "{:.0f}_{:.0f}_{:.0f}_{:.0f}_tensor_dataset.pt".format(args.dataset_selection, args.setting_selection, args.setting_selection_M1_forM2M3, args.if_use_deer_train_data_for_test))

    ## File systems
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output_dir is an empty file now")
    elif args.do_train and not os.path.exists(path_if_finished_training):
        # Q: not removing the files, for experiments on dataset_selection==15;data_type==train
        # # make sure the file is clear
        # shutil.rmtree(args.output_dir)
        # assert not os.path.exists(args.output_dir)
        # os.makedirs(args.output_dir)
        # print("Training not finished yet, output_dir is an empty file now")
        pass
    elif args.do_test and os.path.exists(path_if_finished_training):
        pass
    elif args.do_test:
        pass
    else:
        raise Exception("unexpected file system status")

    # tensorboard
    writer = SummaryWriter(path_tensorboard)

    # Load model and tokenizer
    MODEL_CLASSES = {
        # "bart-base": (BartForConditionalGeneration, BartTokenizer, BartConfig, "facebook/bart-base"),
        "gpt2-lmhead":(GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, "gpt2"),
        "gptj": (AutoModelForCausalLM, AutoTokenizer, GPTJConfig, 'EleutherAI/gpt-j-6B'),
        "gptneo125M": (GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig, "EleutherAI/gpt-neo-125M"),
        "gptneo1.3B": (GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig, "EleutherAI/gpt-neo-1.3B"),
        "gptneo2.7B": (GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig, "EleutherAI/gpt-neo-2.7B"),
        "gptneox20B": (GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoXConfig, "EleutherAI/gpt-neox-20b"),
        "t5-11B": (T5ForConditionalGeneration, T5Tokenizer, T5Config, "t5-11b"),
        "t5-small": (T5ForConditionalGeneration, T5Tokenizer, T5Config, "t5-small"),
        "llama": (AutoModelForCausalLM, LlamaTokenizer, None, "decapoda-research/llama-7b-hf"),
        "vicunallama": (AutoModelForCausalLM, AutoTokenizer, None, "eachadea/vicuna-7b-1.1"),
        "mpt": (AutoModelForCausalLM, GPTNeoXTokenizerFast, None, "mosaicml/mpt-7b")
    }

    Generator_Model, Generator_Tokenizer, Generator_Config, Generator_Model_Name = MODEL_CLASSES[args.generator_model_type]
    if "mpt" in Generator_Model_Name:
        tokenizer_generator = Generator_Tokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer_generator = Generator_Tokenizer.from_pretrained(Generator_Model_Name)
    tokenizer_generator = add_special_tokens(tokenizer_generator)

    model_generator = Generator_Model.from_pretrained(Generator_Model_Name, device_map="auto", torch_dtype=torch.float16)
    if args.if_train_from_scratch:
        configuration = model_generator.config
        model_generator = Generator_Model(config=configuration)
    model_generator.resize_token_embeddings(len(tokenizer_generator))

    # Load and encode the datasets
    logger.info("Loading datasets ...")
    if os.path.exists(path_tensor_dataset):
        print("Find existing tensor_datasets! Begin loading...")
        tensor_datasets = torch.load(path_tensor_dataset)
    else:
        print("Generating tensor_datasets...")
        if args.dataset_selection == 9 or args.dataset_selection == 9.5 or args.dataset_selection == 10 or args.dataset_selection == 11:
            train_datasets = load_data_whyQA_Module123(args, 'train')
            eval_datasets = load_data_whyQA_Module123(args, 'val')
            test_datasets = load_data_whyQA_Module123(args, 'test')
        # In this section, val should not be considered, as we do not have the data, and there's no need for val set
        elif args.dataset_selection == 12:
            # Only use train and test
            if args.do_train:
                if_true_rule_without_prefix = True
            else:
                if_true_rule_without_prefix = False
            train_datasets, train_datasets_notes, dict_id2trueRule_train = load_data_Hypothetical_Induction_Module123(args, 'train', if_save_dict_files=True, if_true_rule_without_prefix=if_true_rule_without_prefix)
            eval_datasets, eval_datasets_notes, dict_id2trueRule_val = load_data_Hypothetical_Induction_Module123(args, 'val', if_save_dict_files=True, if_true_rule_without_prefix=if_true_rule_without_prefix)
            if args.if_use_deer_train_data_for_test == 1:
                assert not args.do_train
                test_datasets, test_datasets_notes, dict_id2trueRule_test = load_data_Hypothetical_Induction_Module123(args, 'train', if_save_dict_files=True)
            else:
                test_datasets, test_datasets_notes, dict_id2trueRule_test = load_data_Hypothetical_Induction_Module123(args, 'test', if_save_dict_files=True, if_true_rule_without_prefix=if_true_rule_without_prefix)
        # In this section, should only consider test, no need to train or val. We can train the model when dataset_selection == 15/16/17/18, and save the model, and load the model here (dataset_selection == 13/14/19) for test
        elif args.dataset_selection == 13 or args.dataset_selection == 14 or args.dataset_selection == 19 or args.dataset_selection == 20:
            # Only use test
            # train and val (if applicable) of 13/14 should use deerlet data instead of unlabeled general of M1 using deer
            train_datasets, train_datasets_notes, dict_id2trueRule_train = load_data_Hypothetical_Induction_Module123(args, 'test', if_save_dict_files=True)
            eval_datasets, eval_datasets_notes, dict_id2trueRule_val = load_data_Hypothetical_Induction_Module123(args, 'test', if_save_dict_files=True)
            test_datasets, test_datasets_notes, dict_id2trueRule_test = load_data_Hypothetical_Induction_Module123(args, 'test', if_save_dict_files=True)
        # In this section, train / val / test should all be considered
        elif args.dataset_selection == 15 or args.dataset_selection == 16 or args.dataset_selection == 17 or args.dataset_selection == 18:
            # Use train / val / test
            # Q: only use new human annotation files
            allowed_existing_annotation_files_val = ["2"]
            allowed_existing_annotation_files_test = ["3", "4"]
            allowed_existing_annotation_files_train = ["5", "6", "7", "8", "9", "10", "11"]
            train_datasets, dict_id2trueRule_train = load_data_Hypothetical_Induction_only_Module23(args, 'train', allowed_existing_annotation_files_val, allowed_existing_annotation_files_test, allowed_existing_annotation_files_train)
            assert len(train_datasets) == 365 or len(train_datasets) == 546
            eval_datasets, dict_id2trueRule_val = load_data_Hypothetical_Induction_only_Module23(args, 'val', allowed_existing_annotation_files_val, allowed_existing_annotation_files_test, allowed_existing_annotation_files_train)
            # assert len(eval_datasets) == 50
            assert len(eval_datasets) == 100
            if args.if_use_deerlet_val_train_for_test == 0:
                test_datasets, dict_id2trueRule_test = load_data_Hypothetical_Induction_only_Module23(args, 'test', allowed_existing_annotation_files_val, allowed_existing_annotation_files_test, allowed_existing_annotation_files_train)
                # assert len(test_datasets) == 150
                assert len(test_datasets) == 200
            elif args.if_use_deerlet_val_train_for_test == 1:
                test_datasets, dict_id2trueRule_test = load_data_Hypothetical_Induction_only_Module23(args, 'val', allowed_existing_annotation_files_val, allowed_existing_annotation_files_test, allowed_existing_annotation_files_train)
            elif args.if_use_deerlet_val_train_for_test == 2:
                test_datasets, dict_id2trueRule_test = load_data_Hypothetical_Induction_only_Module23(args, 'train', allowed_existing_annotation_files_val, allowed_existing_annotation_files_test, allowed_existing_annotation_files_train)
            else:
                raise Exception
            print("len(train_datasets): ", len(train_datasets))
            print("len(eval_datasets): ", len(eval_datasets))
            print("len(test_datasets): ", len(test_datasets))
        else:
            raise NotImplementError

        # Prepare dataset for the model
        datasets = (train_datasets, eval_datasets, test_datasets)
        print("Encoding datasets ...")
        logger.info("Encoding datasets ...")
        encoded_datasets = tokenize_and_encode(datasets, tokenizer_generator, model_type=Generator_Model_Name)
        # when dataset_selection == 13/14, the classification model needs padding
        if args.dataset_selection == 9 or args.dataset_selection == 9.5 or args.dataset_selection == 10 or args.dataset_selection == 11 or (args.dataset_selection >= 12 and args.dataset_selection <= 20):
            # we don't need here since one single function of preprocess_datasets_Module123_hypothetical_induction_leftPadding can process all train/val/test data
            # if args.do_train:
                # thanks for such sweet note
                # raise NotImplementError("please use preprocess_datasets_Module123_hypothetical_induction_leftPadding() to generate train data, remember to specify data_type='train'")
            # tensor_datasets = preprocess_datasets_Module123(args, encoded_datasets, tokenizer_generator)
            if args.dataset_selection == 12:
                tensor_datasets = preprocess_datasets_Module123_hypothetical_induction_leftPadding(args, encoded_datasets, tokenizer_generator, data_notes=(train_datasets_notes, eval_datasets_notes, test_datasets_notes))
            else:
                tensor_datasets = preprocess_datasets_Module123_hypothetical_induction_leftPadding(args, encoded_datasets, tokenizer_generator)
        else:
            raise NotImplementError
        print("INFO: not saving tensor_datasets")
        # Q: not saving tensor_datasets to generate tensor_datasets each time to avoid not using the correct data
        # torch.save(tensor_datasets, path_tensor_dataset)
    train_tensor_dataset, eval_tensor_dataset, test_tensor_dataset = tensor_datasets[0], tensor_datasets[1], tensor_datasets[2]
    print('len(train_tensor_dataset[0]): ', len(train_tensor_dataset[0]))
    print('len(eval_tensor_dataset[0]):', len(eval_tensor_dataset[0]))
    print('len(test_tensor_dataset[0]): ', len(test_tensor_dataset[0]))

    # print args for record
    print(args)

    # generator_eos_id
    if args.generator_model_type == "gpt2-lmhead" or "t5" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
        generator_eos_id = tokenizer_generator.encode(tokenizer_generator.eos_token)[0]
    elif args.generator_model_type == "bart-base" or args.generator_model_type == "bart-large" or 'bert' in args.generator_model_type:
        generator_eos_id = tokenizer_generator.encode(tokenizer_generator.eos_token)[1]
    else:
        raise Exception

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    # train_sampler = torch.utils.data.DataLoader(train_data, shuffle=True)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    # eval_sampler = torch.utils.data.DataLoader(eval_data, shuffle=True)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size)

    test_data = TensorDataset(*test_tensor_dataset)
    test_sampler = SequentialSampler(test_data)
    # test_sampler = torch.utils.data.DataLoader(test_data, shuffle=True)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)


    ## begin training
    t_total = (args.num_train_epochs * len(train_dataloader)) // args.gradient_accumulation_steps
    print('num_train_epochs: ', args.num_train_epochs)
    print('t_total: ', t_total)


    num_warmup_steps = np.maximum(200, int(0.1 * t_total))
    if args.do_train and not os.path.exists(path_if_finished_training):
        # log information
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(tensor_datasets[0]))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Each Epoch has %d steps, and %d actual steps w/ accumulation",
                    len(train_dataloader), len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Total train batch size (w. accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        # initialize optimizer
        param_generator = list(model_generator.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {"params": [p for n, p in param_generator if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, 'lr': args.learning_rate},
                {"params": [p for n, p in param_generator if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': args.learning_rate}
                ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
        # Q: change the optimizer to try finetune t5-11b on A100 80G
        # optimizer = torch.optim.SGD(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
        # begin training
        global_steps = 0
        tr_loss, logging_loss = 0.0, 0.0
        tr_nll_loss, logging_nll_loss = 0.0, 0.0
        best_eval_ppl, best_accuracy, best_f1 = 1e10, 0.0, 0.0
        patience = args.patience
        num_steps_in_one_epoch = len(train_dataloader)
        model_generator.train()
        for id_epoch in range(args.num_train_epochs):
            if patience < 0:
                break
            for step, batch in enumerate(train_dataloader):
                loss, nll_loss, seq_logprobs, accuracy, righ_format_correct_proportion, rewritten_correct_proportion, right_format_proportion, yesNoRatio, batch_generation, f1_counter, true_labels = batch_step(args, batch, model_generator, tokenizer_generator, global_steps, "train")
                # shows the model input and output
                if step % args.logging_steps == 0:
                    # print generation
                    if args.dataset_selection == 12:
                        tmp_input_ids, tmp_attention_masks, tmp_lm_labels = batch[0:3]
                        # print("seq_logprobs: ", seq_logprobs)
                        value, indices = seq_logprobs.max(dim=-1)
                        sample_index = random.randint(0, args.train_batch_size - 1)
                        # print("tmp_input_ids:", tokenizer_generator.decode(tmp_input_ids[sample_index].tolist()))
                        # print("input_mask:", tmp_attention_masks[sample_index][0].tolist())
                        # print("input_lm_labels:", tmp_lm_labels[sample_index][0].tolist())
                        if step == 0:
                            print("indices.size(): ", indices.size())
                        if "gpt2" in args.generator_model_type or "gptj" in args.generator_model_type or 'gptneo' in args.generator_model_type or "llama" in args.generator_model_type:
                            output = indices[sample_index].tolist()[-(args.max_e2+1):]
                        elif "bart" in args.generator_model_type or "bert" in args.generator_model_type or "t5" in args.generator_model_type:
                            output = indices[sample_index].tolist()
                        # print("output ids:", output)
                        try:
                            eos_pos = output.index(generator_eos_id)
                            output = tokenizer_generator.decode(output[:eos_pos])
                        except:
                            output = tokenizer_generator.decode(output)
                        print("output:", output.strip())
                    # print accu and f1
                    if not args.dataset_selection == 12:
                        print("Step:", global_steps, "accuracy: {}; righ_format_correct_proportion: {}; rewritten_correct_proportion: {}; right_format_proportion: {}".format(accuracy, righ_format_correct_proportion, rewritten_correct_proportion, right_format_proportion))
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    nll_loss = nll_loss / args.gradient_accumulation_steps
                # update the model
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_generator.parameters(), args.max_grad_norm)
                tr_loss += loss.item()
                tr_nll_loss += nll_loss.item()
                # Update parameters, print results, and evaluate on val set
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_steps += 1
                    if global_steps % args.logging_steps == 0:
                        loss = (tr_loss - logging_loss)/args.logging_steps
                        nll_loss = (tr_nll_loss - logging_nll_loss)/args.logging_steps
                        PPL = np.exp(nll_loss) if nll_loss < 300 else np.inf
                        print("Step:", global_steps, "Training Loss:", loss, "Nll Loss:", nll_loss, "Smooth loss:", loss-nll_loss, "ppl:", PPL)
                        writer.add_scalar('Train Loss', loss, global_step=global_steps)
                        writer.add_scalar('Train PPL', PPL, global_step=global_steps)
                        if args.dataset_selection == 10 or args.dataset_selection == 11 or (args.dataset_selection >= 15 and args.dataset_selection <= 18):
                            # only use accuracy for Module 2 and Module 3
                            writer.add_scalar("Train accuracy", accuracy, global_step=global_steps)
                            writer.add_scalar("Train righ_format_correct_proportion", righ_format_correct_proportion, global_step=global_steps)
                            writer.add_scalar("Train rewritten_correct_proportion", rewritten_correct_proportion, global_step=global_steps)
                            writer.add_scalar("Train right_format_proportion", right_format_proportion, global_step=global_steps)
                        logging_loss = tr_loss
                        logging_nll_loss = tr_nll_loss
                    # Evaluate on val set during train time
                    if global_steps % args.eval_per_steps == 0 and global_steps != 0:
                        # to avoid CUDA out of memory during eval time
                        # torch.cuda.empty_cache()
                        model_generator.eval()
                        eval_loss, eval_accuracy, eval_f1, eval_righ_format_correct_proportion, eval_rewritten_correct_proportion, eval_right_format_proportion, eval_ttl_yesNoRatio, eval_averaged_precision = evaluate(args, model_generator, tokenizer_generator, eval_dataloader, "val", dict_id2trueRule_val, best_accuracy)
                        eval_ppl = np.exp(eval_loss) if eval_loss < 300 else np.inf
                        print("\n\nevaluating\neval loss:", eval_loss, "ppl", eval_ppl)
                        writer.add_scalar('Eval Loss', eval_loss, global_step=global_steps)
                        writer.add_scalar('Eval PPL', eval_ppl, global_step=global_steps)
                        # writer.add_scalar('Eval Acc', eval_accuracy, global_step=global_steps)
                        # writer.add_scalar('Eval F1', eval_f1, global_step=global_steps)
                        if args.dataset_selection == 9 or args.dataset_selection == 9.5 or args.dataset_selection == 12:
                            # early stopping
                            if eval_ppl < best_eval_ppl:
                                assert args.if_already_fintuned_for_test == 0
                                torch.save(model_generator.state_dict(), path_generator_final)
                                print("model saved at step", global_steps)
                                print("global_steps: ", global_steps, "prev eval ppl:", best_eval_ppl, "cur eval ppl:", eval_ppl)
                                best_eval_ppl = eval_ppl
                                patience = args.patience
                            else:
                                patience -= 1
                                print("patience: ", patience)
                                if patience < 0:
                                    break
                        elif args.dataset_selection == 10 or args.dataset_selection == 11 or args.dataset_selection == 13 or args.dataset_selection == 14 or (args.dataset_selection >= 15 and args.dataset_selection <= 18) or args.dataset_selection == 19 or args.dataset_selection == 20:
                            # only use accuracy for Module 2 and Module 3
                            print("global_steps: ", global_steps, "eval_accuracy: {}; eval_f1: {}, eval_righ_format_correct_proportion: {}; eval_rewritten_correct_proportion: {}; eval_right_format_proportion: {}".format(eval_accuracy, eval_f1, eval_righ_format_correct_proportion, eval_rewritten_correct_proportion, eval_right_format_proportion))
                            writer.add_scalar("Eval accuracy", eval_accuracy, global_step=global_steps)
                            writer.add_scalar("Eval F1", eval_f1, global_step=global_steps)
                            writer.add_scalar("Eval righ_format_correct_proportion", eval_righ_format_correct_proportion, global_step=global_steps)
                            writer.add_scalar("Eval rewritten_correct_proportion", eval_rewritten_correct_proportion, global_step=global_steps)
                            writer.add_scalar("Eval right format proportion", eval_right_format_proportion, global_step=global_steps)
                            # early stopping; Here use ppl as metric instead of f1 or accuracy
                            # if eval_f1 > best_f1:
                            if eval_accuracy > best_accuracy:
                            # if eval_ppl < best_eval_ppl:
                                assert args.if_already_fintuned_for_test == 0
                                torch.save(model_generator.state_dict(), path_generator_final)
                                print("model saved at step", global_steps)
                                # eval_f1
                                # print("prev f1:", best_f1, "cur f1:", eval_f1)
                                # best_f1 = eval_f1
                                # eval_acc
                                print("prev accuracy:", best_accuracy, "cur accuracy:", eval_accuracy)
                                best_accuracy = eval_accuracy
                                # eval_ppl;
                                # print("prev eval ppl:", best_eval_ppl, "cur eval ppl:", eval_ppl)
                                # best_eval_ppl = eval_ppl
                                # patience
                                patience = args.patience
                            else:
                                patience -= 1
                                print("patience: ", patience)
                                if patience < 0:
                                    break
                        else:
                            raise NotImplementError
                        # return model status to .train() and prepare further training
                        # torch.cuda.empty_cache()
                        model_generator.train()
        torch.save(torch.ones(1,1), path_if_finished_training)
        if patience < 0:
            print("Early breaking happens!")

    # End of training, begin testing
    if args.do_test:
        with torch.no_grad():
            # Begin testing
            # if the model has been trained, then we should load finetuned model. otherwise we do not need re-load
            # Q: might need to figure out how to load in parallel using load_state_dict
            if args.do_train or args.if_already_fintuned_for_test:
                # find path_generator_final for args.if_already_fintuned_for_test == 1
                if args.if_already_fintuned_for_test == 1:
                    # dataset_selection_during_finetuning
                    if args.dataset_selection == 13:
                        dataset_selection_during_finetuning = 16
                    elif args.dataset_selection == 14:
                        dataset_selection_during_finetuning = 17
                    elif args.dataset_selection == 19:
                        dataset_selection_during_finetuning = 15
                    elif args.dataset_selection == 20:
                        dataset_selection_during_finetuning = 18
                    elif args.dataset_selection == 15 or args.dataset_selection == 16 or args.dataset_selection == 17 or args.dataset_selection == 18:
                        dataset_selection_during_finetuning = args.dataset_selection
                    else:
                        raise NotImplementError
                    setting_selection_M1_forM2M3_finetuning = 0
                    path_generator_final = os.path.join(args.finetuned_checkpoint_dir, 'generator_final_{:.0f}_{:.0f}.pt'.format(dataset_selection_during_finetuning, setting_selection_M1_forM2M3_finetuning))
                # load model to cuda, some model might need to be parallelized
                model_generator.load_state_dict(torch.load(path_generator_final, map_location='cuda'), strict=False)
            model_generator.eval()
            # evaluate
            print("len(test_dataloader): ", len(test_dataloader))
            test_loss, test_accuracy, test_f1, test_righ_format_correct_proportion, test_rewritten_correct_proportion, test_right_format_proportion, test_ttl_yesNoRatio, test_averaged_precision = evaluate(args, model_generator, tokenizer_generator, test_dataloader, "test", dict_id2trueRule_test)
            # saving results...
            if args.dataset_selection >= 13 and args.dataset_selection <= 20:
                if args.dataset_selection == 13:
                    # output_file_name = 'Module2_results_{:.0f}_{:.0f}.csv'.format(args.setting_selection, args.setting_selection_M1_forM2M3)
                    output_file_name = 'Module2_results_{:.0f}_{:.0f}_{:.0f}.csv'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)
                elif args.dataset_selection == 14:
                    output_file_name = 'Module3_results_{:.0f}_{:.0f}_{:.0f}.csv'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)
                elif args.dataset_selection == 19:
                    output_file_name = 'Module4_results_{:.0f}_{:.0f}_{:.0f}.csv'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)
                elif args.dataset_selection == 20:
                    output_file_name = 'Module5_results_{:.0f}_{:.0f}_{:.0f}.csv'.format(args.setting_selection, args.setting_selection_M1_forM2M3, args.if_already_fintuned_for_test)
                elif args.dataset_selection >= 15 and args.dataset_selection <= 18:
                    # assert args.if_already_fintuned_for_test == 0
                    assert args.if_already_fintuned_for_test == 0 or (args.setting_selection == 3 and args.if_use_deerlet_val_train_for_test == 1)
                    # output_file_name = 'M2M3_rlt_'+str(args.dataset_selection)+'_'+str(args.setting_selection)+'.csv'
                    output_file_name = 'M2M3_rlt_{:.0f}_{:.0f}_{:.0f}.csv'.format(args.dataset_selection, args.setting_selection, args.if_use_deerlet_val_train_for_test)
                else:
                    raise NotImplementError
                # save test_datasets for understanding the classification results
                with open(os.path.join(args.output_dir, output_file_name), 'w', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["data_id", "fact", "generated_rule", "golden_rule", "yesNoRatio"])
                    # test_datasets will not be influenced by num_gene_times in dataset_selection 13 and 14
                    for id_query in range(len(test_datasets)):
                        writer.writerow([test_datasets[id_query][3], test_datasets[id_query][0], test_datasets[id_query][1], test_datasets[id_query][2], test_ttl_yesNoRatio[id_query].item()*100])
            print("\n\ntesting\ntest loss:", test_loss, "ppl:", np.exp(test_loss) if test_loss < 300 else np.inf)
            print("test_accuracy: {}; test_f1: {}; test_averaged_precision: {}; test_righ_format_correct_proportion: {}; test_rewritten_correct_proportion: {}; test_right_format_proportion: {}".format(test_accuracy, test_f1, test_averaged_precision, test_righ_format_correct_proportion, test_rewritten_correct_proportion, test_right_format_proportion))

if __name__ == "__main__":
    main()
