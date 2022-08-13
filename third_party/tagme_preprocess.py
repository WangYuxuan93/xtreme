# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, 
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
import json
import jsonlines
import string
from tkinter.messagebox import NO

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
  WEIGHTS_NAME,
  AdamW,
  AlbertConfig,
  AlbertForQuestionAnswering,
  AlbertTokenizer,
  BertConfig,
  BertForQuestionAnswering,
  BertTokenizer,
  DistilBertConfig,
  DistilBertForQuestionAnswering,
  DistilBertTokenizer,
  XLMConfig,
  XLMForQuestionAnswering,
  XLMTokenizer,
  XLNetConfig,
  XLNetForQuestionAnswering,
  XLNetTokenizer,
  get_linear_schedule_with_warmup,
  XLMRobertaTokenizer,
  MEAEConfig,
  MEAEForQuestionAnswering,
)
from xlm_roberta import XLMRobertaForQuestionAnswering, XLMRobertaConfig

MODEL_CLASSES = {
  "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
  "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
  "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
  "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
  "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
  "xlm-roberta": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
  "meae": (MEAEConfig, MEAEForQuestionAnswering, XLMRobertaTokenizer)
}

from processors.squad import (
  SquadResult,
  SquadV1Processor,
  SquadV2Processor,
  squad_convert_examples_to_features
)

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

try:
  import tagme
  tagme.GCUBE_TOKEN = "9a00adf4-cd6a-407a-a491-d66381f3ed13-843339462"
except:
  print ("Failed to load TAGME!")

logger = logging.getLogger(__name__)

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
  return tensor.detach().cpu().tolist()


def get_external_mention_boundary(features, example_indices, max_seq_length=384, 
                                  language="en", threshold=0.2, tagme_data=None, offset=-1): 
  mention_boundaries = []
  for i, example_index in enumerate(example_indices):
    #if offset > 0:
    #  cache_idx = example_index.item() - offset
    #if tagme_data is not None and len(tagme_data) > cache_idx:
    if tagme_data is not None and example_index.item() in tagme_data:
      mb_label = tagme_data[example_index.item()]
    else:
      eval_feature = features[example_index.item()]
      #print ("input_ids:\n", batch[0][i])
      #print ("feature:\ntoken_to_orig_map:\n", eval_feature.token_to_orig_map)
      tokenized_tokens = eval_feature.tokens
      #print ("tokens:\n", eval_feature.tokens)
      orig_tokens = []
      token_to_orig_map = {}
      for idx, tok in enumerate(tokenized_tokens):
        if tok in ["<s>","</s>"] or tok in string.punctuation: 
          orig_tokens.append(tok) 
        elif tok.startswith("▁"):
          orig_tokens.append(tok[1:])
        else:
          orig_tokens[-1] = orig_tokens[-1] + tok
        token_to_orig_map[idx] = len(orig_tokens)-1
      orig_to_token_map = {}
      for tok, orig in token_to_orig_map.items():
        if orig not in orig_to_token_map:
          orig_to_token_map[orig] = []
        orig_to_token_map[orig].append(tok)
      #print ("orig tokens:\n", orig_tokens)
      #print ("orig_to_token_map:\n", orig_to_token_map)
      seq = " ".join(orig_tokens)
      mention_result = tagme.mentions(seq, lang=language)
      mention_preds = []
      if mention_result:
        for mention in mention_result.mentions:
          if mention.linkprob > threshold:
            mention_preds.append(mention)
            #print (mention)
      cur_start, cur_end = 0, 0
      mid = 0
      mb_label = [0] * len(tokenized_tokens)
      for idx, token in enumerate(orig_tokens):
        cur_start = cur_end
        cur_end += len(token)
        #print ("cur start:{}, end:{}, mention start:{}, end:{}".format(cur_start, cur_end, mention_preds[mid].begin, mention_preds[mid].end))
        if mid < len(mention_preds) and mention_preds[mid].begin == cur_start:
          tok_ids = orig_to_token_map[idx]
          mb_label[tok_ids[0]] = 1
          if mid < len(mention_preds) and mention_preds[mid].end >= cur_end:
            for tok_id in tok_ids[1:]:
              mb_label[tok_id] = 2
            if mention_preds[mid].end <= cur_end:
              mid += 1 
        elif mid < len(mention_preds) and mention_preds[mid].begin < cur_start and mention_preds[mid].end >= cur_end:
          for tid in orig_to_token_map[idx]:
            mb_label[tid] = 2
          if mention_preds[mid].end <= cur_end:
            mid += 1
        cur_end += 1
    while len(mb_label) < max_seq_length:
      mb_label.append(-100)
    #print ("mb_label:\n", mb_label)
    mention_boundaries.append(mb_label)
    #exit()
  mention_boundaries_ = torch.from_numpy(np.array(mention_boundaries)).long()
  return mention_boundaries_, mention_boundaries

def preprocess(args, tokenizer, split='dev', prefix="", language='en', lang2id=None):
  dataset, examples, features = load_and_cache_examples(args, tokenizer, split, output_examples=True,
                              language=language, lang2id=lang2id)


  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # Eval!
  logger.info("***** Running evaluation {} *****".format(prefix))
  logger.info("  Num examples = %d", len(dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  
  tag_part = False
  if args.begin > 0 and args.end > 0:
    tag_part = True

  #tagme_mbs = []
  dataset_file = args.valid_file if split=='dev' else args.predict_file.replace("<lc>", language)
  if tag_part:
    tagme_file = dataset_file + ".tagme_prediction.maxseq_{}_{}-{}.jsonl".format(args.max_seq_length, args.begin, args.end)
  else:
    tagme_file = dataset_file + ".tagme_prediction.maxseq_{}.jsonl".format(args.max_seq_length)
  tagme_data = None
  if os.path.exists(tagme_file):
    tagme_data = {}
    with jsonlines.open(tagme_file, "r") as fi:
      for data in fi:
        #tagme_data.append(data)
        tagme_data[data["id"]] = data["mb"]
    #tagme_data = json.load(open(tagme_file, "r"))

  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #batch = tuple(t.to(args.device) for t in batch)
    example_indices = batch[3]
    # only start gettting mention when it reaches the start index
    if tag_part and example_indices[-1].item() < args.start: continue
    if tag_part and example_indices[0].item() > args.end: break
    _, mbs = get_external_mention_boundary(
          features, 
          example_indices, 
          max_seq_length=args.max_seq_length, 
          language=language, 
          threshold=args.tagme_threshold, 
          tagme_data=tagme_data,
          offset=args.start
        )
    #tagme_mbs.extend(mbs)

    if not tag_part:
      if tagme_data is None or len(tagme_data) <= example_indices[0].item():
        logger.info("Writing TAGME predictions to: {}.".format(tagme_file))
        with jsonlines.open(tagme_file, "a") as fo:
          for data in mbs:
            fo.write({"id":example_indices[0].item(), "mb":data})
    else:
      offset = args.start
      logger.info("Writing TAGME predictions to: {}.".format(tagme_file))
      for i, example_index in enumerate(example_indices):
        exp_id = example_index.item()
        if exp_id < args.start: continue
        if exp_id > args.end: break
        #if tagme_data is None or len(tagme_data) <= exp_id-offset:
        if tagme_data is None or exp_id not in tagme_data:
          with jsonlines.open(tagme_file, "a") as fo:
            fo.write({"id":exp_id, "mb":mbs[i]})


def load_and_cache_examples(args, tokenizer, split='dev', output_examples=False,
              language='en', lang2id=None):
  evaluate = False if split == 'train' else True
  if args.local_rank not in [-1, 0] and not evaluate:
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    torch.distributed.barrier()

  if split == 'train':
    filename = os.path.basename(args.train_file)
  elif split == 'dev':
    filename = os.path.basename(args.valid_file)
  elif split == 'test':
    filename = os.path.basename(args.predict_file)
    filename = filename.replace("<lc>", language)
  else:
    logger.info("Unsupported split: {}.".format(split))
  
  # Load data features from cache or dataset file
  if split == 'test':
    input_dir = os.path.dirname(args.predict_file)
  else:
    input_dir = args.data_dir if args.data_dir else "."
  
  cached_features_file = os.path.join(
    input_dir,
    "cached_{}_{}_{}_{}".format(
      filename,
      args.model_type if evaluate else list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length),
      str(language)
    ),
  )

  # Init features and dataset from cache if it exists
  if os.path.exists(cached_features_file) and not args.overwrite_cache: #and not output_examples:
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    if output_examples:
      processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
      if split == 'test':
        pred_file = args.predict_file.replace("<lc>", language)
        examples = processor.get_dev_examples(args.data_dir, filename=pred_file, language=language)
      elif split == 'dev':
        examples = processor.get_dev_examples(args.data_dir, filename=args.valid_file, language=language)
      else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)
  else:
    logger.info("Creating features from dataset file at %s", input_dir)

    if not args.data_dir and ((split=='dev' and not args.valid_file) or (split=='test' and not args.predict_file) or (split=='train' and not args.train_file)):
      try:
        import tensorflow_datasets as tfds
      except ImportError:
        raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

      if args.version_2_with_negative:
        logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

      tfds_examples = tfds.load("squad")
      examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate, language=language)
    else:
      processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
      if split == 'test':
        pred_file = args.predict_file.replace("<lc>", language)
        examples = processor.get_dev_examples(args.data_dir, filename=pred_file, language=language)
      elif split == 'dev':
        examples = processor.get_dev_examples(args.data_dir, filename=args.valid_file, language=language)
      else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)

    features, dataset = squad_convert_examples_to_features(
      examples=examples,
      tokenizer=tokenizer,
      max_seq_length=args.max_seq_length,
      doc_stride=args.doc_stride,
      max_query_length=args.max_query_length,
      is_training=not evaluate,
      return_dataset="pt",
      threads=args.threads,
      lang2id=lang2id,
      #threshold=args.tagme_threshold,
      #get_external_mention_boundary=(args.get_external_mention_boundary or args.use_external_mention_boundary),
    )

    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save({"features": features, "dataset": dataset}, cached_features_file)

  if args.local_rank == 0 and not evaluate:
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    torch.distributed.barrier()

  if output_examples:
    return dataset, examples, features
  return dataset


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
  )
  parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
  )

  # Other parameters
  parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="The input data dir. Should contain the .json files for the task."
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--train_file",
    default=None,
    type=str,
    help="The input training file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--valid_file",
    default=None,
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    help="The input test file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )

  parser.add_argument(
    "--version_2_with_negative",
    action="store_true",
    help="If true, the SQuAD examples contain some that do not have an answer.",
  )
  parser.add_argument(
    "--null_score_diff_threshold",
    type=float,
    default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.",
  )

  parser.add_argument(
    "--max_seq_length",
    default=384,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
    "longer than this will be truncated, and sequences shorter than this will be padded.",
  )
  parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help="When splitting up a long document into chunks, how much stride to take between chunks.",
  )
  parser.add_argument(
    "--max_query_length",
    default=64,
    type=int,
    help="The maximum number of tokens for the question. Questions longer than this will "
    "be truncated to this length.",
  )
  parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
  parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set.")
  parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
  )
  parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
  )

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
  parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
  )
  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
  )
  parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
  )
  parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
  parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
  )
  parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help="The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another.",
  )
  parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
  )

  parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
  parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
  parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
  )
  parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
  parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
  )
  parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
  )
  parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

  parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
  parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )
  parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
  )
  parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

  parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

  parser.add_argument("--train_lang", type=str, default="en", help="The language of the training data")
  parser.add_argument("--eval_langs", type=str, default="en", help="The language of the test data")
  parser.add_argument("--log_file", type=str, default=None, help="log file")

  parser.add_argument(
    "--eval_test_set", action="store_true", help="Whether to evaluate test set durinng training",
  )
  parser.add_argument(
    "--save_only_best_checkpoint", action="store_true", help="save only the best checkpoint"
  )
  parser.add_argument("--target_task_name", type=str, default="mlqa", help="The target task name")
  parser.add_argument("--freeze_params", type=str, default="", help="prefix to be freezed, split by ',' (e.g. entity,bio).")
  parser.add_argument("--output_entity_info", action="store_true",
            help="Output entity info for debug.")
  parser.add_argument("--get_external_mention_boundary", action="store_true",
            help="Use TAGME to obtain external mention boundary.")
  parser.add_argument("--use_external_mention_boundary", action="store_true",
            help="Use TAGME to predicted external mention boundary.")
  parser.add_argument("--tagme_threshold", default=0.2, type=float, help="Threshold for TAGME.")
  parser.add_argument("--start", type=int, default=-1, help="Beginning example index")
  parser.add_argument("--end", type=int, default=-1, help="End example index")

            
  args = parser.parse_args()

  if args.model_type != "meae":
    args.output_entity_info = False

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd

    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(
    handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S',
    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN
  )


  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
  )

  # Set seed
  set_seed(args)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  preprocess(args, tokenizer, split='dev', language=args.train_lang)


if __name__ == "__main__":
  main()
