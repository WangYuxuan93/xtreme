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
from evaluate_mlqa import evaluate as squad_eval_metric

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

from transformers.data.metrics.squad_metrics import (
  compute_predictions_log_probs,
  compute_predictions_logits,
  #squad_evaluate,
)

from xlm_roberta import XLMRobertaForQuestionAnswering, XLMRobertaConfig

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

ALL_MODELS = sum(
  (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
  (),
)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
  "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
  "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
  "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
  "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
  "xlm-roberta": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
  "meae": (MEAEConfig, MEAEForQuestionAnswering, XLMRobertaTokenizer)
}


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
  return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
  """ Train the model """
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Check if saved optimizer or scheduler states exist
  if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    os.path.join(args.model_name_or_path, "scheduler.pt")
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info(
    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  )
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 1
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_name_or_path):
    try:
      # set global_step to gobal_step of last saved checkpoint from model path
      checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
      global_step = int(checkpoint_suffix)
      epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
      steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

      logger.info("  Continuing training from checkpoint, will skip to saved global_step")
      logger.info("  Continuing training from epoch %d", epochs_trained)
      logger.info("  Continuing training from global step %d", global_step)
      logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    except ValueError:
      logger.info("  Starting fine-tuning.")

  best_score = 0
  best_checkpoint = None
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  # Added here for reproductibility
  set_seed(args)

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)

      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None,
        "start_positions": batch[3],
        "end_positions": batch[4],
      }

      if args.use_external_mention_boundary:
        inputs["mention_boundaries"] = batch[8]
        inputs["use_external_mention_boundary"] = True

      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if args.version_2_with_negative:
          inputs.update({"is_impossible": batch[7]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[7]
      outputs = model(**inputs)
      # model outputs are always tuple in transformers (see doc)
      loss = outputs[0]

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        # Log metrics
        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Only evaluate when single GPU otherwise metrics may not average well
          if args.local_rank == -1 and args.evaluate_during_training:
            #output_dev_file = os.path.join(args.output_dir, 'eval_dev_results')

            results = evaluate(args, model, tokenizer)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            
            #with open(output_dev_file, 'a') as writer:
            #  writer.write('\n======= Evaluate using the model from checkpoint-{}:\n'.format(global_step))
            #  writer.write('{}={}\n'.format(args.eval_langs, results['f1']))

          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

        # Save model checkpoint
        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.eval_test_set:
            output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
            total_f1 = total_em = num = 0.0
            with open(output_predict_file, 'a') as writer:
              writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
              for language in args.eval_langs.split(','):
                result = evaluate(args, model, tokenizer, split='test', prefix="", language=language)
                writer.write('{}: f1={}, em={}\n'.format(language, result['f1'], result['exact_match']))
                logger.info('{}: f1={}, em={}'.format(language, result['f1'], result['exact_match']))
                total_f1 += result['f1']
                total_em += result['exact_match']
                num += 1
              writer.write('Avg f1={}, em={}\n'.format(total_f1 / num, total_em / num))

          if args.save_only_best_checkpoint:
            output_dev_file = os.path.join(args.output_dir, 'eval_dev_results')
            result = evaluate(args, model, tokenizer)
            with open(output_dev_file, 'a') as writer:
              writer.write('\n======= Evaluate using the model from checkpoint-{}:\n'.format(global_step))
              writer.write('{}: f1={}, em={}\n'.format(args.train_lang, result['f1'], result['exact_match']))
            logger.info(" Dev {} f1 = {}, em = {}".format(args.train_lang, result['f1'], result['exact_match']))
            if result['f1'] > best_score:
              logger.info(" result['f1']={} > best_score={}".format(result['f1'], best_score))
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              best_score = result['f1']
              # Save model checkpoint
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              model_to_save = (
                model.module if hasattr(model, "module") else model
              )  # Take care of distributed/parallel training
              model_to_save.save_pretrained(output_dir)
              tokenizer.save_pretrained(output_dir)

              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving model checkpoint to %s", output_dir)

              torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
              torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
              logger.info("Saving optimizer and scheduler states to %s", output_dir)
          else:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break
  
  if args.save_only_best_checkpoint:
    output_dev_file = os.path.join(args.output_dir, 'eval_dev_results')
    with open(output_dev_file, 'a') as writer:
      writer.write("Global_step = {}, average loss = {}\n".format(global_step, tr_loss / global_step))
      writer.write("Best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))
  
  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step


def get_external_mention_boundary(features, example_indices, max_seq_length=384, 
                                  language="en", threshold=0.2, tagme_data=None): 
  mention_boundaries = []
  for i, example_index in enumerate(example_indices):
    if tagme_data is not None and len(tagme_data) > example_index.item():
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

def evaluate(args, model, tokenizer, split='dev', prefix="", language='en', lang2id=None):
  dataset, examples, features = load_and_cache_examples(args, tokenizer, split, output_examples=True,
                              language=language, lang2id=lang2id)

  if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(args.output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation {} *****".format(prefix))
  logger.info("  Num examples = %d", len(dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)

  all_results = []
  start_time = timeit.default_timer()
  entity_positions = []
  mention_bounds = None
  all_input_ids = None
  
  if args.get_external_mention_boundary or args.use_external_mention_boundary: 
    tagme_mbs = []
    dataset_file = args.valid_file if split=='dev' else args.predict_file.replace("<lc>", language)
    tagme_file = dataset_file + ".tagme_prediction.maxseq_{}.jsonl".format(args.max_seq_length)
    tagme_data = None
    if os.path.exists(tagme_file):
      tagme_data = []
      with jsonlines.open(tagme_file, "r") as fi:
        for data in fi:
          tagme_data.append(data)
      #tagme_data = json.load(open(tagme_file, "r"))
  else:
    tagme_mbs = None

  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None,
      }
      example_indices = batch[3]
      if args.output_entity_info:
        inputs["output_entity_info"] = True
      if (args.get_external_mention_boundary or args.use_external_mention_boundary) and language in ["en", "de", "it"]:
        """
        mention_boundaries, mbs = get_external_mention_boundary(
              features, 
              example_indices, 
              max_seq_length=args.max_seq_length, 
              language=language, 
              threshold=args.tagme_threshold, 
              tagme_data=tagme_data
            )
        tagme_mbs.extend(mbs)
        """
        #print ("input_ids:{}, mb:{}".format(batch[0].shape, mention_boundaries.shape))
        #exit()
        if args.use_external_mention_boundary:
          #inputs["mention_boundaries"] = mention_boundaries.to(args.device)
          inputs["mention_boundaries"] = batch[7]
          inputs["use_external_mention_boundary"] = True
        
        if (args.get_external_mention_boundary or args.use_external_mention_boundary) and language in ["en", "de", "it"]:
          #tagme_file = os.path.join(pred_dir, "tagme_prediction_{}.json".format(language))
          #if not os.path.exists(tagme_file):
          if tagme_data is None or len(tagme_data) <= example_indices[0].item():
            logger.info("Writing TAGME predictions to: {}.".format(tagme_file))
            with jsonlines.open(tagme_file, "a") as fo:
              for data in mbs:
                fo.write(data)

      # XLNet and XLM use more arguments for their predictions
      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[6]

      outputs = model(**inputs)

    if args.output_entity_info:
      bio_logits, score, positions = outputs[-3:]

      if positions is not None:
        positions = positions.detach().cpu().numpy()
        offset = 0
        for i in range(bio_logits.size(0)):
          entity_positions.append([])
          while offset < positions.shape[1] and positions[0, offset] <= i:
            if positions[0, offset] == i:
              entity_positions[-1].append((positions[1,offset],positions[2,offset]))
              offset += 1
      else:
        for i in range(bio_logits.size(0)):
            entity_positions.append([])
      
      if mention_bounds is None:
        mention_bounds = bio_logits.detach().cpu().numpy()
        all_input_ids = inputs["input_ids"].detach().cpu().numpy()
      else:
        mention_bounds = np.append(mention_bounds, bio_logits.detach().cpu().numpy(), axis=0)
        all_input_ids = np.append(all_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
    outputs = outputs[:-3]
  
    for i, example_index in enumerate(example_indices):
      eval_feature = features[example_index.item()]
      unique_id = int(eval_feature.unique_id)

      output = [to_list(output[i]) for output in outputs]

      # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
      # models only use two.
      if len(output) >= 5:
        start_logits = output[0]
        start_top_index = output[1]
        end_logits = output[2]
        end_top_index = output[3]
        cls_logits = output[4]

        result = SquadResult(
          unique_id,
          start_logits,
          end_logits,
          start_top_index=start_top_index,
          end_top_index=end_top_index,
          cls_logits=cls_logits,
        )

      else:
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)

      all_results.append(result)

  if args.output_entity_info:
    mention_preds = np.argmax(mention_bounds, axis=2)
  
  evalTime = timeit.default_timer() - start_time
  logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

  # Compute predictions
  if split == 'dev':
    pred_dir = os.path.join(args.output_dir, "predictions/squad")
  else:
    pred_dir = os.path.join(args.output_dir, "predictions/{}".format(args.target_task_name))
  if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
  output_prediction_file = os.path.join(pred_dir, "predictions_{}_{}.json".format(language, prefix))
  output_nbest_file = os.path.join(pred_dir, "nbest_predictions_{}_{}.json".format(language, prefix))

  if args.version_2_with_negative:
    output_null_log_odds_file = os.path.join(pred_dir, "null_odds_{}.json".format(prefix))
  else:
    output_null_log_odds_file = None

  #if (args.get_external_mention_boundary or args.use_external_mention_boundary) and language in ["en", "de", "it"]:
    #tagme_file = os.path.join(pred_dir, "tagme_prediction_{}.json".format(language))
  #  if not os.path.exists(tagme_file):
  #    logger.info("Writing TAGME predictions to: {}.".format(tagme_file))
  #    with open(tagme_file, "w") as fo:
  #      json.dump(tagme_mbs, fo)

  if args.output_entity_info:
    write_entity_info(args, tokenizer, all_input_ids, mention_preds, language, entity_positions, pred_dir,
                      tagme_mbs=tagme_mbs)


  # XLNet and XLM use a more complex post-processing procedure
  if args.model_type in ["xlnet", "xlm"]:
    start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
    end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    predictions = compute_predictions_log_probs(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      start_n_top,
      end_n_top,
      args.version_2_with_negative,
      tokenizer,
      args.verbose_logging,
    )
  else:
    predictions = compute_predictions_logits(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      args.do_lower_case,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      args.verbose_logging,
      args.version_2_with_negative,
      args.null_score_diff_threshold,
      tokenizer,
      is_zh=True if language == 'zh' else False,
    )

  # Compute the F1 and exact scores.
  #results = squad_evaluate(examples, predictions)
  dataset_file = args.valid_file if split=='dev' else args.predict_file.replace("<lc>", language)
  results = eval_squad(dataset_file, predictions, language=language)
  #results = eval_squad(dataset_file, output_prediction_file)

  return results


def eval_squad(dataset_file, predictions, language):
  with open(dataset_file) as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
  #with open(prediction_file) as prediction_file:
  #  predictions = json.load(prediction_file)
  return squad_eval_metric(dataset, predictions, language)


def write_entity_info(args, tokenizer, all_input_ids, mention_preds, lang, entity_positions, pred_dir,
                      tagme_mbs=None):
  if tagme_mbs is None:
    output_dir = os.path.join(pred_dir, "{}_entity_info.txt".format(lang))
    num_bio_pred = 0
    num_mention_pred = 0
    bound_map = {0: "O", 1: "B", 2: "I", -100: "<PAD>"}
    with open(output_dir, "w") as f:
      for i, input_ids in enumerate(all_input_ids):
        mention_labels = mention_preds[i]
        input_toks = tokenizer.convert_ids_to_tokens(input_ids)
        ent_pos = entity_positions[i]
        ent_info = "Entities: "
        for ent_s, ent_e in ent_pos:
          ent = " ".join(input_toks[ent_s:ent_e+1])
          ent_info += "{}-{}:{} | ".format(ent_s,ent_e,ent)
          num_mention_pred += 1
        f.write(ent_info+"\n")
        for j, tok in enumerate(input_toks):
          if tok == "<pad>": break
          items = [tok, bound_map[mention_labels[j]]]
          if mention_labels[j] == 1:
            num_bio_pred += 1
          f.write("\t".join(items)+"\n")
        f.write("\n")
      print ("B Total Pred: {}, Mention Total Pred:{}".format(num_bio_pred, num_mention_pred))
      f.write("\n########BIO Head Prediction########\nB Total Pred: {}, Mention Total Pred:{}\n".format(num_bio_pred, num_mention_pred))
  else:
    output_dir = os.path.join(pred_dir, "{}_entity_info.txt".format(lang))
    num_gold_ner = 0
    num_bio_corr = 0
    num_bio_pred = 0
    num_mention_pred = 0
    bound_map = {0: "O", 1: "B", 2: "I", -100: "<PAD>"}
    with open(output_dir, "w") as f:
      for i, input_ids in enumerate(all_input_ids):
        tagme_labels = tagme_mbs[i]
        mention_labels = mention_preds[i]
        input_toks = tokenizer.convert_ids_to_tokens(input_ids)
        ent_pos = entity_positions[i]
        ent_info = "Entities: "
        for ent_s, ent_e in ent_pos:
          ent = " ".join(input_toks[ent_s:ent_e+1])
          ent_info += "{}-{}:{} | ".format(ent_s,ent_e,ent)
          num_mention_pred += 1
          flag = False
          if tagme_labels[ent_s] == 1:
            flag = True
            for k in range(ent_s+1, ent_e+1):
              if tagme_labels[k] != 2:
                flag = False
          if flag:
            num_bio_corr += 1
        f.write(ent_info+"\n")
        for j, tok in enumerate(input_toks):
          if tok == "<pad>": break
          items = [tok, bound_map[tagme_labels[j]], bound_map[mention_labels[j]]]
          if mention_labels[j] == 1:
            num_bio_pred += 1
          if tagme_labels[j] == 1:
            num_gold_ner += 1
          f.write("\t".join(items)+"\n")
        f.write("\n")

      #print ("B Total Pred: {}, Mention Total Pred:{}".format(num_bio_pred, num_mention_pred))
      #f.write("\n########BIO Head Prediction########\nB Total Pred: {}, Mention Total Pred:{}\n".format(num_bio_pred, num_mention_pred))
      print ("TAGME Mentions: {}, Bio Total Pred: {}, Mention Pred: {}, Bio Pred Corr:{}".format(num_gold_ner, num_bio_pred, num_mention_pred, num_bio_corr))
      p = float(num_bio_corr) / num_mention_pred
      r = float(num_bio_corr) / num_gold_ner
      print ("Precision: {}, Recall: {}".format(p, r))
      f.write("\n########BIO Head Prediction########\nTAGME Mentions: {}, Bio Total Pred: {}, Bio Pred Corr:{}\n".format(num_gold_ner, num_bio_pred, num_bio_corr))
      f.write("Precision: {}, Recall: {}".format(p, r))


def load_and_cache_examples(args, tokenizer, split='train', output_examples=False,
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
      threshold=args.threshold,
      get_external_mention_boundary=(args.get_external_mention_boundary or args.use_external_mention_boundary),
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
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
  )
  parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints and predictions will be written.",
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

            
  args = parser.parse_args()

  if args.model_type != "meae":
    args.output_entity_info = False

  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

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
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  # Set using of language embedding to True
  if args.model_type == "xlm":
    config.use_lang_emb = True
  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  model = model_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  if len(args.freeze_params) > 0:
    freeze(model, prefix=args.freeze_params.split(","))

  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("lang2id = {}".format(lang2id))

  if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)

  # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
  # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
  # remove the need for this code, but it is still valid.
  if args.fp16:
    try:
      import apex

      apex.amp.register_half_function(torch, "einsum")
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

  # Training
  if args.do_train:
    train_dataset = load_and_cache_examples(args, tokenizer, split='train', output_examples=False, language=args.train_lang, lang2id=lang2id)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Save the trained model and the tokenizer
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir, force_download=True)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)

  results = {}
  if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
    best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
  else:
    best_checkpoint = args.output_dir
  best_score = 0
  # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
  if args.do_eval and args.local_rank in [-1, 0]:
    logger.info("Loading checkpoints saved during training for evaluation")
    checkpoints = [args.output_dir]
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')) and best_checkpoint != args.output_dir:
      checkpoints.append(best_checkpoint)
    if args.eval_all_checkpoints:
      checkpoints = list(
        os.path.dirname(c)
        for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
      )
      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
      # Reload the model
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
      model = model_class.from_pretrained(checkpoint, force_download=True)
      model.to(args.device)

      # Evaluate
      result = evaluate(args, model, tokenizer, split='dev', prefix=prefix, language=args.train_lang)
      if result['f1'] > best_score:
        best_checkpoint = checkpoint
        best_score = result['f1']
      result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
      results.update(result)

    output_eval_file = os.path.join(args.output_dir, 'eval_results')
    with open(output_eval_file, 'w') as writer:
      for key, value in results.items():
        writer.write('{} = {}\n'.format(key, value))
      writer.write("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
      logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
    #logger.info("Results: {}".format(results))
  
  # Prediction
  if args.do_predict and args.local_rank in [-1, 0]:
    results = {}
    """
    if args.do_train:
      logger.info("Loading checkpoints saved during training for evaluation")
      checkpoints = [args.output_dir]
      if args.eval_all_checkpoints:
        checkpoints = list(
          os.path.dirname(c)
          for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    else:
      logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
      checkpoints = [args.model_name_or_path]
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    """
    # Reload the model
    model = model_class.from_pretrained(best_checkpoint, force_download=True)
    model.to(args.device)
    output_predict_file = os.path.join(args.output_dir, 'test_results.txt')
    total_f1 = total_em = num = 0.0
    with open(output_predict_file, 'a') as writer:
      writer.write('======= Predict using the model from {} for test:\n'.format(best_checkpoint))
      for language in args.eval_langs.split(','):
        result = evaluate(args, model, tokenizer, split='test', prefix="", language=language)
        writer.write('{}: f1={}, em={}\n'.format(language, result['f1'], result['exact_match']))
        logger.info('{}: f1={}, em={}'.format(language, result['f1'], result['exact_match']))
        total_f1 += result['f1']
        total_em += result['exact_match']
        num += 1
      writer.write('Avg f1={}, em={}\n'.format(total_f1 / num, total_em / num))

  return results


def freeze(model, prefix=["entity_embedding"]):
  for name, param in model.named_parameters():
    for s in prefix:
      if s in name:
        param.requires_grad = False
  
  update_params = []
  fixed_params = []
  for name, param in model.named_parameters():
    if param.requires_grad:
      update_params.append(name)
    else:
      fixed_params.append(name)
  logger.info("Updating keys:\n{}".format(update_params))
  logger.info("Fixed keys:\n{}".format(fixed_params))

if __name__ == "__main__":
  main()
