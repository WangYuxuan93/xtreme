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
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import enum
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file

from entity_vocab import EntityVocab

from transformers import (
  AdamW,
  get_linear_schedule_with_warmup,
  WEIGHTS_NAME,
  BertConfig,
  BertTokenizer,
  BertForTokenClassification,
  XLMConfig,
  XLMTokenizer,
  XLMRobertaConfig,
  XLMRobertaTokenizer,
  XLMRobertaForTokenClassification,
  MEAEConfig,
  MEAEForTokenClassification,
)
from xlm import XLMForTokenClassification


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
  (tuple(conf.pretrained_config_archive_map.keys())
    for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
  ()
)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
  "meae": (MEAEConfig, MEAEForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# B-LOC, B-ORG, B-PER, I-LOC, I-ORG, I-PER, O
panx2bio = {0:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:0, -100:-100}
def create_bio_targets(label_ids):
  #print ("label_ids:\n", label_ids)
  bio_targets = label_ids.clone()
  batch_size, seq_len = label_ids.size()
  for i in range(batch_size):
    for j in range(seq_len):
      bio_targets[i,j] = panx2bio[int(label_ids[i,j])]
  #print ("bio_targets:\n", bio_targets)
  return bio_targets

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id=None):
  """Train the model."""
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
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                              output_device=args.local_rank,
                              find_unused_parameters=True)

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps * (
          torch.distributed.get_world_size() if args.local_rank != -1 else 1))
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  best_score = 0.0
  best_checkpoint = None
  patience = 0
  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
  set_seed(args) # Add here for reproductibility (even between python 2 and 3)

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
      model.train()
      batch = tuple(t.to(args.device) for t in batch if t is not None)
      inputs = {"input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3]}

      if args.model_type != "distilbert":
        # XLM and RoBERTa don"t use segment_ids
        inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

      if args.model_type == "xlm":
        inputs["langs"] = batch[4]
      
      if args.fine_tune_with_bio:
        inputs["mention_boundaries"] = create_bio_targets(batch[3])

      outputs = model(**inputs)
      loss = outputs[0]

      if args.n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()
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

        scheduler.step()  # Update learning rate schedule
        optimizer.step()
        model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          if args.local_rank == -1 and args.evaluate_during_training:
            # Only evaluate on single GPU otherwise metrics may not average well
            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=args.train_langs, lang2id=lang2id)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.eval_test_set:
            output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
            total = num = 0.0
            with open(output_predict_file, 'a') as writer:
              writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
              for lang in args.predict_langs.split(','):
                result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", lang=lang, lang2id=lang2id)
                writer.write('{}={}\n'.format(lang, result['f1']))
                total += result['f1']
                num += 1
              writer.write('avg={}\n'.format(total / num))

          if args.save_only_best_checkpoint:
            output_dev_file = os.path.join(args.output_dir, 'eval_dev_results')
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, lang2id=lang2id)
            with open(output_dev_file, 'a') as writer:
              writer.write('\n======= Evaluate using the model from checkpoint-{}:\n'.format(global_step))
              writer.write('{}={}\n'.format(args.train_langs, result['f1']))
            if result["f1"] > best_score:
              logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
              best_score = result["f1"]
              # Save the best model checkpoint
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              # Take care of distributed/parallel training
              model_to_save = model.module if hasattr(model, "module") else model
              model_to_save.save_pretrained(output_dir)
              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving the best model checkpoint to %s", output_dir)
              logger.info("Reset patience to 0")
              patience = 0
            else:
              patience += 1
              logger.info("Hit patience={}".format(patience))
              if args.eval_patience > 0 and patience > args.eval_patience:
                logger.info("early stop! patience={}".format(patience))
                epoch_iterator.close()
                train_iterator.close()
                if args.local_rank in [-1, 0]:
                  tb_writer.close()
                return global_step, tr_loss / global_step
          else:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

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


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None, print_result=True):
  eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang, lang2id=lang2id)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  entity_positions = []
  entity_topks = []
  mention_bounds = None
  all_input_ids = None
  model.eval()
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
      inputs = {"input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3]}
      if args.model_type != "distilbert":
        # XLM and RoBERTa don"t use segment_ids
        inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
      if args.model_type == 'xlm':
        inputs["langs"] = batch[4]
      if args.output_entity_info:
        inputs["output_entity_info"] = True
      outputs = model(**inputs)
      tmp_eval_loss, logits = outputs[:2]

      if args.n_gpu > 1:
        # mean() to average on multi-gpu parallel evaluating
        tmp_eval_loss = tmp_eval_loss.mean()

      eval_loss += tmp_eval_loss.item()
    nb_eval_steps += 1
    if preds is None:
      preds = logits.detach().cpu().numpy()
      out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
      preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
      out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
    if args.output_entity_info:
      bio_logits, score, positions = outputs[-3:]

      if positions is not None:
        if args.output_entity_topk > 0:
          probs = torch.nn.functional.softmax(score, dim=1)
          entity_score, entity_idx = torch.topk(probs, args.output_entity_topk, dim=1)
          assert entity_idx.shape[0] == positions.shape[1]
          entity_score = entity_score.detach().cpu().numpy()
          entity_idx = entity_idx.detach().cpu().numpy()

        positions = positions.detach().cpu().numpy()
        offset = 0
        for i in range(bio_logits.size(0)):
          entity_positions.append([])
          if args.output_entity_topk > 0:
            entity_topks.append([])
          while offset < positions.shape[1] and positions[0, offset] <= i:
            if positions[0, offset] == i:
              entity_positions[-1].append((positions[1,offset],positions[2,offset]))
              if args.output_entity_topk > 0:
                entity_topks[-1].append({id:score for id, score in zip(entity_idx[offset], entity_score[offset])})
              offset += 1
      else:
        for i in range(bio_logits.size(0)):
          entity_positions.append([])
          if args.output_entity_topk > 0:
            entity_topks.append([])
      
      if mention_bounds is None:
        mention_bounds = bio_logits.detach().cpu().numpy()
        all_input_ids = inputs["input_ids"].detach().cpu().numpy()
      else:
        mention_bounds = np.append(mention_bounds, bio_logits.detach().cpu().numpy(), axis=0)
        all_input_ids = np.append(all_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
  
  if args.output_entity_info:
    mention_preds = np.argmax(mention_bounds, axis=2)

  if nb_eval_steps == 0:
    results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
  else:
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
          out_label_list[i].append(label_map[out_label_ids[i][j]])
          preds_list[i].append(label_map[preds[i][j]])

    results = {
      "loss": eval_loss,
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list)
    }
  if args.output_entity_info:
    write_entity_info(args, tokenizer, all_input_ids, out_label_ids, label_map, preds, 
                      mention_preds, lang, entity_positions, entity_topks=entity_topks)
  if print_result:
    logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
    for key in sorted(results.keys()):
      logger.info("  %s = %s", key, str(results[key]))

  return results, preds_list


def BIO_to_BIO2(labels):
  bio2_labels = []
  prev_label = None
  for i, label in enumerate(labels):
    new_label = label
    if label.startswith('I'):
      if i == 0:
        new_label = 'B'+label[1:]
      elif i == 1 and labels[0] == '<PAD>':
        new_label = 'B'+label[1:]
      else:
        if prev_label is not None and prev_label == 'O':
          new_label = 'B'+label[1:]
        elif prev_label is not None and prev_label.split('-')[1] != label.split('-')[1]:
          new_label = 'B'+label[1:]
    if new_label != '<PAD>':
      prev_label = new_label
    bio2_labels.append(new_label)
  return bio2_labels

def detokenize(tokens):
  word = []
  for token in tokens:
    if token.startswith("▁"):
      word.append(token[1:])
    elif len(word) == 0:
      word.append(token)
    else:
      word[-1] += token
  return " ".join(word)

def write_entity_info(args, tokenizer, all_input_ids, out_label_ids, label_map, preds, mention_preds, 
                      lang, entity_positions, entity_topks=None):
  output_dir = os.path.join(args.output_dir, "{}_entity_info.txt".format(lang))
  entity_vocab = None
  if entity_topks is not None and os.path.exists(args.entity_vocab_file):
    entity_vocab = EntityVocab(args.entity_vocab_file)
    try:
      import Levenshtein
    except:
      print ("Failed loading Levenshtein!")
  num_gold_ner = 0
  num_bio_corr = 0
  num_b_corr = 0
  num_bio_pred = 0
  num_mention_pred = 0
  num_em_ent_match = 0
  sum_em_levenshtein_ratio = 0
  num_pred_ent_match = 0
  sum_pred_levenshtein_ratio = 0
  assert len(out_label_ids) == len(mention_preds)
  #infile = os.path.join(args.data_dir, lang, "test.{}".format(list(filter(None, args.model_name_or_path.split("/"))).pop()))
  #idxfile = infile + '.idx'
  bound_map = {0: "O", 1: "B", 2: "I", -100: "<PAD>"}
  label_map[-100] = "<PAD>"
  with open(output_dir, "w") as f:
    for i, label_ids in enumerate(out_label_ids):
      #pred_labels = preds[i]
      pred_labels = [label_map[id] for id in preds[i]]
      mention_labels = mention_preds[i]
      gold_labels = [label_map[id] for id in label_ids]
      if args.ner_label_type == "bio":
        gold_labels = BIO_to_BIO2(gold_labels)
        pred_labels = BIO_to_BIO2(pred_labels)
      input_ids = all_input_ids[i]
      input_toks = tokenizer.convert_ids_to_tokens(input_ids)
      ent_pos = entity_positions[i]
      if entity_vocab is not None:
        ent_topk = entity_topks[i]
      ent_info = "Entities: "
      for offset, (ent_s, ent_e) in enumerate(ent_pos):
        #ent = " ".join(input_toks[ent_s:ent_e+1])
        ent = detokenize(input_toks[ent_s:ent_e+1])
        ent_info += "{}-{}:{} | ".format(ent_s,ent_e,ent)
        
        num_mention_pred += 1
        flag = False
        #if label_map[label_ids[ent_s]].startswith("B"):
        if gold_labels[ent_s].startswith("B"):
          flag = True
          for k in range(ent_s+1, ent_e+1):
            #if not label_map[label_ids[ent_s]].startswith("I"):
            if not (gold_labels[ent_s] == "<PAD>" or gold_labels[ent_s].startswith("I")):
              flag = False
        if flag:
          num_bio_corr += 1

        if entity_vocab is not None:
          topk = ent_topk[offset]
          topk_preds = ", ".join(["{} : {}".format(entity_vocab.get_title_by_id(id, lang), score) for id, score in topk.items()])
          ent_info += topk_preds
          # compute Levenshtein ratio for top1 entity pred
          best_id = sorted(topk.items(), key=lambda x:x[1], reverse=True)[0][0]
          ent_pred = entity_vocab.get_title_by_id(best_id, lang)
          ent_info += " @ Best Entity: {}".format(ent_pred)
          lev_rat = Levenshtein.ratio(ent, ent_pred)
          if lev_rat == 1:
            num_pred_ent_match += 1
          sum_pred_levenshtein_ratio += lev_rat
          if flag:
            if lev_rat == 1:
              num_em_ent_match += 1
            sum_em_levenshtein_ratio += lev_rat
  
      f.write(ent_info+"\n")
      for j, tok in enumerate(input_toks):
        if tok == "<pad>": break
        #items = [tok, label_map[label_ids[j]], label_map[pred_labels[j]], bound_map[mention_labels[j]]]
        items = [tok, gold_labels[j], pred_labels[j], bound_map[mention_labels[j]]]
        if label_ids[j] >= 0:
          if mention_labels[j] == 1:
            num_bio_pred += 1
          #gold_l = label_map[label_ids[j]]
          gold_l = gold_labels[j]
          if gold_l.startswith("B"):
            num_gold_ner += 1
            if mention_labels[j] == 1:
              num_b_corr += 1
        f.write("\t".join(items)+"\n")
      f.write("\n")
  
    #print ("Gold NER: {}, Bio Total Pred: {}, Bio Pred Corr:{}".format(num_gold_ner, num_bio_pred, num_bio_corr))
    print ("Gold NER: {}, Bio Total Pred: {}, Mention Pred: {}, Bio Pred Corr:{}".format(num_gold_ner, num_bio_pred, num_mention_pred, num_bio_corr))
    p = float(num_bio_corr) / num_mention_pred
    r = float(num_bio_corr) / num_gold_ner
    b_p = float(num_b_corr) / num_mention_pred
    b_r = float(num_b_corr) / num_gold_ner
    print ("Exact Match | Precision: {}, Recall: {}".format(p, r))
    print ("B Match | Precision: {}, Recall: {}".format(b_p, b_r))
    f.write("\n########BIO Head Prediction########\nGold NER: {}, Bio Total Pred: {}, Mention Pred: {}, BIO Pred Corr:{}, B Pred Corr:{}".format(num_gold_ner, num_bio_pred, num_mention_pred, num_bio_corr, num_b_corr))
    f.write("\nExact Match | Precision: {}, Recall: {}\n".format(p, r))
    f.write("\nB Match | Precision: {}, Recall: {}".format(b_p, b_r))
    
    pred_ent_match_acc = num_pred_ent_match / float(num_mention_pred)
    avg_pred_levenshtein_ratio = sum_pred_levenshtein_ratio / float(num_mention_pred)
    em_ent_match_acc = num_em_ent_match / float(num_bio_corr)
    avg_em_levenshtein_ratio = sum_em_levenshtein_ratio / float(num_bio_corr)
    print ("\nEntity Prediction Corr for all Pred Mention: {}, Entity Prediction Corr for Exact Match Mention: {}".format(
            num_pred_ent_match, num_em_ent_match))
    print ("\nEntity Prediction Accuracy | For EM Mention: {}, For all Predicted Mention: {}".format(
            em_ent_match_acc, pred_ent_match_acc))
    print ("\nAverage Levenshtein ratio | For EM Mention: {}, For all Predicted Mention: {}".format(
            avg_em_levenshtein_ratio, avg_pred_levenshtein_ratio))
    f.write("\nEntity Prediction Corr for all Pred Mention: {}, Entity Prediction Corr for Exact Match Mention: {}".format(
            num_pred_ent_match, num_em_ent_match))
    f.write("\nEntity Prediction Accuracy | For EM Mention: {}, For all Predicted Mention: {}".format(
            em_ent_match_acc, pred_ent_match_acc))
    f.write("\nAverage Levenshtein ratio | For EM Mention: {}, For all Predicted Mention: {}".format(
            avg_em_levenshtein_ratio, avg_pred_levenshtein_ratio))

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
    list(filter(None, args.model_name_or_path.split("/"))).pop(),
    str(args.max_seq_length)))
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    langs = lang.split(',')
    logger.info("all languages = {}".format(lang))
    features = []
    for lg in langs:
      data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, list(filter(None, args.model_name_or_path.split("/"))).pop()))
      logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
      examples = read_examples_from_file(data_file, lg, lang2id)
      features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                          cls_token_at_end=bool(args.model_type in ["xlnet"]),
                          cls_token=tokenizer.cls_token,
                          cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                          sep_token=tokenizer.sep_token,
                          sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                          pad_on_left=bool(args.model_type in ["xlnet"]),
                          pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                          pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                          pad_token_label_id=pad_token_label_id,
                          lang=lg
                          )
      features.extend(features_lg)
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  if few_shot > 0 and mode == 'train':
    logger.info("Original no. of examples = {}".format(len(features)))
    features = features[: few_shot]
    logger.info('Using few-shot learning on {} examples'.format(len(features)))

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
  if args.model_type == 'xlm' and features[0].langs is not None:
    all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
    logger.info('all_langs[0] = {}'.format(all_langs[0]))
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
  else:
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  return dataset


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
            help="The input data dir. Should contain the training files for the NER/POS task.")
  parser.add_argument("--model_type", default=None, type=str, required=True,
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
  parser.add_argument("--output_dir", default=None, type=str, required=True,
            help="The output directory where the model predictions and checkpoints will be written.")

  ## Other parameters
  parser.add_argument("--labels", default="", type=str,
            help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
  parser.add_argument("--config_name", default="", type=str,
            help="Pretrained config name or path if not the same as model_name")
  parser.add_argument("--tokenizer_name", default="", type=str,
            help="Pretrained tokenizer name or path if not the same as model_name")
  parser.add_argument("--cache_dir", default=None, type=str,
            help="Where do you want to store the pre-trained models downloaded from s3")
  parser.add_argument("--max_seq_length", default=128, type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
               "than this will be truncated, sequences shorter will be padded.")
  parser.add_argument("--do_train", action="store_true",
            help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true",
            help="Whether to run eval on the dev set.")
  parser.add_argument("--do_predict", action="store_true",
            help="Whether to run predictions on the test set.")
  parser.add_argument("--do_predict_dev", action="store_true",
            help="Whether to run predictions on the dev set.")
  parser.add_argument("--init_checkpoint", default=None, type=str,
            help="initial checkpoint for train/predict")
  parser.add_argument("--evaluate_during_training", action="store_true",
            help="Whether to run evaluation during training at each logging step.")
  parser.add_argument("--do_lower_case", action="store_true",
            help="Set this flag if you are using an uncased model.")
  parser.add_argument("--few_shot", default=-1, type=int,
            help="num of few-shot exampes")

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
            help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
            help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument("--learning_rate", default=5e-5, type=float,
            help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float,
            help="Weight decay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
            help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float,
            help="Max gradient norm.")
  parser.add_argument("--num_train_epochs", default=3.0, type=float,
            help="Total number of training epochs to perform.")
  parser.add_argument("--max_steps", default=-1, type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
  parser.add_argument("--warmup_steps", default=0, type=int,
            help="Linear warmup over warmup_steps.")

  parser.add_argument("--logging_steps", type=int, default=50,
            help="Log every X updates steps.")
  parser.add_argument("--save_steps", type=int, default=50,
            help="Save checkpoint every X updates steps.")
  parser.add_argument("--save_only_best_checkpoint", action="store_true",
            help="Save only the best checkpoint during training")
  parser.add_argument("--eval_all_checkpoints", action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
  parser.add_argument("--no_cuda", action="store_true",
            help="Avoid using CUDA when available")
  parser.add_argument("--overwrite_output_dir", action="store_true",
            help="Overwrite the content of the output directory")
  parser.add_argument("--overwrite_cache", action="store_true",
            help="Overwrite the cached training and evaluation sets")
  parser.add_argument("--seed", type=int, default=42,
            help="random seed for initialization")

  parser.add_argument("--fp16", action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
  parser.add_argument("--fp16_opt_level", type=str, default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
               "See details at https://nvidia.github.io/apex/amp.html")
  parser.add_argument("--local_rank", type=int, default=-1,
            help="For distributed training: local_rank")
  parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
  parser.add_argument("--predict_langs", type=str, default="en", help="prediction languages")
  parser.add_argument("--train_langs", default="en", type=str,
            help="The languages in the training sets.")
  parser.add_argument("--log_file", type=str, default=None, help="log file")
  parser.add_argument("--eval_patience", type=int, default=-1, help="wait N times of decreasing dev score before early stop during training")
  
  parser.add_argument("--output_entity_info", action="store_true",
            help="Output entity info for debug.")
  parser.add_argument("--fine_tune_with_bio", action="store_true",
            help="Fine-tuning target task along with Bio classification.")
  parser.add_argument(
    "--eval_test_set",
    action="store_true",
    help="Whether to evaluate test set durinng training",
  )
  parser.add_argument("--output_entity_topk", type=int, default=0,
            help="Output topk entity prediction for each detected mention.")
  parser.add_argument("--entity_vocab_file", type=str, default=None, help="entity vocab file")
  parser.add_argument("--ner_label_type", type=str, default="bio2", choices=["bio","bio2"], help="log file")
  args = parser.parse_args()

  if args.model_type != "meae":
    args.output_entity_info = False
    args.output_entity_topk = 0
  if not args.output_entity_info:
    args.output_entity_topk = 0

  if os.path.exists(args.output_dir) and os.listdir(
      args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir))

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:
  # Initializes the distributed backend which sychronizes nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
                      format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt = '%m/%d/%Y %H:%M:%S',
                      level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logging.info("Input args: %r" % args)
  logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
           args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

  # Set seed
  set_seed(args)

  # Prepare NER/POS task
  labels = get_labels(args.labels)
  num_labels = len(labels)
  # Use cross entropy ignore index as padding label id
  # so that only real label ids contribute to the loss later
  pad_token_label_id = CrossEntropyLoss().ignore_index

  # Load pretrained model and tokenizer
  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                      num_labels=num_labels,
                      cache_dir=args.cache_dir if args.cache_dir else None)
  tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                        do_lower_case=args.do_lower_case,
                        cache_dir=args.cache_dir if args.cache_dir else None)

  if args.init_checkpoint:
    logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
    model = model_class.from_pretrained(args.init_checkpoint,
                                        config=config,
                                        cache_dir=args.init_checkpoint)
  else:
    logger.info("loading from cached model = {}".format(args.model_name_or_path))
    model = model_class.from_pretrained(args.model_name_or_path,
                      from_tf=bool(".ckpt" in args.model_name_or_path),
                      config=config,
                      cache_dir=args.cache_dir if args.cache_dir else None)
  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("Using lang2id = {}".format(lang2id))

  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank == 0:
    torch.distributed.barrier()
  model.to(args.device)
  logger.info("Training/evaluation parameters %s", args)

  # Training
  if args.do_train:
    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Saving best-practices: if you use default names for the model,
  # you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    # Save model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    logger.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

  # Initialization for evaluation
  results = {}
  if args.init_checkpoint:
    best_checkpoint = args.init_checkpoint
  elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
    best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
  else:
    best_checkpoint = args.output_dir
  best_f1 = 0

  # Evaluation
  if args.do_eval and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
      checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
      logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      model = model_class.from_pretrained(checkpoint)
      model.to(args.device)
      result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, lang2id=lang2id)
      if result["f1"] > best_f1:
        best_checkpoint = checkpoint
        best_f1 = result["f1"]
      if global_step:
        result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
      results.update(result)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
      for key in sorted(results.keys()):
        writer.write("{} = {}\n".format(key, str(results[key])))
      writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

  # Prediction
  if args.do_predict and args.local_rank in [-1, 0]:
    logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(best_checkpoint)
    model.to(args.device)

    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_results_file, "a") as result_writer:
      for lang in args.predict_langs.split(','):
        if not os.path.exists(os.path.join(args.data_dir, lang, 'test.{}'.format(list(filter(None, args.model_name_or_path.split("/"))).pop()))):
          logger.info("Language {} does not exist".format(lang))
          continue
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", lang=lang, lang2id=lang2id)

        # Save results
        result_writer.write("=====================\nlanguage={}\n".format(lang))
        for key in sorted(result.keys()):
          result_writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
        infile = os.path.join(args.data_dir, lang, "test.{}".format(list(filter(None, args.model_name_or_path.split("/"))).pop()))
        idxfile = infile + '.idx'
        save_predictions(args, predictions, output_test_predictions_file, infile, idxfile, output_word_prediction=True)

  # Predict dev set
  if args.do_predict_dev and args.local_rank in [-1, 0]:
    logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(best_checkpoint)
    model.to(args.device)

    output_test_results_file = os.path.join(args.output_dir, "dev_results.txt")
    with open(output_test_results_file, "w") as result_writer:
      for lang in args.predict_langs.split(','):
        if not os.path.exists(os.path.join(args.data_dir, lang, 'dev.{}'.format(list(filter(None, args.model_name_or_path.split("/"))).pop()))):
          logger.info("Language {} does not exist".format(lang))
          continue
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=lang, lang2id=lang2id)

        # Save results
        result_writer.write("=====================\nlanguage={}\n".format(lang))
        for key in sorted(result.keys()):
          result_writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "dev_{}_predictions.txt".format(lang))
        infile = os.path.join(args.data_dir, lang, "dev.{}".format(list(filter(None, args.model_name_or_path.split("/"))).pop()))
        idxfile = infile + '.idx'
        save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)

def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
  # Save predictions
  with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
    text = text_reader.readlines()
    index = idx_reader.readlines()
    assert len(text) == len(index)

  # Sanity check on the predictions
  with open(output_file, "w") as writer:
    example_id = 0
    prev_id = int(index[0])
    for line, idx in zip(text, index):
      if line == "" or line == "\n":
        example_id += 1
      else:
        cur_id = int(idx)
        output_line = '\n' if cur_id != prev_id else ''
        if output_word_prediction:
          output_line += line.split()[0] + '\t'
        output_line += predictions[example_id].pop(0) + '\n'
        writer.write(output_line)
        prev_id = cur_id

if __name__ == "__main__":
  main()
