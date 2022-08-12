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
"""Utility functions for NER/POS tagging tasks."""

from __future__ import absolute_import, division, print_function

import itertools
import math
import warnings
from typing import Dict, List, Tuple

import logging
import os
from io import open
from transformers import XLMTokenizer

import numpy as np
from seqeval.scheme import IOB1, IOB2, Entities, Entity

NON_ENTITY = "O"

logger = logging.getLogger(__name__)


class InputExample(object):
  """A single training/test example for token classification."""

  def __init__(self, guid, words, labels, langs=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      words: list. The words of the sequence.
      labels: (Optional) list. The labels for each word of the sequence. This should be
      specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.words = words
    self.labels = labels
    self.langs = langs


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_ids, langs=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.langs = langs

def read_examples_from_file(file_path, lang, lang2id=None):
  if not os.path.exists(file_path):
    logger.info("[Warming] file {} not exists".format(file_path))
    return []
  guid_index = 1
  examples = []
  subword_len_counter = 0
  if lang2id:
    lang_id = lang2id.get(lang, lang2id['en'])
  else:
    lang_id = 0
  logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
  with open(file_path, encoding="utf-8") as f:
    words = []
    labels = []
    langs = []
    sentence_boundaries = [0]
    for line in f:
      line = line.rstrip()
      # split by document, sentence_boundaries denotes each sentence
      if line.startswith("-DOCSTART"):
        if words:
          assert sentence_boundaries[0] == 0
          assert sentence_boundaries[-1] == len(words)
          examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                         words=words,
                         labels=labels,
                         langs=langs,
                         sentence_boundaries=sentence_boundaries))
          guid_index += 1
          words = []
          labels = []
          sentence_boundaries = [0]
        continue
      if not line:
        if len(words) != sentence_boundaries[-1]:
          sentence_boundaries.append(len(words))
      else:
        splits = line.split("\t")
        words.append(splits[0])
        langs.append(lang_id)
        if len(splits) > 1:
          labels.append(splits[-1].replace("\n", ""))
        else:
          # Examples could have no label for mode = "test"
          labels.append("O")
    if words:
      examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                     words=words,
                     labels=labels,
                     langs=langs,
                     sentence_boundaries=sentence_boundaries))
  return examples

def convert_examples_to_features(examples,
                 label_list,
                 max_seq_length,
                 tokenizer,
                 cls_token_at_end=False,
                 cls_token="[CLS]",
                 cls_token_segment_id=1,
                 sep_token="[SEP]",
                 sep_token_extra=False,
                 pad_on_left=False,
                 pad_token=0,
                 pad_token_segment_id=0,
                 pad_token_label_id=-1,
                 sequence_a_segment_id=0,
                 mask_padding_with_zero=True,
                 lang='en',
                 max_mention_length=16,
                 max_entity_length=128,
                 entity_id=1):
  """ Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
  """

  label_map = {label: i for i, label in enumerate(label_list)}
  max_num_subwords = max_seq_length - 2

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d of %d", ex_index, len(examples))

    if isinstance(tokenizer, XLMTokenizer):
      toks = [tokenizer.tokenize(w, lang=lang) for w in example.words]
    else:
      toks = [tokenizer.tokenize(w) for w in example.words]
    subwords = [sw for token in toks for sw in token]

    subword2token = list(itertools.chain(*[[i] * len(token) for i, token in enumerate(toks)]))
    token2subword = [0] + list(itertools.accumulate(len(token) for token in toks))
    subword_start_positions = frozenset(token2subword)
    subword_sentence_boundaries = [sum(len(token) for token in toks[:p]) for p in example.sentence_boundaries]
    
    # extract entities from IOB tags
    # we need to pass sentence by sentence
    entities = []
    for s, e in zip(example.sentence_boundaries[:-1], example.sentence_boundaries[1:]):
      for ent in Entities([example.labels[s:e]], scheme=IOB2).entities[0]:
        ent.start += s
        ent.end += s
        entities.append(ent)

    span_to_entity_label = dict()
    for ent in entities:
      subword_start = token2subword[ent.start]
      subword_end = token2subword[ent.end]
      span_to_entity_label[(subword_start, subword_end)] = ent.tag
    
    # split data according to sentence boundaries
    for n in range(len(subword_sentence_boundaries) - 1):
      # process (sub) words
      doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]
      assert doc_sent_end - doc_sent_start < max_num_subwords

      left_length = doc_sent_start
      right_length = len(subwords) - doc_sent_end
      sentence_length = doc_sent_end - doc_sent_start
      half_context_length = int((max_num_subwords - sentence_length) / 2)

      if left_length < right_length:
        left_context_length = min(left_length, half_context_length)
        right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
      else:
        right_context_length = min(right_length, half_context_length)
        left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

      doc_offset = doc_sent_start - left_context_length
      tokens = subwords[doc_offset : doc_sent_end + right_context_length]

      # add special tokens
      tokens += [sep_token]
      if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
      segment_ids = [sequence_a_segment_id] * len(tokens)
      if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
      else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      # only pad_on_right
      assert not pad_on_left
      padding_length = max_seq_length - len(input_ids)
      input_ids += ([pad_token] * padding_length)
      input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
      segment_ids += ([pad_token_segment_id] * padding_length)

      if example.langs and len(example.langs) > 0:
        langs = [example.langs[0]] * max_seq_length
      else:
        print('example.langs', example.langs, example.words, len(example.langs))
        print('ex_index', ex_index, len(examples))
        langs = None

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(langs) == max_seq_length

      # process entities
      entity_start_positions = []
      entity_end_positions = []
      entity_ids = []
      entity_position_ids = []
      original_entity_spans = []
      labels = []

      for entity_start in range(left_context_length, left_context_length + sentence_length):
        doc_entity_start = entity_start + doc_offset
        if doc_entity_start not in subword_start_positions:
          continue
        for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
          doc_entity_end = entity_end + doc_offset
          if doc_entity_end not in subword_start_positions:
            continue

          if entity_end - entity_start > max_mention_length:
            continue

          entity_start_positions.append(entity_start + 1)
          entity_end_positions.append(entity_end)
          entity_ids.append(entity_id)

          position_ids = list(range(entity_start + 1, entity_end + 1))
          position_ids += [-1] * (max_mention_length - entity_end + entity_start)
          entity_position_ids.append(position_ids)

          original_entity_spans.append(
            (subword2token[doc_entity_start], subword2token[doc_entity_end - 1] + 1)
          )
          labels.append(span_to_entity_label.pop((doc_entity_start, doc_entity_end), NON_ENTITY))

      label_ids = [label_map[label] for label in labels]
      # split instances
      split_size = math.ceil(len(entity_ids) / max_entity_length)
      for i in range(split_size):
        entity_size = math.ceil(len(entity_ids) / split_size)
        start = i * entity_size
        end = start + entity_size

        if ex_index < 5:
          logger.info("*** Example ***")
          logger.info("guid(doc_id): %s", example.guid)
          logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
          logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
          logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
          logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
          logger.info("langs: {}".format(langs))
          logger.info("entity_start_positions: %s", " ".join([str(x) for x in entity_start_positions[start:end]]))
          logger.info("entity_end_positions: %s", " ".join([str(x) for x in entity_end_positions[start:end]]))
          logger.info("label_ids: %s", " ".join([str(x) for x in label_ids[start:end]]))
          logger.info("original_entity_spans: %s", " ".join([str(x) for x in original_entity_spans[start:end]]))
          logger.info("input_words: %s", " ".join([str(x) for x in example.words]))

        features.append(
            InputFeatures(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    entity_start_positions=entity_start_positions[start:end],
                    entity_end_positions=entity_end_positions[start:end],
                    label_ids=label_ids[start:end],
                    original_entity_spans=original_entity_spans[start:end],
                    input_words=example.words,
                    doc_id=example.guid,
                    langs=langs))
        """
        fields = {
                "word_ids": TextField(word_ids, token_indexers=self.token_indexers),
                "entity_start_positions": TensorField(np.array(entity_start_positions[start:end])),
                "entity_end_positions": TensorField(np.array(entity_end_positions[start:end])),
                "original_entity_spans": TensorField(np.array(original_entity_spans[start:end]), padding_value=-1),
                "labels": ListField([LabelField(l) for l in labels[start:end]]),
                "doc_id": MetadataField(doc_index),
                "input_words": MetadataField(words),
            }
        """

  return features


def get_labels(path):
  with open(path, "r") as f:
    labels = f.read().splitlines()
  if "O" not in labels:
    labels = ["O"] + labels
  return labels
