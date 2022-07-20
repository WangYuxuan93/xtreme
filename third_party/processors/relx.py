# -*- coding: utf-8 -*-

import logging
import os
import json
import copy
import torch
import numpy as np
from torch.utils.data import TensorDataset
from transformers import DataProcessor
from transformers import XLMTokenizer

logger = logging.getLogger(__name__)

class InputReExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, 
                guid,
                text, 
                label=None,
                language=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.language = language

    def show(self):
        logger.info("guid={}".format(self.guid))
        logger.info("text={}".format(self.text))
        logger.info("label={}".format(self.label))
        logger.info("language={}".format(self.language))
    
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputRelxFeatures(object):
    """A single set of features of data."""
    def __init__(
            self, 
            input_ids, 
            attention_mask, 
            token_type_ids,
            ent1_id, 
            ent2_id, 
            label=None
        ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.label = label


class RelxProcessor(DataProcessor):
    def __init__(self, task):
        #super().__init__(task)
        self.task = task
        self.root_token = "<ROOT>"

    def get_labels(self, train_filename):
        labels = []
        with open(train_filename) as f:
            raw_lines = f.read().splitlines()

        for i in range(len(raw_lines)//4):
            labels.append(raw_lines[4*i+1].strip())

        self.labels = sorted(list(set(labels)))
        self.label_map = {l:i for i, l in enumerate(self.labels)}
        return self.labels

    def get_examples(self, data_dir, language="en", split="train"):

        examples = []
        for lg in language.split(','):
            filename = os.path.join(data_dir, "{}-{}.tsv".format(split, lg))
            with open(filename) as f:
                raw_lines = f.read().splitlines()
        
            for i in range(len(raw_lines)//4):
                guid = "%s-%s-%s" % (split, language, i)
                text = raw_lines[4*i].split('\t')[1][1:-1]
                text = text.replace('<e1>', '<e1> ').replace('</e1>', ' </e1>').replace('<e2>', '<e2> ').replace('</e2>', ' </e2>')
                label = raw_lines[4*i+1].strip()
                examples.append(InputReExample(guid=guid, text=text, label=label, language=language))

        return examples

    def get_train_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='train')

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='test')


def convert_examples_to_features(
        examples, 
        tokenizer,
        max_length=512,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        lang2id=None,
    ):

    label_map = {label: i for i, label in enumerate(label_list)}

    # The special token id for markers before two entities
    ent1_tok_id = tokenizer._convert_token_to_id('<e1>')
    ent2_tok_id = tokenizer._convert_token_to_id('<e2>')

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        if isinstance(tokenizer, XLMTokenizer):
            inputs = tokenizer.encode_plus(example.text, add_special_tokens=True, max_length=max_length, lang=example.language)
        else:
            inputs = tokenizer.encode_plus(example.text, add_special_tokens=True, max_length=max_length)
        
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        try:
            ent1_id = input_ids.index(ent1_tok_id)
        except:
            logger.info("Cannot find ent1 <e1> (%d):\n%s" % (ent1_tok_id, " ".join([str(x) for x in input_ids])))
            example.show()
            exit()
        try:
            ent2_id = input_ids.index(ent2_tok_id)
        except:
            logger.info("Cannot find ent2 <e2> (%d):\n%s" % (ent2_tok_id, " ".join([str(x) for x in input_ids])))
            example.show()
            exit()

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        if lang2id is not None:
            lid = lang2id.get(example.language, lang2id["en"])
        else:
            lid = 0
            langs = [lid] * max_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 2:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            #logger.info("sid: %s" % (example.sid))
            #example.show()
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("ent1_id: %d, ent2_id: %d" % (ent1_id, ent2_id))
            logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("language: %s, (lid = %d)" % (example.language, lid))

        features.append(
            InputRelxFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                ent1_id=ent1_id,
                ent2_id=ent2_id,
                langs=langs,
                label=label
            )
        )

    return features
