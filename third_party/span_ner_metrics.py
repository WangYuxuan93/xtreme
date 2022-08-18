import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


class SpanToLabelF1:
  def __init__(
        self, 
        label_map, 
        prediction_save_path: str = None, 
        output_mode="json",
        doc_id_to_sentence_boundaries = None
    ):
    self.label_map = label_map

    self.prediction = defaultdict(list)
    self.gold_labels = defaultdict(list)
    self.doc_id_to_words = {}
    self.doc_id_to_sentence_boundaries = {}
    self.prediction_save_path = prediction_save_path
    self.output_mode = output_mode
    self.doc_id_to_sentence_boundaries = doc_id_to_sentence_boundaries

  def __call__(
    self,
    prediction,
    gold_labels,
    prediction_scores,
    original_entity_spans,
    doc_id,
    input_words = None,
  ):

    if self.prediction_save_path is not None and input_words is None:
      raise RuntimeError("If you want to dump predictions, you need input_words.")

    #prediction, gold_labels, prediction_scores, original_entity_spans = self.detach_tensors(
    #  prediction, gold_labels, prediction_scores, original_entity_spans
    #)

    if input_words is not None:
      for id_, words in zip(doc_id, input_words):
        self.doc_id_to_words[id_] = words

    for pred, gold, scores, spans, id_ in zip(
      prediction, gold_labels, prediction_scores, original_entity_spans, doc_id
    ):
      pred = pred.tolist()
      gold = gold.tolist()
      scores = scores.tolist()
      #spans = spans.tolist()
      for p, g, score, span in zip(pred, gold, scores, spans):
        if g == -100:
          continue
        #p = self.vocab.get_token_from_index(p, namespace=self.label_namespace)
        #g = self.vocab.get_token_from_index(g, namespace=self.label_namespace)
        p = self.label_map[p]
        g = self.label_map[g]

        self.prediction[id_].append((score, span, p))
        self.gold_labels[id_].append((0, span, g))

  def reset(self):
    self.prediction = defaultdict(list)
    self.gold_labels = defaultdict(list)
  
  def write_txt(self, results, f):
    for doc_id in self.gold_labels.keys():
      prediction = self.span_to_label_sequence(self.prediction[doc_id])
      gold = self.span_to_label_sequence(self.gold_labels[doc_id])
      results.append({"words": self.doc_id_to_words[doc_id], "gold": gold, "prediction": prediction})
      sentence_boundaries = self.doc_id_to_sentence_boundaries[doc_id]
      words = self.doc_id_to_words[doc_id]
      for n in range(len(sentence_boundaries)-1):
        sent_start, sent_end = sentence_boundaries[n : n + 2]
        sent_words = words[sent_start:sent_end]
        sent_gold = gold[sent_start:sent_end]
        sent_pred = prediction[sent_start:sent_end]
        for w, g, p in zip(sent_words, sent_gold, sent_pred):
          f.write("\t".join([w, g, p])+"\n")
        f.write("\n")

  def get_metric(self):
    all_prediction_sequence = []
    all_gold_sequence = []
    results = []
    for doc_id in self.gold_labels.keys():
      prediction = self.span_to_label_sequence(self.prediction[doc_id])
      gold = self.span_to_label_sequence(self.gold_labels[doc_id])
      all_prediction_sequence.append(prediction)
      all_gold_sequence.append(gold)
      results.append({"words": self.doc_id_to_words[doc_id], "gold": gold, "prediction": prediction})

    if self.prediction_save_path is not None:
      with open(self.prediction_save_path, "w") as f:
        if self.output_mode == "json":
          json.dump(results, f)
        elif self.output_mode == "txt":
          if self.doc_id_to_sentence_boundaries is None:
            print ("doc_id_to_sentence_boundaries is not provided!")
            exit()
          self.write_txt(results, f)
        else:
          print ("Output_mode {} not recognized!".format(self.output_mode))

    return {
      "f1": f1_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
      "precision": precision_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
      "recall": recall_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
    }

    #return dict(
    #  f1=f1_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
    #  precision=precision_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
    #  recall=recall_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
    #)

  @staticmethod
  def span_to_label_sequence(span_and_labels: List[Tuple[float, Tuple[int, int], str]]) -> List[str]:
    sequence_length = max([end for score, (start, end), label in span_and_labels])
    label_sequence = ["O"] * sequence_length
    for score, (start, end), label in sorted(span_and_labels, key=lambda x: -x[0]):
      if label == "O" or any([l != "O" for l in label_sequence[start:end]]):
        continue
      label_sequence[start:end] = ["I-" + label] * (end - start)
      label_sequence[start] = "B-" + label

    return label_sequence
