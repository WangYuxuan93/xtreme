import json
import logging
import math
import multiprocessing
from collections import Counter, OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, TextIO

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, MASK_TOKEN}

Entity = namedtuple("Entity", ["title", "language"])

logger = logging.getLogger(__name__)


class EntityVocab:
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        # allow tsv files for backward compatibility
        if vocab_file.endswith(".tsv"):
            logger.info("Detected vocab file type: tsv")
            self._parse_tsv_vocab_file(vocab_file)
        elif vocab_file.endswith(".jsonl"):
            logger.info("Detected vocab file type: jsonl")
            self._parse_jsonl_vocab_file(vocab_file)
        elif "mluke" in vocab_file:
            logger.info("Detected vocab file type: pretrained transformers")
            self._from_pretrained_mluke(vocab_file)
        elif "luke" in vocab_file:
            logger.info("Detected vocab file type: pretrained transformers")
            self._from_pretrained_luke(vocab_file)
        else:
            raise ValueError(f"Unrecognized vocab_file format: {vocab_file}")

        self.special_token_ids = {}
        for special_token in SPECIAL_TOKENS:
            special_token_entity = self.search_across_languages(special_token)[0]
            self.special_token_ids[special_token] = self.get_id(*special_token_entity)

        self.mask_id = self.special_token_ids[MASK_TOKEN]
        self.pad_id = self.special_token_ids[PAD_TOKEN]
        self.unk_id = self.special_token_ids[UNK_TOKEN]

    def _from_pretrained_mluke(self, transformer_model_name: str):
        from transformers.models.mluke.tokenization_mluke import MLukeTokenizer

        mluke_tokenizer = MLukeTokenizer.from_pretrained(transformer_model_name)
        title_to_idx = mluke_tokenizer.entity_vocab
        mluke_special_tokens = SPECIAL_TOKENS | {"[MASK2]"}
        for title, idx in title_to_idx.items():
            if title in mluke_special_tokens:
                entity = Entity(title, None)
            else:
                language, title = title.split(":", maxsplit=1)
                entity = Entity(title, language)
            self.vocab[entity] = idx
            self.counter[entity] = None
            self.inv_vocab[idx].append(entity)

    def _from_pretrained_luke(self, transformer_model_name: str):
        from transformers.models.luke.tokenization_luke import LukeTokenizer

        luke_tokenizer = LukeTokenizer.from_pretrained(transformer_model_name)
        title_to_idx = luke_tokenizer.entity_vocab
        for title, idx in title_to_idx.items():
            entity = Entity(title, None)
            self.vocab[entity] = idx
            self.counter[entity] = None
            self.inv_vocab[idx].append(entity)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        return self.get_id(key, language=None)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: str = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def search_across_languages(self, title: str) -> List[Entity]:
        results = []
        for entity in self.vocab.keys():
            if entity.title == title:
                results.append(entity)
        return results

    def save(self, out_file: str):

        if Path(out_file).suffix != ".jsonl":
            raise ValueError(
                "The saved file has to explicitly have the jsonl extension so that it will be loaded properly,\n"
                f"but the name provided is {out_file}."
            )

        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count}
                json.dump(item, f)
                f.write("\n")