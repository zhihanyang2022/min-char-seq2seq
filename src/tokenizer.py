from typing import List
import os
import pickle
import numpy as np
import torch


sos_token = "<SOS>"
eos_token = "<EOS>"
pad_token = "<PAD>"


class CharacterLevelTokenizer:

    """
    Tokenize a string of alphabets and integers into individual characters.
    For example, "ab6710d" =(tokenizer)=> ["a", "b", "6", "7", "1", "0", "d"].
    """

    def __init__(self):
        self.vocabulary = [sos_token, eos_token, pad_token]
        self.int2token = {}
        self.token2int = {}

    def learn_new_vocabulary(self, sentences: List[str]) -> None:
        """
        Build a set consisting of all unique words in a list of sentences.
        Can be called multiple times.
        """
        for sentence in sentences:
            for char in sentence:
                if char not in self.vocabulary:
                    self.vocabulary.append(char)

    def build_int2token_and_token2int(self) -> None:
        """
        Build two dictionaries from vocabulary.
        Should be called only once, after all calls to learn_new_vocabulary are done.
        """
        for token_idx, token in enumerate(self.vocabulary):
            self.int2token[token_idx] = token
            self.token2int[token] = token_idx

    def encode(self, sentences: List[str], for_decoder: bool) -> torch.tensor:
        """Obtain training data from raw sentences."""
        max_len = np.max([len(s) for s in sentences])
        encoded_sentences = []
        for sentence in sentences:
            chars = list(sentence)
            if for_decoder:
                chars = [sos_token] + chars + [eos_token] + [pad_token] * (max_len - len(chars))
            else:
                chars = chars + [pad_token] * (max_len - len(chars))
            encoded_sentence = []
            for c in chars:
                encoded_sentence.append(self.token2int[c])
            encoded_sentences.append(encoded_sentence)
        return torch.tensor(encoded_sentences).long()

    def decode(self, encoded_sentence: list):
        decoded_sentence = []
        for token_idx in encoded_sentence:
            decoded_sentence.append(self.int2token[token_idx])
        return "".join(decoded_sentence)

    def save(self, save_dir):
        with open(os.path.join(save_dir, "vocabulary.ob"), 'wb+') as fp:
            pickle.dump(self.vocabulary, fp)
        with open(os.path.join(save_dir, "int2token.ob"), 'wb+') as fp:
            pickle.dump(self.int2token, fp)
        with open(os.path.join(save_dir, "token2int.ob"), 'wb+') as fp:
            pickle.dump(self.token2int, fp)

    def load(self, save_dir):
        with open(os.path.join(save_dir, "vocabulary.ob"), 'rb') as fp:
            self.vocabulary = pickle.load(fp)
        with open(os.path.join(save_dir, "int2token.ob"), 'rb') as fp:
            self.int2token = pickle.load(fp)
        with open(os.path.join(save_dir, "token2int.ob"), 'rb') as fp:
            self.token2int = pickle.load(fp)
