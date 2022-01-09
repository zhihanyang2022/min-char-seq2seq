import pickle
import numpy as np


train_size = 200
test_size = 10

np.random.seed(42)

src_sentences, tgt_sentences = [], []
for i in range(train_size):
    src_sentence = str(np.random.randint(1000, 10000))
    src_sentences.append(src_sentence)
    tgt_sentences.append("".join(list(reversed(src_sentence))))

# generate testing data through rejection sampling (?)
# for simplicity, testing data is not gonna contain additional tokens

src_sentences_test, tgt_sentences_test = [], []
while len(src_sentences_test) < test_size:
    src_sentence_test = str(np.random.randint(1000, 10000))
    if src_sentence_test not in src_sentences:  # ensure that training and testing data do not overlap
        src_sentences_test.append(src_sentence_test)
        tgt_sentences_test.append("".join(list(reversed(src_sentence_test))))

for src_sentence_test in src_sentences_test:
    assert src_sentence_test not in src_sentences

with open('datasets/reverse_int_short/src_train.ob', 'wb+') as fp:
    pickle.dump(src_sentences, fp)

with open('datasets/reverse_int_short/tgt_train.ob', 'wb+') as fp:
    pickle.dump(tgt_sentences, fp)

with open('datasets/reverse_int_short/src_test.ob', 'wb+') as fp:
    pickle.dump(src_sentences_test, fp)

with open('datasets/reverse_int_short/tgt_test.ob', 'wb+') as fp:
    pickle.dump(tgt_sentences_test, fp)
