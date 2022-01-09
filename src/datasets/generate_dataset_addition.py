import pickle
import numpy as np


train_size = 5000
test_size = 10

np.random.seed(42)

src_sentences, tgt_sentences = [], []
for i in range(train_size):
    first_integer_str = "".join(list(map(str, list(np.random.randint(10, size=(3, ))))))
    second_integer_str = "".join(list(map(str, list(np.random.randint(10, size=(3, ))))))
    src_sentences.append(first_integer_str + "+" + second_integer_str)
    tgt_sentences.append(str(int(first_integer_str) + int(second_integer_str)))

# generate testing data through rejection sampling (?)
# for simplicity, testing data is not gonna contain additional tokens

src_sentences_test, tgt_sentences_test = [], []
while len(src_sentences_test) < test_size:
    first_integer_str = "".join(list(map(str, list(np.random.randint(10, size=(3, ))))))
    second_integer_str = "".join(list(map(str, list(np.random.randint(10, size=(3, ))))))
    src_sentence_test = first_integer_str + "+" + second_integer_str
    if src_sentence_test not in src_sentences:  # ensure that training and testing data do not overlap
        src_sentences_test.append(src_sentence_test)
        tgt_sentences_test.append(str(int(first_integer_str) + int(second_integer_str)))

for src_sentence_test in src_sentences_test:
    assert src_sentence_test not in src_sentences

with open('./addition/src_train.ob', 'wb+') as fp:
    pickle.dump(src_sentences, fp)

with open('./addition/tgt_train.ob', 'wb+') as fp:
    pickle.dump(tgt_sentences, fp)

with open('./addition/src_test.ob', 'wb+') as fp:
    pickle.dump(src_sentences_test, fp)

with open('./addition/tgt_test.ob', 'wb+') as fp:
    pickle.dump(tgt_sentences_test, fp)
