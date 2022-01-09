import os
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import gin

from tokenizer import CharacterLevelTokenizer, sos_token, eos_token, pad_token
from seq2seq import Seq2Seq
from train_utils import get_training_hyperparams, load_dataset


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--expdir', type=str, default="experiments/test")
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    gin.parse_config_file(os.path.join(args.expdir, 'config.gin'))

    device = get_device()
    num_epochs, batch_size = get_training_hyperparams()

    # load toy datasets

    (src_sentences, tgt_sentences), (src_sentences_test, tgt_sentences_test) = load_dataset()

    if not args.infer:

        # train tokenizer

        tokenizer = CharacterLevelTokenizer()
        tokenizer.learn_new_vocabulary(src_sentences)
        tokenizer.learn_new_vocabulary(tgt_sentences)
        tokenizer.build_int2token_and_token2int()

        # preprocess training and testing data

        input_seqs = tokenizer.encode(src_sentences, for_decoder=False).to(device)
        target_seqs = tokenizer.encode(tgt_sentences, for_decoder=True).to(device)

        train_ds = TensorDataset(input_seqs, target_seqs)
        train_dl = DataLoader(train_ds, batch_size=32)

        input_seqs_test = tokenizer.encode(src_sentences_test, for_decoder=False).to(device)
        target_seqs_test = tokenizer.encode(tgt_sentences_test, for_decoder=True).to(device)

        # instantiate algorithm

        algo = Seq2Seq(
            num_tokens=len(tokenizer.vocabulary),
            sos_token_index=tokenizer.token2int[sos_token],
            eos_token_index=tokenizer.token2int[eos_token],
            pad_token_index=tokenizer.token2int[pad_token],
            device=get_device()
        )

        # training and testing loop

        acc_test_prev_best = 0

        for i in range(num_epochs):

            losses, accs = [], []

            for input_seqs_b, target_seqs_b in train_dl:

                loss_b, acc_b = algo.update_networks(input_seqs_b, target_seqs_b)  # for simplicity, train using all data

                losses.append(loss_b)
                accs.append(acc_b)

            loss = np.mean(losses)
            acc = np.mean(accs)

            loss_test, acc_test = algo.update_networks(input_seqs_test, target_seqs_test, just_do_forward=True)

            if acc_test > acc_test_prev_best:

                acc_test_prev_best = acc_test

                print(
                    f"Epoch {i+1:4} | Loss (train) {loss:10.2f} | Acc (train) {acc * 100.:10.2f} %" + " | " +
                    f"Loss (test) {loss:10.2f} | Acc (test) {acc_test * 100.:10.2f} % | NEW BEST TEST SCORE!"
                )

            else:

                print(
                    f"Epoch {i+1:4} | Loss (train) {loss:10.2f} | Acc (train) {acc * 100.:10.2f} %" + " | " +
                    f"Loss (test) {loss:10.2f} | Acc (test) {acc_test * 100.:10.2f} %"
                )

            if (i + 1) % 10 == 0:  # print some example source and generated target sequences

                print()

                print("##### Test set tranduction results #####")

                for j in range(len(src_sentences_test)):
                    if algo.use_attention:
                        predicted_target_seq, _ = algo.transduce(input_seqs_test[j])
                    else:
                        predicted_target_seq = algo.transduce(input_seqs_test[j])
                    predicted_target_sentence = tokenizer.decode(predicted_target_seq)
                    is_correct = tgt_sentences_test[j] == predicted_target_sentence
                    str_to_print = \
                        f"Source seq: {src_sentences_test[j]:>10} | Target seq (true): {tgt_sentences_test[j]:>10} | Target seq (gen): {predicted_target_sentence:>20} | {'correct' if is_correct else 'incorrect'}"
                    print(str_to_print)

                print()

        print("Saving tokenizer and model ...")

        tokenizer_dir = os.path.join(args.expdir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save(tokenizer_dir)

        print(tokenizer_dir)
        
        model_dir = os.path.join(args.expdir, "model")
        os.makedirs(model_dir, exist_ok=True)
        algo.save(model_dir)

    else:

        (_, _), (src_sentences_test, tgt_sentences_test) = load_dataset()

        tokenizer_dir = os.path.join(args.expdir, "tokenizer")
        model_dir = os.path.join(args.expdir, "model")

        tokenizer = CharacterLevelTokenizer()
        tokenizer.load(tokenizer_dir)

        algo = Seq2Seq(
            num_tokens=len(tokenizer.vocabulary),
            sos_token_index=tokenizer.token2int[sos_token],
            eos_token_index=tokenizer.token2int[eos_token],
            pad_token_index=tokenizer.token2int[pad_token],
            device=get_device(),
        )
        algo.load(model_dir)

        i = np.random.randint(len(src_sentences_test))

        print('Source               :', src_sentences_test[i])
        print('Target (truth)       :', tgt_sentences_test[i])

        input_seq = tokenizer.encode([src_sentences_test[i]], for_decoder=False).to(device)
        
        if algo.use_attention:

            predicted_target_seq, attention_matrix = algo.transduce(input_seq[0])
            predicted_target_sentence = tokenizer.decode(predicted_target_seq)
            
            print('Target (predicted)   :', predicted_target_sentence)
            
            plt.matshow(attention_matrix, cmap='gray')
            
            plt.xticks(list(range(len(src_sentences_test[i]))))
            plt.gca().set_xticklabels(list(src_sentences_test[i]))
            plt.xlim(-0.5, len(src_sentences_test[i]) - 0.5)
            
            plt.yticks(list(range(len(predicted_target_sentence))))
            plt.gca().set_yticklabels(list(predicted_target_sentence))
            plt.ylim(len(predicted_target_sentence) - 0.5, -0.5)

            plt.xlabel("Source Sentence (from left to right)")
            plt.gca().xaxis.set_label_position ('top')
            plt.ylabel("Predicted Target Sentence (from top to bottom)")

            plt.savefig(os.path.join(args.expdir, "attention_matrix.png"))

        else:
            predicted_target_seq = algo.transduce(input_seq[0])
            print('Target (predicted)   :', tokenizer.decode(predicted_target_seq))
