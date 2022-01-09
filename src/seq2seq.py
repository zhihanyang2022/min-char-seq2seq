import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import gin

from networks import Encoder, Decoder, DecoderWithAttention


SEQ_LEN_UPPER_LIM = 100


@gin.configurable(module=__name__)
class Seq2Seq:

    """Implement the sequence-2-sequence algorithm."""

    def __init__(self, num_tokens, sos_token_index, eos_token_index, pad_token_index, device, use_attention=False, lr=1e-4, max_grad_norm=1):

        self.num_tokens = num_tokens
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_index, reduction='sum')
        self.sos_token_index = sos_token_index
        self.eos_token_index = eos_token_index
        self.pad_token_index = pad_token_index
        self.device = device
        
        self.use_attention = use_attention
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        self.encoder = Encoder(num_tokens).to(device)
        self.decoder = DecoderWithAttention(num_tokens).to(device) if self.use_attention else Decoder(num_tokens).to(device)

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=self.lr)

    def update_networks(self, src_seqs, tgt_seqs, just_do_forward=False):

        if just_do_forward:

            self.encoder.eval()
            self.decoder.eval()

        else:

            self.encoder.train()
            self.decoder.train()

        if isinstance(self.decoder, DecoderWithAttention):
            encoder_outputs, hiddens = self.encoder(src_seqs, return_outputs=True)
        else:
            hiddens = self.encoder(src_seqs, return_outputs=False)

        current_indices = tgt_seqs[:, 0]  # represents SOS

        loss_sum = 0
        num_correct = 0
        max_seq_len = tgt_seqs.shape[1]

        for t in range(0, max_seq_len - 1):

            # prediction

            if isinstance(self.decoder, DecoderWithAttention):
                next_logits, hiddens, _ = self.decoder.predict_next_logits(current_indices, hiddens, encoder_outputs)  # next_logits has shape (bs, num_tokens)
            else:
                next_logits, hiddens = self.decoder.predict_next_logits(current_indices, hiddens)

            # computing loss

            next_indices_true = tgt_seqs[:, t+1]  # (bs, )

            loss_sum_t = self.loss_fn(next_logits, next_indices_true)
            loss_sum += loss_sum_t

            # computing acc

            next_indices_generated = next_logits.argmax(dim=1)

            num_correct_t = torch.sum(
                next_indices_generated.eq(next_indices_true) * ~(next_indices_true.eq(self.pad_token_index))
            )
            num_correct += num_correct_t

            # preparing for next timestep

            next_indices = next_indices_true if np.random.uniform() <= 0.5 else next_indices_generated

            current_indices = next_indices  # for next iteration

        # computing the number of entries over which we computed loss_sum and num_correct

        num_non_pad_entries = int(torch.sum(~tgt_seqs[:, 1:].eq(self.pad_token_index)))
        loss_ = loss_sum / num_non_pad_entries
        acc_ = num_correct / num_non_pad_entries

        if not just_do_forward:

            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()

            loss_.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.max_grad_norm)

            self.encoder_optim.step()
            self.decoder_optim.step()

        return float(loss_), float(acc_)

    def transduce(self, input_seq):
        """
        Take a single src sequence, and transduce it into a single tgt sequence.
        :param input_seq:
        :param start_token:
        :param end_token:
        :return:
        """

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():

            # input_seq has shape (seq_len, )
            # start_token has shape (1, )

            input_seq = input_seq.unsqueeze(0)  # (1, seq_len)

            encoder_outputs, hidden = self.encoder(input_seq, return_outputs=True)  # (1, seq_len, hidden_size)

            current_index = torch.tensor([self.sos_token_index]).long().to(self.device)
            eos_token_index = torch.tensor([self.eos_token_index]).long().to(self.device)

            target_seq_generated = [] # int(self.sos_token_index)]

            list_of_attention_vectors = []

            while True:

                if isinstance(self.decoder, DecoderWithAttention):
                    next_logits, hidden, attention_vectors = self.decoder.predict_next_logits(current_index, hidden,
                                                                                              encoder_outputs)
                    # next_logits has shape (bs, num_tokens)
                    list_of_attention_vectors.append(attention_vectors)
                else:
                    next_logits, hidden = self.decoder.predict_next_logits(current_index, hidden)

                next_index_generated = next_logits.argmax(dim=1)

                # print(int(next_index_generated), (attention_vectors.view(-1) > 0.5).float().cpu().numpy())

                if int(next_index_generated) != eos_token_index:
                    target_seq_generated.append(int(next_index_generated))
                
                current_index = next_index_generated  # for next iteration  

                if int(current_index) == eos_token_index or len(target_seq_generated) >= SEQ_LEN_UPPER_LIM:
                    break

            if isinstance(self.decoder, DecoderWithAttention):

                # print(list_of_attention_vectors[0].shape)

                attention_matrix = torch.cat(list_of_attention_vectors, dim=0).cpu().numpy()
                # attention_matrix =
                # attention_matrix[np.abs(attention_matrix) < 0.01] = 0
                # attention_matrix[np.abs(attention_matrix) > 0.01] = 1
                # print(attention_matrix)

                return target_seq_generated, attention_matrix

            else:

                return target_seq_generated

    def save(self, save_dir):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))

    def load(self, save_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(save_dir, "encoder.pth"), map_location=self.device))
        self.decoder.load_state_dict(torch.load(os.path.join(save_dir, "decoder.pth"), map_location=self.device))
