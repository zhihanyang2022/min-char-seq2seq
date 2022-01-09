import torch
import torch.nn as nn

import gin


@gin.configurable(module=__name__)
class Encoder(nn.Module):

    def __init__(self, num_tokens, embedding_size=300, embedding_dropout=0.1, hidden_size=1024, bidirectional=False):
        super().__init__()

        self.num_tokens = num_tokens  # include PAD, SOS and EOS
        self.embedding_size = embedding_size
        self.embedding_dropout = embedding_dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(self.embedding_dropout)
        self.embedding = nn.Embedding(self.num_tokens, self.embedding_size)
        self.gru = nn.RNN(self.embedding_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.outputs_projector = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.hiddens_projector = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, src_seqs: torch.tensor, return_outputs) -> torch.tensor:

        # input: src_seqs is a padded tensor, (bs, max_seq_len, num_tokens)
        # return: 
        # - hiddens is a tensor of shape (1, bs, hidden_size)
        # = outputs is a tensor of shape (bs, max_seq_len, hidden_size)

        embedded_seqs = self.dropout(self.embedding(src_seqs))
        outputs, hiddens = self.gru(embedded_seqs)  # (bs, max_le, num_directions), (num_directions, bs, hidden_size)

        if self.bidirectional:

            outputs = self.outputs_projector(outputs)

            forward_hiddens = hiddens[0]  # (bs, hidden_size)
            backward_hiddens = hiddens[0]  # (bs, hidden_size)
            combined_hiddens = self.hiddens_projector(torch.cat([forward_hiddens, backward_hiddens], dim=-1))
            hiddens = combined_hiddens.unsqueeze(0)

        if return_outputs:
            return outputs, hiddens
        else:
            return hiddens


@gin.configurable(module=__name__)
class Decoder(nn.Module):

    def __init__(self, num_tokens, embedding_size=300, embedding_dropout=0.1, hidden_size=1024):
        super().__init__()

        self.num_tokens = num_tokens  # include PAD, SOS and EOS
        self.embedding_size = embedding_size
        self.embedding_dropout = embedding_dropout
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(self.embedding_dropout)
        self.embedding = nn.Embedding(self.num_tokens, self.embedding_size)
        self.gru = nn.RNN(self.embedding_size, self.hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_tokens)

    # def forward(self, tgt_seqs_a, hiddens_from_encoder):
    #     """
    #     For training without teacher forcing (I will be using this since it's used in training transformers).
    #     :param tgt_seqs_a: this is just a convenient name for tgt_seqs[:, :-1, :], shape (bs, max_seq_len - 1, num_tokens)
    #     :param hiddens_from_encoder: context vectors from encoder, shape (1, bs, hidden_size)
    #     :return: predicted logits for tgt_seqs_b (tgt_seqs[:, 1:, :])
    #     """
    #     embedded_seqs = self.embedding(tgt_seqs_a)
    #     temp, _ = self.gru(embedded_seqs, hiddens_from_encoder)
    #     tgt_seqs_b_predicted_logit = self.fc(temp)
    #     return tgt_seqs_b_predicted_logit  # (bs, max_seq_len, vocab_size)

    def predict_next_logits(self, current_indices, hiddens_from_prev_step):
        """
        For generating an output seqence on step at a time.
        :param encoder_outputs:
        :param current_indices: shape (bs, )
        :param hiddens_from_prev_step: shape (1, bs, hidden_size)
        :return: logits for the next token, updated hiddens
        """

        current_indices = current_indices.unsqueeze(1)  # (bs, 1)
        embedded = self.dropout(self.embedding(current_indices))  # (bs, 1, embedding_size)

        outputs, hiddens = self.gru(embedded, hiddens_from_prev_step)  # outputs has shape (bs, 1, hidden_size)

        next_logits = self.fc(outputs).view(-1, self.num_tokens)  # next_logits has shape (bs, num_tokens)

        return next_logits, hiddens


@gin.configurable(module=__name__)
class DecoderWithAttention(nn.Module):

    def __init__(self, num_tokens, embedding_size=300, embedding_dropout=0.1, hidden_size=1024):
        super().__init__()

        self.num_tokens = num_tokens  # include PAD, SOS and EOS
        self.embedding_size = embedding_size
        self.embedding_dropout = embedding_dropout
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(self.embedding_dropout)
        self.embedding = nn.Embedding(self.num_tokens, self.embedding_size)
        self.gru = nn.RNN(self.embedding_size + self.hidden_size, self.hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_tokens)

    def predict_next_logits(self, current_indices, hiddens_from_prev_step, encoder_outputs):
        """
        :param encoder_outputs:
        :param current_indices: shape (bs, )
        :param hiddens_from_prev_step: shape (1, bs, hidden_size)
        :return: logits for the next token, updated hiddens
        """

        current_indices = current_indices.unsqueeze(1)  # (bs, 1)
        embedded = self.dropout(self.embedding(current_indices))  # (bs, 1, embedding_size)

        # ATTENTION MECHANISM

        # (1) computing attention vectors (one per batch)

        # hiddens_from_prev_step: (1, bs, hidden_size) =(permute)=> (bs, hidden_size, 1)
        # encoder_outputs: (bs, max_seq_len, hidden_size)

        # batch-matmul shape: (bs, max_seq_len, hidden_size) @ (bs, hidden_size, 1) = (bs, max_seq_len, 1)
        # [:, :, 0] simply removes the training one; using squeeze causes problem when batch size is 1

        energy_vectors = torch.bmm(encoder_outputs,
                                   hiddens_from_prev_step.permute(1, 2, 0))[:, :, 0]  # (bs, max_seq_len)
        attention_vectors = energy_vectors.softmax(dim=-1)  # (bs, max_seq_len)

        # (2) computing context vectors (one per batch)

        # attention_vectors: (bs, max_seq_len) =(unsqueeze)=> (bs, max_seq_len, 1)
        # encoder_outputs: (bs, max_seq_len, hidden_size) =(permute)=> (bs, hidden_size, max_seq_len)

        # remember from linear algebra the the result of matmul between a matrix and a vector
        # is the linear combination of the columns of the matrix weighted by the entries of the vectors,
        # which is exactly what we what

        # batch-matmul shape: (bs, hidden_size, max_seq_len) @ (bs, max_seq_len, 1) = (bs, hidden_size, 1)

        context_vectors = torch.bmm(encoder_outputs.permute(0, 2, 1),
                                    attention_vectors.unsqueeze(-1))[:, :, 0]

        # (3) passing the concatenation of hiddens_from_prev_step and context_vectors into gru

        # hiddens_from_prev_step_modified = torch.cat([hiddens_from_prev_step, context_vectors.unsqueeze(0)], dim=-1)

        outputs, hiddens = self.gru(torch.cat([embedded, context_vectors.unsqueeze(1)], dim=-1), hiddens_from_prev_step)

        next_logits = self.fc(outputs).view(-1, self.num_tokens)  # next_logits has shape (bs, num_tokens)
        return next_logits, hiddens, attention_vectors
