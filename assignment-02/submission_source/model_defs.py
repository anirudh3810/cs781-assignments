import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
MAX_LEN_HI = 30
class RNNCell(nn.Module):
    """Compute one vanilla-RNN timestep from primitive linear operations.

    Formula: ``h_t = tanh(W_ih x_t + b_ih + W_hh h_prev + b_hh)``.

    Args:
        input_size: Dimension of each input vector.
        hidden_size: Dimension of the recurrent hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Return ``h_t`` with shape ``[batch_size, hidden_size]``."""
        h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_prev))
        return h_t

class LSTMCell(nn.Module):
    """Compute one LSTM timestep using packed primitive linear projections.

    Four gate/candidate transformations are packed in ``x2g`` and ``h2g``.
    The returned state order is always ``(h_t, c_t)``.

    Args:
        input_size: Dimension of each input vector.
        hidden_size: Dimension of both hidden and cell states.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2g = nn.Linear(input_size, 4 * hidden_size)
        self.h2g = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x_t: torch.Tensor, state: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h_t, c_t)``, each shaped ``[batch_size, hidden_size]``."""
        h_prev, c_prev = state
        gates = self.x2g(x_t) + self.h2g(h_prev)
        f_pre, i_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
        f_t = torch.sigmoid(f_pre)
        i_t = torch.sigmoid(i_pre)
        g_t = torch.tanh(g_pre)
        o_t = torch.sigmoid(o_pre)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class GRUCell(nn.Module):
    """Compute one GRU timestep from six primitive linear transformations.

    Formula:
        ``z_t = sigmoid(W_iz x_t + b_iz + W_hz h_prev + b_hz)``
        ``r_t = sigmoid(W_ir x_t + b_ir + W_hr h_prev + b_hr)``
        ``n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_prev + b_hn))``
        ``h_t = (1 - z_t) * n_t + z_t * h_prev``

    Args:
        input_size: Dimension of each input vector.
        hidden_size: Dimension of the recurrent hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2z = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size)
        self.x2r = nn.Linear(input_size, hidden_size)
        self.h2r = nn.Linear(hidden_size, hidden_size)
        self.x2n = nn.Linear(input_size, hidden_size)
        self.h2n = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Return ``h_t`` with shape ``[batch_size, hidden_size]``."""
        z_t = torch.sigmoid(self.x2z(x_t) + self.h2z(h_prev))
        r_t = torch.sigmoid(self.x2r(x_t) + self.h2r(h_prev))
        n_t = torch.tanh(self.x2n(x_t) + r_t * self.h2n(h_prev))
        h_t = (1.0 - z_t) * n_t + z_t * h_prev
        return h_t

class RecurrentEncoder(nn.Module):
    """Build a multi-layer, optionally bidirectional scratch recurrent encoder.

    Args:
        cell_cls: One of ``RNNCell``, ``LSTMCell``, or ``GRUCell``.
        input_size: Input feature dimension at the first recurrent layer.
        hidden_size: Hidden-state dimension for every cell.
        num_layers: Number of stacked recurrent layers.
        bidirectional: Whether every layer has independent forward/backward cells.
        is_lstm: Whether cells propagate ``(hidden, cell)`` tuple states.
    """

    def __init__(self, cell_cls: type, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = False, is_lstm: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.is_lstm = is_lstm
        self.num_directions = 2 if bidirectional else 1
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = input_size if layer_idx == 0 else hidden_size * self.num_directions
            for _direction in range(self.num_directions):
                self.cells.append(cell_cls(in_dim, hidden_size))

    def _run_direction(self, cell: nn.Module, x: torch.Tensor, reverse: bool):
        """Explicitly unroll one recurrent cell across one sequence direction."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        c = torch.zeros_like(h) if self.is_lstm else None
        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        outputs = [None] * seq_len

        for timestep in steps:
            x_t = x[:, timestep, :]
            if self.is_lstm:
                h, c = cell(x_t, (h, c))
            else:
                h = cell(x_t, h)
            outputs[timestep] = h

        stacked_outputs = torch.stack(outputs, dim=1)
        final_state = (h, c) if self.is_lstm else h
        return stacked_outputs, final_state

    def forward(self, x: torch.Tensor):
        """Return top-layer sequence outputs and concatenated final hidden state."""
        layer_input = x
        cell_idx = 0
        final_h = None

        for _layer_idx in range(self.num_layers):
            dir_outputs, dir_finals = [], []
            for direction_idx in range(self.num_directions):
                cell = self.cells[cell_idx]
                cell_idx += 1
                is_reverse = self.bidirectional and direction_idx == 1
                output, final = self._run_direction(cell, layer_input, is_reverse)
                dir_outputs.append(output)
                h_final = final[0] if self.is_lstm else final
                dir_finals.append(h_final)

            layer_input = torch.cat(dir_outputs, dim=-1)
            final_h = torch.cat(dir_finals, dim=-1)

        return layer_input, final_h

class EmotionClassifier(nn.Module):
    """Combine token embeddings, a scratch recurrent encoder, and a linear head.

    Args:
        vocab_size: Number of entries in the emotion vocabulary.
        emb_dim: Token embedding dimension.
        hidden_size: Hidden dimension of each recurrent direction.
        num_classes: Number of independent emotion labels.
        cell_cls: Scratch recurrent cell class.
        num_layers: Number of recurrent layers.
        bidirectional: Whether the encoder reads in both directions.
        is_lstm: Whether the recurrent state includes an LSTM cell state.
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_classes: int,
                 cell_cls: type, num_layers: int, bidirectional: bool, is_lstm: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = RecurrentEncoder(
            cell_cls, emb_dim, hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            is_lstm=is_lstm,
        )
        classifier_in_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(classifier_in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized 28-label logits for token IDs ``x``."""
        embeds = self.embedding(x)
        _, final_hidden = self.encoder(embeds)
        logits = self.fc(self.dropout(final_hidden))
        return logits

class Encoder(nn.Module):
    """Embedding plus a 2-layer bidirectional native GRU and decoder bridge."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.bridge = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src: torch.Tensor):
        emb = self.embedding(src)
        lengths = (src != 0).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, final_h = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=src.size(1))
        dec_h0 = torch.tanh(self.bridge(torch.cat([final_h[-2], final_h[-1]], dim=-1)))
        return outputs, dec_h0

class DecoderStep(nn.Module):
    """Step-wise 2-layer unidirectional decoder using the scratch Q1 GRUCell."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cells = nn.ModuleList([
            GRUCell(emb_dim if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def hidden_step(self, tok: torch.Tensor, hidden_states: list[torch.Tensor]):
        """Advance the scratch GRU stack once without projecting vocabulary logits."""
        inp = self.embedding(tok)
        new_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(inp, hidden_states[layer])
            new_states.append(hidden)
            inp = hidden
        return new_states

    def forward(self, tok: torch.Tensor, hidden_states: list[torch.Tensor]):
        new_states = self.hidden_step(tok, hidden_states)
        logits = self.out(new_states[-1])
        return logits, new_states

class Seq2Seq(nn.Module):
    """Plain autoregressive wrapper supporting scratch RNN, LSTM, and GRU cells."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_layers: int = 2, is_lstm: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_layers = num_layers
        self.is_lstm = is_lstm
        self._tf_generator = torch.Generator(device="cpu")
        self._tf_generator.manual_seed(42)

    def _initial_states(self, initial_state):
        if self.is_lstm:
            h0, c0 = initial_state
            return [(h0.clone(), c0.clone()) for _ in range(self.num_layers)]
        return [initial_state.clone() for _ in range(self.num_layers)]

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor,
                teacher_forcing_ratio: float = 0.7):
        _, initial_state = self.encoder(src)
        states = self._initial_states(initial_state)
        tgt_len = tgt_in.size(1)
        input_token = tgt_in[:, 0]
        top_hidden_steps = []
        if teacher_forcing_ratio >= 1.0:
            forcing_draws = [0.0] * max(tgt_len - 1, 0)
        elif teacher_forcing_ratio <= 0.0:
            forcing_draws = [1.0] * max(tgt_len - 1, 0)
        else:
            forcing_draws = torch.rand(
                max(tgt_len - 1, 0), generator=self._tf_generator
            ).tolist()
        for timestep in range(tgt_len):
            states = self.decoder.hidden_step(input_token, states)
            top_hidden = states[-1][0] if self.is_lstm else states[-1]
            top_hidden_steps.append(top_hidden)
            if timestep + 1 < tgt_len:
                if forcing_draws[timestep] < teacher_forcing_ratio:
                    input_token = tgt_in[:, timestep + 1]
                else:
                    input_token = self.decoder.out(top_hidden).argmax(dim=-1)
        return self.decoder.out(torch.stack(top_hidden_steps, dim=1))

    @torch.no_grad()
    def greedy_decode(self, src_tensor: torch.Tensor,
                      max_len: int = MAX_LEN_HI - 1):
        self.eval()
        run_device = next(self.parameters()).device
        src_tensor = src_tensor.to(run_device, non_blocking=True)
        _, initial_state = self.encoder(src_tensor)
        states = self._initial_states(initial_state)
        input_token = torch.full(
            (src_tensor.size(0),), stoi_hi[SOS], dtype=torch.long,
            device=run_device,
        )
        finished = torch.zeros(
            src_tensor.size(0), dtype=torch.bool, device=run_device
        )
        predictions = []
        for _ in range(max_len):
            logits, states = self.decoder(input_token, states)
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, stoi_hi[PAD]), next_token
            )
            predictions.append(next_token.cpu())
            finished |= next_token.eq(stoi_hi[EOS])
            input_token = next_token
            if bool(finished.all()):
                break
        if predictions:
            return torch.stack(predictions, dim=1)
        return torch.empty(src_tensor.size(0), 0, dtype=torch.long)

class DotProductAttention(nn.Module):
    """Luong-style attention with a learned decoder-to-encoder projection."""

    def __init__(self, dec_hidden: int, enc_hidden: int):
        super().__init__()
        self.proj = nn.Linear(dec_hidden, enc_hidden)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor,
                mask: torch.Tensor = None):
        query = self.proj(dec_hidden).unsqueeze(1)
        scores = torch.bmm(query, enc_outputs.transpose(1, 2)).squeeze(1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights

class BahdanauAttention(nn.Module):
    """Additive attention with a learned nonlinear alignment energy."""

    def __init__(self, dec_hidden: int, enc_hidden: int, attn_dim: int = 128):
        super().__init__()
        self.W_a = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.U_a = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor,
                mask: torch.Tensor = None):
        query = self.W_a(dec_hidden).unsqueeze(1)
        keys = self.U_a(enc_outputs)
        energy = self.v_a(torch.tanh(query + keys))
        scores = energy.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights

class AttnDecoderStep(nn.Module):
    """Two-layer scratch-GRU decoder with dynamic attention and input feeding."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 enc_hidden: int, attention_module: nn.Module, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = attention_module
        self.cells = nn.ModuleList([
            GRUCell(emb_dim + enc_hidden if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size + enc_hidden, vocab_size)

    def hidden_step(self, tok: torch.Tensor, hidden_states: list[torch.Tensor],
                    enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        """Advance the recurrent stack and return the pre-projection features."""
        token_embedding = self.embedding(tok)
        context, weights = self.attention(hidden_states[-1], enc_outputs, mask)
        layer_input = torch.cat([token_embedding, context], dim=-1)
        new_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(layer_input, hidden_states[layer])
            new_states.append(hidden)
            layer_input = hidden
        output_features = torch.cat([new_states[-1], context], dim=-1)
        return new_states, weights, output_features

    def forward(self, tok: torch.Tensor, hidden_states: list[torch.Tensor],
                enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        new_states, weights, output_features = self.hidden_step(
            tok, hidden_states, enc_outputs, mask
        )
        logits = self.out(output_features)
        return logits, new_states, weights

class Seq2SeqAttn(nn.Module):
    """Attention wrapper compatible with scratch or native step-wise decoders."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_layers: int = 2, is_lstm: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_layers = num_layers
        self.is_lstm = is_lstm
        self._tf_generator = torch.Generator(device="cpu")
        self._tf_generator.manual_seed(42)

    def _initial_states(self, initial_state):
        if self.is_lstm:
            h0, c0 = initial_state
            return [(h0.clone(), c0.clone()) for _ in range(self.num_layers)]
        return [initial_state.clone() for _ in range(self.num_layers)]

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor,
                teacher_forcing_ratio: float = 0.7,
                return_attn: bool = False):
        enc_outputs, initial_state = self.encoder(src)
        states = self._initial_states(initial_state)
        mask = src.ne(stoi_en[PAD])
        tgt_len = tgt_in.size(1)
        input_token = tgt_in[:, 0]
        features_by_step, attention_by_step = [], []
        if teacher_forcing_ratio >= 1.0:
            forcing_draws = [0.0] * max(tgt_len - 1, 0)
        elif teacher_forcing_ratio <= 0.0:
            forcing_draws = [1.0] * max(tgt_len - 1, 0)
        else:
            forcing_draws = torch.rand(
                max(tgt_len - 1, 0), generator=self._tf_generator
            ).tolist()
        for timestep in range(tgt_len):
            states, weights, features = self.decoder.hidden_step(
                input_token, states, enc_outputs, mask
            )
            features_by_step.append(features)
            if return_attn:
                attention_by_step.append(weights)
            if timestep + 1 < tgt_len:
                if forcing_draws[timestep] < teacher_forcing_ratio:
                    input_token = tgt_in[:, timestep + 1]
                else:
                    input_token = self.decoder.out(features).argmax(dim=-1)
        logits = self.decoder.out(torch.stack(features_by_step, dim=1))
        if return_attn:
            return logits, torch.stack(attention_by_step, dim=1)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_tensor: torch.Tensor,
                      max_len: int = MAX_LEN_HI - 1,
                      return_attn: bool = True):
        self.eval()
        run_device = next(self.parameters()).device
        src_tensor = src_tensor.to(run_device, non_blocking=True)
        enc_outputs, initial_state = self.encoder(src_tensor)
        states = self._initial_states(initial_state)
        mask = src_tensor.ne(stoi_en[PAD])
        input_token = torch.full(
            (src_tensor.size(0),), stoi_hi[SOS], dtype=torch.long,
            device=run_device,
        )
        finished = torch.zeros(
            src_tensor.size(0), dtype=torch.bool, device=run_device
        )
        predictions, attention_steps = [], []
        for _ in range(max_len):
            logits, states, weights = self.decoder(
                input_token, states, enc_outputs, mask
            )
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, stoi_hi[PAD]), next_token
            )
            predictions.append(next_token.cpu())
            if return_attn:
                attention_steps.append(weights.cpu())
            finished |= next_token.eq(stoi_hi[EOS])
            input_token = next_token
            if bool(finished.all()):
                break
        tokens = (
            torch.stack(predictions, dim=1)
            if predictions else torch.empty(src_tensor.size(0), 0, dtype=torch.long)
        )
        attention = None
        if return_attn:
            attention = (
                torch.stack(attention_steps, dim=1)
                if attention_steps
                else torch.empty(src_tensor.size(0), 0, src_tensor.size(1))
            )
        return tokens, attention

class CustomEncoder(nn.Module):
    """Two-layer bidirectional encoder built from the Question 1 scratch cells."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 cell_cls: type, is_lstm: bool = False, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = RecurrentEncoder(
            cell_cls, emb_dim, hidden_size, num_layers=num_layers,
            bidirectional=True, is_lstm=is_lstm,
        )
        self.bridge = nn.Linear(2 * hidden_size, hidden_size)
        self.is_lstm = is_lstm

    def forward(self, src: torch.Tensor):
        emb = self.embedding(src)
        outputs, final_h_tensor = self.rnn(emb)
        dec_h0 = torch.tanh(self.bridge(final_h_tensor))
        if self.is_lstm:
            dec_c0 = torch.zeros_like(dec_h0)
            return outputs, (dec_h0, dec_c0)
        return outputs, dec_h0

class CustomDecoderStep(nn.Module):
    """Step-wise stacked decoder using a selected Question 1 scratch cell."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 cell_cls: type, is_lstm: bool = False, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_lstm = is_lstm
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cells = nn.ModuleList([
            cell_cls(emb_dim if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def hidden_step(self, tok: torch.Tensor, states: list):
        x = self.embedding(tok)
        new_states = []
        inp = x
        for layer, cell in enumerate(self.cells):
            if self.is_lstm:
                h_prev, c_prev = states[layer]
                h_next, c_next = cell(inp, (h_prev, c_prev))
                new_states.append((h_next, c_next))
                inp = h_next
            else:
                h_prev = states[layer]
                h_next = cell(inp, h_prev)
                new_states.append(h_next)
                inp = h_next
        return new_states

    def forward(self, tok: torch.Tensor, states: list):
        new_states = self.hidden_step(tok, states)
        top_hidden = new_states[-1][0] if self.is_lstm else new_states[-1]
        logits = self.out(top_hidden)
        return logits, new_states

class NativeRNSEncoder(nn.Module):
    """Two-layer bidirectional encoder using PyTorch's optimized nn.RNN."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(
            emb_dim, hidden_size, num_layers=num_layers, bidirectional=True,
            batch_first=True,
        )
        self.bridge = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src: torch.Tensor):
        emb = self.embedding(src)
        lengths = src.ne(0).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h_n = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        h_n = h_n.reshape(self.num_layers, 2, src.size(0), self.hidden_size)
        top_fwd = h_n[-1, 0]
        top_bwd = h_n[-1, 1]
        top_layer_combined = torch.cat([top_fwd, top_bwd], dim=-1)
        dec_h0 = torch.tanh(self.bridge(top_layer_combined))
        return outputs, dec_h0

class NativeLSTMEncoder(nn.Module):
    """Two-layer bidirectional encoder using PyTorch's optimized nn.LSTM."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_size, num_layers=num_layers, bidirectional=True,
            batch_first=True,
        )
        self.bridge_h = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_c = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src: torch.Tensor):
        emb = self.embedding(src)
        lengths = src.ne(0).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h_n, c_n) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        h_n = h_n.reshape(self.num_layers, 2, src.size(0), self.hidden_size)
        c_n = c_n.reshape(self.num_layers, 2, src.size(0), self.hidden_size)
        top_fwd_h, top_bwd_h = h_n[-1, 0], h_n[-1, 1]
        top_fwd_c, top_bwd_c = c_n[-1, 0], c_n[-1, 1]
        dec_h0 = torch.tanh(self.bridge_h(torch.cat([top_fwd_h, top_bwd_h], dim=-1)))
        dec_c0 = torch.tanh(self.bridge_c(torch.cat([top_fwd_c, top_bwd_c], dim=-1)))
        return outputs, (dec_h0, dec_c0)

class NativeGRUEncoder(nn.Module):
    """Two-layer bidirectional encoder using PyTorch's optimized nn.GRU."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            emb_dim, hidden_size, num_layers=num_layers, bidirectional=True,
            batch_first=True,
        )
        self.bridge = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src: torch.Tensor):
        emb = self.embedding(src)
        lengths = src.ne(0).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h_n = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        h_n = h_n.reshape(self.num_layers, 2, src.size(0), self.hidden_size)
        top_layer_combined = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        dec_h0 = torch.tanh(self.bridge(top_layer_combined))
        return outputs, dec_h0

class NativeRNNDecoderStep(nn.Module):
    """Step-wise plain decoder using built-in nn.RNNCell modules."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cells = nn.ModuleList([
            nn.RNNCell(emb_dim if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def hidden_step(self, tok: torch.Tensor, hidden_states: list):
        inp = self.embedding(tok)
        new_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(inp, hidden_states[layer])
            new_states.append(hidden)
            inp = hidden
        return new_states

    def forward(self, tok: torch.Tensor, hidden_states: list):
        new_states = self.hidden_step(tok, hidden_states)
        logits = self.out(new_states[-1])
        return logits, new_states

class NativeLSTMDecoderStep(nn.Module):
    """Step-wise plain decoder using built-in nn.LSTMCell modules."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cells = nn.ModuleList([
            nn.LSTMCell(emb_dim if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def hidden_step(self, tok: torch.Tensor, states: list):
        inp = self.embedding(tok)
        new_states = []
        for layer, cell in enumerate(self.cells):
            h_prev, c_prev = states[layer]
            h_next, c_next = cell(inp, (h_prev, c_prev))
            new_states.append((h_next, c_next))
            inp = h_next
        return new_states

    def forward(self, tok: torch.Tensor, states: list):
        new_states = self.hidden_step(tok, states)
        logits = self.out(new_states[-1][0])
        return logits, new_states

class NativeGRUDecoderStep(nn.Module):
    """Step-wise plain decoder using built-in nn.GRUCell modules."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cells = nn.ModuleList([
            nn.GRUCell(emb_dim if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def hidden_step(self, tok: torch.Tensor, hidden_states: list):
        inp = self.embedding(tok)
        new_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(inp, hidden_states[layer])
            new_states.append(hidden)
            inp = hidden
        return new_states

    def forward(self, tok: torch.Tensor, hidden_states: list):
        new_states = self.hidden_step(tok, hidden_states)
        logits = self.out(new_states[-1])
        return logits, new_states

class NativeAttnRNNDecoderStep(nn.Module):
    """Step-wise attention decoder using built-in nn.RNNCell modules."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 enc_hidden: int, attention_module: nn.Module,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = attention_module
        self.cells = nn.ModuleList([
            nn.RNNCell(
                emb_dim + enc_hidden if layer == 0 else hidden_size,
                hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size + enc_hidden, vocab_size)

    def hidden_step(self, tok: torch.Tensor, hidden_states: list,
                    enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        x = self.embedding(tok)
        context, weights = self.attention(hidden_states[-1], enc_outputs, mask)
        inp = torch.cat([x, context], dim=1)
        new_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(inp, hidden_states[layer])
            new_states.append(hidden)
            inp = hidden
        features = torch.cat([new_states[-1], context], dim=1)
        return new_states, weights, features

    def forward(self, tok: torch.Tensor, hidden_states: list,
                enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        new_states, weights, features = self.hidden_step(
            tok, hidden_states, enc_outputs, mask
        )
        logits = self.out(features)
        return logits, new_states, weights

class NativeAttnLSTMDecoderStep(nn.Module):
    """Step-wise attention decoder using built-in nn.LSTMCell modules."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 enc_hidden: int, attention_module: nn.Module,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = attention_module
        self.cells = nn.ModuleList([
            nn.LSTMCell(
                emb_dim + enc_hidden if layer == 0 else hidden_size,
                hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size + enc_hidden, vocab_size)

    def hidden_step(self, tok: torch.Tensor, states: list,
                    enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        x = self.embedding(tok)
        context, weights = self.attention(states[-1][0], enc_outputs, mask)
        inp = torch.cat([x, context], dim=1)
        new_states = []
        for layer, cell in enumerate(self.cells):
            h_prev, c_prev = states[layer]
            h_next, c_next = cell(inp, (h_prev, c_prev))
            new_states.append((h_next, c_next))
            inp = h_next
        features = torch.cat([new_states[-1][0], context], dim=1)
        return new_states, weights, features

    def forward(self, tok: torch.Tensor, states: list,
                enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        new_states, weights, features = self.hidden_step(
            tok, states, enc_outputs, mask
        )
        logits = self.out(features)
        return logits, new_states, weights

class NativeAttnGRUDecoderStep(nn.Module):
    """Step-wise attention decoder using built-in nn.GRUCell modules."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int,
                 enc_hidden: int, attention_module: nn.Module,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = attention_module
        self.cells = nn.ModuleList([
            nn.GRUCell(
                emb_dim + enc_hidden if layer == 0 else hidden_size,
                hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size + enc_hidden, vocab_size)

    def hidden_step(self, tok: torch.Tensor, hidden_states: list,
                    enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        x = self.embedding(tok)
        context, weights = self.attention(hidden_states[-1], enc_outputs, mask)
        inp = torch.cat([x, context], dim=1)
        new_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(inp, hidden_states[layer])
            new_states.append(hidden)
            inp = hidden
        features = torch.cat([new_states[-1], context], dim=1)
        return new_states, weights, features

    def forward(self, tok: torch.Tensor, hidden_states: list,
                enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        new_states, weights, features = self.hidden_step(
            tok, hidden_states, enc_outputs, mask
        )
        logits = self.out(features)
        return logits, new_states, weights

class NativeSeq2Seq(Seq2Seq):
    """Plain Seq2Seq wrapper for PyTorch-native recurrent encoders and cells."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_layers: int = 2, is_lstm: bool = False):
        super().__init__(encoder, decoder, num_layers=num_layers, is_lstm=is_lstm)

class NativeSeq2SeqAttn(Seq2SeqAttn):
    """Attention Seq2Seq wrapper for PyTorch-native recurrent components."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 num_layers: int = 2, is_lstm: bool = False):
        super().__init__(encoder, decoder, num_layers=num_layers, is_lstm=is_lstm)
