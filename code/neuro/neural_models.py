# -*- coding: utf-8 -*-
# neural_models.py
# Модуль содержит архитектуры глубоких нейросетей для решения ATSP:
#   - PointerNetworkATSP (матричная версия Pointer Network)
#   - AttentionModelATSP (матричная Transformer-модель)
#   - TSPGNNATSP (графовая свёрточная сеть с предсказанием рёбер)
#   - MatNet (матричный кодировщик + декодер внимания)
#   - MatPOENet (UniCO – улучшенная версия MatNet с кросс-аттеншном)
#   - AttentionModelAM (координатная версия AM, требует 2D-координат)
#
# Все модели реализованы на PyTorch и могут использоваться для обучения с подкреплением
# либо для прямого инференса (детерминированный выбор действий).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
import numpy as np

# Вспомогательные функции
def normalize_distance_matrix(D, eps=1e-8):
    """Нормирует матрицу расстояний на максимальное значение."""
    max_val = D.max()
    if max_val > eps:
        return D / max_val
    return D


def compute_tour_length_from_matrix(route, D):
    """
    Вычисляет длину замкнутого маршрута по матрице расстояний.
    Предполагается, что route – список индексов (незамкнутый),
    замыкание (возврат к началу) добавляется автоматически.
    """
    length = 0.0
    for i in range(len(route) - 1):
        length += D[route[i], route[i + 1]]
    length += D[route[-1], route[0]]
    return length

class PointerEncoder(nn.Module):
    """LSTM-энкодер для Pointer Network."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class PointerDecoder(nn.Module):
    """
    Декодер Pointer Network с механизмом внимания.
    Поддерживает как argmax (для инференса), так и вычисление вероятностей
    для последующего сэмплирования.
    """
    def __init__(self, hidden_dim, n_cities):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.n_cities = n_cities

    def forward(self, encoder_outputs, hidden, mask):
        """
        Стандартный вызов: возвращает (выбранный индекс, вероятности).
        Используется при return_log_probs=False.
        """
        probs = self._compute_probs(encoder_outputs, hidden, mask)
        idx = torch.argmax(probs, dim=1)
        return idx, probs

    def get_probs(self, encoder_outputs, hidden, mask):
        """
        Возвращает только вероятности (для сэмплирования).
        """
        return self._compute_probs(encoder_outputs, hidden, mask)

    def _compute_probs(self, encoder_outputs, hidden, mask):
        batch, n, hdim = encoder_outputs.shape
        # скрытое состояние LSTM имеет форму (1, batch, hidden_dim)
        query = hidden.squeeze(0)   # (batch, hidden_dim)
        scores = []
        for i in range(n):
            h_i = encoder_outputs[:, i, :]
            combined = torch.cat([query, h_i], dim=1)
            score = self.attention(combined)
            scores.append(score)
        scores = torch.cat(scores, dim=1)              # (batch, n)
        scores = scores.masked_fill(mask, -1e9)
        probs = F.softmax(scores, dim=1)
        return probs


class PointerNetworkATSP(nn.Module):
    """
    Pointer Network для ATSP.
    На вход подаётся матрица расстояний (batch, n, n).

    При return_log_probs=False возвращает замкнутый маршрут (включая возврат)
    без информации о вероятностях.

    При return_log_probs=True сэмплирует действия и возвращает
    (маршрут, сумма логарифмов вероятностей сэмплированных действий).
    """
    def __init__(self, n_cities, hidden_dim=128):
        super().__init__()
        self.n_cities = n_cities
        self.input_proj = nn.Linear(n_cities, hidden_dim)
        self.encoder = PointerEncoder(hidden_dim, hidden_dim)
        self.decoder = PointerDecoder(hidden_dim, n_cities)

    def forward(self, dist_matrix, start_idx=0, return_log_probs=False):
        """
        Аргументы:
            dist_matrix (Tensor): (batch, n, n) матрица расстояний
            start_idx (int или Tensor): начальный индекс(ы)
            return_log_probs (bool): если True, возвращает (маршрут, log_prob)
        """
        batch, n, _ = dist_matrix.shape
        device = dist_matrix.device
        x = self.input_proj(dist_matrix)
        enc_out, hidden, cell = self.encoder(x)

        mask = torch.zeros(batch, n, dtype=torch.bool, device=device)
        mask[:, start_idx] = True
        route = [torch.full((batch,), start_idx, dtype=torch.long, device=device)]
        log_probs_list = [] if return_log_probs else None

        for _ in range(n - 1):
            if return_log_probs:
                probs = self.decoder.get_probs(enc_out, hidden, mask)
                m = dist.Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                log_probs_list.append(log_prob)
            else:
                action, _ = self.decoder(enc_out, hidden, mask)

            route.append(action)
            # Безопасное обновление маски: создаём новый тензор, избегая inplace-операций
            mask = mask | F.one_hot(action, n).bool()

        route = torch.stack(route, dim=1)               # (batch, n) незамкнутый

        if return_log_probs:
            sum_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1)
            return route, sum_log_prob                  # незамкнутый для REINFORCE
        else:
            route_closed = torch.cat([route, route[:, :1]], dim=1)  # замыкаем
            return route_closed

# Attention Model (матричная версия) – AttentionModelATSP
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        return self.attn(query, key, value, key_padding_mask=mask)[0]


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class AttentionModelATSP(nn.Module):
    """
    Модель на основе Transformer-энкодера с последующим декодированием вниманием.
    Принимает матрицу расстояний, возвращает замкнутый маршрут.

    При return_log_probs=True возвращает (маршрут, сумма log_prob).
    """
    def __init__(self, n_cities, embed_dim=128, num_heads=8, num_layers=3, ff_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(n_cities, embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.embed_dim = embed_dim

    def encode(self, dist_matrix):
        x = self.input_proj(dist_matrix)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, encoder_out, start_idx=0, sample=False):
        """
        Декодирует маршрут.
        Если sample=False – argmax, возвращает только маршрут.
        Если sample=True – сэмплирует из вероятностей, возвращает (маршрут, list of log_prob).
        """
        batch, n, d = encoder_out.shape
        device = encoder_out.device
        mask = torch.zeros(batch, n, dtype=torch.bool, device=device)
        mask[:, start_idx] = True
        query = encoder_out[:, start_idx:start_idx+1, :]
        route = [torch.full((batch,), start_idx, dtype=torch.long, device=device)]
        log_probs = [] if sample else None

        for _ in range(n - 1):
            scores = torch.matmul(query, encoder_out.transpose(1, 2)) / math.sqrt(d)
            scores = scores.squeeze(1)                      # (batch, n)
            scores = scores.masked_fill(mask, -1e9)
            probs = F.softmax(scores, dim=-1)

            if sample:
                m = dist.Categorical(probs)
                next_idx = m.sample()                       # (batch,)
                log_prob = m.log_prob(next_idx)             # (batch,)
                log_probs.append(log_prob)
            else:
                next_idx = torch.argmax(probs, dim=-1)      # (batch,)

            route.append(next_idx)
            # Безопасное обновление маски (не inplace)
            mask = mask | F.one_hot(next_idx, n).bool()
            query = encoder_out[torch.arange(batch), next_idx].unsqueeze(1)

        if sample:
            # незамкнутый маршрут — не добавляем стартовую точку
            route = torch.stack(route, dim=1)               # (batch, n)
            return route, log_probs
        else:
            # замкнутый маршрут
            route.append(route[0])                          # замыкание
            route = torch.stack(route, dim=1)               # (batch, n+1)
            return route

    def forward(self, dist_matrix, start_idx=0, return_log_probs=False):
        enc_out = self.encode(dist_matrix)
        if return_log_probs:
            route, log_probs_list = self.decode(enc_out, start_idx, sample=True)
            sum_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1)
            return route, sum_log_prob
        else:
            route = self.decode(enc_out, start_idx, sample=False)
            return route

# GNN – TSPGNNATSP
class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        deg = adj.sum(dim=-1, keepdim=True) + 1e-9
        adj_norm = adj / deg
        out = torch.bmm(adj_norm, x)
        out = self.W(out)
        return F.relu(out)


class TSPGNNATSP(nn.Module):
    """
    Графовая нейронная сеть: строит тепловую карту вероятностей рёбер,
    из которой затем жадным алгоритмом собирается маршрут.
    """
    def __init__(self, n_cities, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(n_cities, hidden_dim)
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, dist_matrix):
        batch, n, _ = dist_matrix.shape
        x = self.input_proj(dist_matrix)
        D_norm = dist_matrix / (dist_matrix.max(dim=1, keepdim=True)[0] + 1e-9)
        adj = torch.exp(-D_norm)
        for conv in self.convs:
            x = conv(x, adj)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        x_i = x[:, i_idx, :]
        x_j = x[:, j_idx, :]
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        logits = self.edge_predictor(edge_feat).squeeze(-1)
        probs = torch.zeros(batch, n, n, device=dist_matrix.device)
        probs[:, i_idx, j_idx] = torch.sigmoid(logits)
        probs[:, j_idx, i_idx] = probs[:, i_idx, j_idx]
        return probs

    def build_route_from_heatmap(self, probs, dist_matrix, start_idx=0):
        """Строит жадный маршрут по тепловой карте."""
        batch, n, _ = probs.shape
        routes = []
        for b in range(batch):
            mask = torch.zeros(n, dtype=torch.bool)
            route = [start_idx]
            mask[start_idx] = True
            current = start_idx
            for _ in range(n - 1):
                row = probs[b, current, :]
                row = row.masked_fill(mask, -1e9)
                next_node = torch.argmax(row).item()
                route.append(next_node)
                mask[next_node] = True
                current = next_node
            route.append(start_idx)
            routes.append(route)
        return routes

class MatNet(nn.Module):
    """
    Модель MatNet: кодирует строки матрицы расстояний в эмбеддинги,
    затем последовательно выбирает следующий город с помощью MLP-внимания.
    Возвращает последовательность действий (незамкнутую) и логарифмы вероятностей.
    """
    def __init__(self, n_nodes, embedding_dim=128, n_heads=8, n_layers=3, ff_dim=512, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.row_embedding = nn.Sequential(
            nn.Linear(n_nodes, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, dist_matrix, start_idx=0, return_log_probs=False):  # добавлен параметр
        """
        Args:
            dist_matrix: (batch, n_nodes, n_nodes)
            start_idx: int или Tensor – начальный индекс
            return_log_probs: bool, игнорируется (оставлен для совместимости)
        Returns:
            actions: (batch, n_nodes) – незамкнутый маршрут
            log_probs: (batch,) – сумма лог-вероятностей действий
        """
        batch_size, n_nodes, _ = dist_matrix.shape
        device = dist_matrix.device
        D_norm = dist_matrix / (dist_matrix.max(dim=1, keepdim=True)[0] + 1e-9)
        node_embeddings = self.row_embedding(D_norm)
        node_embeddings = self.encoder(node_embeddings)

        if isinstance(start_idx, int):
            mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
            mask[:, start_idx] = True
            actions = [torch.full((batch_size,), start_idx, dtype=torch.long, device=device)]
        else:
            mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
            mask = mask.scatter(1, start_idx.unsqueeze(1), True)
            actions = [start_idx]

        log_probs = []
        for _ in range(n_nodes - 1):
            last_action = actions[-1]
            query = node_embeddings[torch.arange(batch_size), last_action].unsqueeze(1)
            keys = node_embeddings
            scores = self.output_layer(torch.cat([query.expand(-1, n_nodes, -1), keys], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(mask, -1e9)
            probs = F.log_softmax(scores, dim=-1)
            next_node = torch.argmax(probs, dim=-1)      # детерминированный выбор
            actions.append(next_node)
            log_probs.append(probs[torch.arange(batch_size), next_node])
            # Безопасное обновление маски
            mask = mask | F.one_hot(next_node, n_nodes).bool()

        actions = torch.stack(actions, dim=1)  # (batch, n_nodes) незамкнутый
        log_probs = torch.stack(log_probs, dim=1)
        return actions, log_probs.sum(dim=-1)

class MatPOENet(nn.Module):
    """
    Улучшенная версия MatNet с кросс-аттеншном и более глубоким энкодером.
    """
    def __init__(self, n_nodes, embedding_dim=128, n_heads=8, n_layers=3, ff_dim=512, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.poe_embedding = nn.Sequential(
            nn.Linear(n_nodes, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.cross_attn_layer = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.scale = embedding_dim ** 0.5

    def forward(self, dist_matrix, start_idx=0, return_log_probs=False):  # добавлен параметр
        """
        Args:
            dist_matrix: (batch, n_nodes, n_nodes)
            start_idx: int или Tensor
            return_log_probs: bool, игнорируется
        Returns:
            actions: (batch, n_nodes) – незамкнутый маршрут
            log_probs: (batch,) – сумма лог-вероятностей
        """
        batch_size, n_nodes, _ = dist_matrix.shape
        device = dist_matrix.device
        D_norm = dist_matrix / (dist_matrix.max(dim=1, keepdim=True)[0] + 1e-9)
        node_embeddings = self.poe_embedding(D_norm)
        attn_out, _ = self.cross_attn_layer(node_embeddings, node_embeddings, node_embeddings)
        node_embeddings = self.norm1(node_embeddings + attn_out)
        node_embeddings = self.encoder(node_embeddings)

        mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        if isinstance(start_idx, int):
            mask[:, start_idx] = True
            actions = [torch.full((batch_size,), start_idx, dtype=torch.long, device=device)]
        else:
            mask = mask.scatter(1, start_idx.unsqueeze(1), True)
            actions = [start_idx]

        log_probs = []
        Q = self.W_q(node_embeddings)
        K = self.W_k(node_embeddings)
        for _ in range(n_nodes - 1):
            last_action = actions[-1]
            q = Q[torch.arange(batch_size), last_action].unsqueeze(1)
            scores = torch.matmul(q, K.transpose(1, 2)) / self.scale
            scores = scores.squeeze(1)
            scores = scores.masked_fill(mask, -1e9)
            probs = F.log_softmax(scores, dim=-1)
            next_node = torch.argmax(probs, dim=-1)
            actions.append(next_node)
            log_probs.append(probs[torch.arange(batch_size), next_node])
            mask = mask | F.one_hot(next_node, n_nodes).bool()

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return actions, log_probs.sum(dim=-1)

class TSPGNNLearner(nn.Module):
    """
    Обучаемая графовая модель для ATSP.
    Использует графовую свёрточную сеть для получения эмбеддингов узлов,
    затем на каждом шаге вычисляет attention-оценки от текущего узла
    к непосещённым, формирует распределение и выбирает действие.
    При return_log_probs=True сэмплирует из softmax и возвращает (маршрут, log_prob).
    При False – argmax, возвращает замкнутый маршрут (инференс).
    """
    def __init__(self, n_cities, hidden_dim=128, num_layers=3):
        super().__init__()
        self.n_cities = n_cities
        self.input_proj = nn.Linear(n_cities, hidden_dim)   # проекция строк матрицы
        self.convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        # Attention: запрос (текущий узел) и ключи (все узлы) -> скор
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, dist_matrix, start_idx=0, return_log_probs=False):
        """
        dist_matrix: (batch, n, n)
        start_idx: int или Tensor
        return_log_probs: bool
        Возвращает:
            если return_log_probs=True:  (tour_unclosed, log_prob)
            иначе:                        tour_closed
        """
        batch, n, _ = dist_matrix.shape
        device = dist_matrix.device

        # Строим эмбеддинги узлов через GNN
        x = self.input_proj(dist_matrix)                     # (batch, n, hidden_dim)
        D_norm = dist_matrix / (dist_matrix.max(dim=2, keepdim=True)[0] + 1e-9)
        adj = torch.exp(-D_norm)                             # взвешенная матрица смежности
        for conv in self.convs:
            x = conv(x, adj)                                 # (batch, n, hidden_dim)

        # Вычисляем ключи один раз
        K = self.attn_key(x)                                 # (batch, n, hidden_dim)

        mask = torch.zeros(batch, n, dtype=torch.bool, device=device)
        mask[:, start_idx] = True
        route = [torch.full((batch,), start_idx, dtype=torch.long, device=device)]
        log_probs_list = [] if return_log_probs else None

        for _ in range(n - 1):
            last_idx = route[-1]                             # (batch,)
            q = self.attn_query(x[torch.arange(batch), last_idx]).unsqueeze(1)  # (batch, 1, hidden)
            # attention scores
            scores = torch.matmul(q, K.transpose(1, 2)).squeeze(1) / self.scale  # (batch, n)
            scores = scores.masked_fill(mask, -1e9)
            probs = F.softmax(scores, dim=-1)

            if return_log_probs:
                m = dist.Categorical(probs)
                next_node = m.sample()
                log_prob = m.log_prob(next_node)
                log_probs_list.append(log_prob)
            else:
                next_node = torch.argmax(probs, dim=-1)

            route.append(next_node)
            mask = mask | F.one_hot(next_node, n).bool()

        route = torch.stack(route, dim=1)                    # (batch, n) незамкнутый
        if return_log_probs:
            sum_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1)
            return route, sum_log_prob
        else:
            # Замыкаем для инференса
            return torch.cat([route, route[:, :1]], dim=1)
