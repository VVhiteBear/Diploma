# -*- coding: utf-8 -*-
# rl_environment.py
import numpy as np

class ATSPEnvironment:
    """
    Среда для асимметричной задачи коммивояжёра (ATSP) с изменяемой матрицей расстояний.

    Изменения:
    - Состояние дополнено вектором расстояний от текущего узла до всех (нормализованным).
      Это позволяет агенту учитывать дорожную обстановку и адаптироваться к меняющимся матрицам.
    - Если выбор уже посещённого узла произошёл (например, из-за численной ошибки),
      агент мягко перенаправляется на ближайший допустимый узел без штрафа.
      Основная защита от невалидных действий – маскирование в нейросети.

    Состояние: current (int), visited_mask (float32[n_nodes]), dist_vector (float32[n_nodes]).
    Действие: выбор следующего узла из непосещённых.
    Награда: отрицательное расстояние перехода (пошаговая), плюс возврат в стартовую точку в конце.
    """

    def __init__(self, distance_matrix, start_idx=0):
        """
        Args:
            distance_matrix (np.ndarray): квадратная асимметричная матрица расстояний
            start_idx (int): индекс начального узла
        """
        self.distance_matrix = distance_matrix
        self.n_nodes = distance_matrix.shape[0]
        self.start_idx = start_idx
        self.reset()

    def reset(self):
        """Сброс среды в начальное состояние."""
        self.current = self.start_idx
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.visited[self.start_idx] = True
        self.route = [self.start_idx]
        self.total_distance = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Формирует словарь состояния для агента.
           Содержит нормализованный вектор расстояний от текущего узла.
        """
        # Расстояния от текущего узла до всех остальных (в метрах)
        raw_dists = self.distance_matrix[self.current]  # shape (n_nodes,)
        # Нормализация: делим на максимум всей матрицы, чтобы значения были порядка 1.
        # Используем максимум по всей матрице, чтобы агенту было проще сравнивать сценарии.
        max_dist = max(self.distance_matrix.max(), 1e-6)
        norm_dists = raw_dists / max_dist
        return {
            'current': self.current,
            'visited_mask': self.visited.astype(np.float32),
            'dist_vector': norm_dists.astype(np.float32)
        }

    def step(self, action):
        """
        Выполняет действие агента.

        Args:
            action (int): индекс следующего узла

        Returns:
            state (dict): новое состояние
            reward (float): награда за шаг
            done (bool): флаг завершения эпизода
            info (dict): дополнительная информация
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        # Проверка на выход за границы
        if action < 0 or action >= self.n_nodes:
            raise ValueError(f"Action {action} out of bounds [0, {self.n_nodes})")

        # Если агент выбрал уже посещённый узел (чего не должно происходить благодаря маскированию),
        # не наказываем его экстремально, а просто перенаправляем на ближайший допустимый узел.
        if self.visited[action]:
            # Находим все непосещённые узлы
            valid_actions = np.where(~self.visited)[0]
            if len(valid_actions) > 0:
                # Выбираем узел с минимальным оставшимся расстоянием от текущего
                action = valid_actions[np.argmin(self.distance_matrix[self.current, valid_actions])]
            else:
                # Такого случиться не может, но на всякий случай завершаем
                self.done = True
                # Искусственно добавляем возврат
                self.total_distance += self.distance_matrix[self.current, self.start_idx]
                self.route.append(self.start_idx)
                return self._get_state(), 0.0, True, {}

        # Основная награда – отрицательное расстояние перехода
        dist = self.distance_matrix[self.current, action]
        reward = -dist
        self.total_distance += dist

        # Обновление состояния
        self.current = action
        self.visited[action] = True
        self.route.append(action)

        # Проверка завершения (посещены все узлы)
        self.done = np.all(self.visited)

        if self.done:
            # Добавляем возврат в стартовую точку
            return_dist = self.distance_matrix[self.current, self.start_idx]
            self.total_distance += return_dist
            reward -= return_dist
            self.route.append(self.start_idx)

        return self._get_state(), float(reward), self.done, {}

    def get_route_info(self):
        """Возвращает информацию о текущем маршруте."""
        return {
            'route': self.route,
            'distance_km': self.total_distance / 1000.0,
            'visited_count': int(np.sum(self.visited))
        }
