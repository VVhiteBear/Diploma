# -*- coding: utf-8 -*-
# utils.py
# Вспомогательные функции для проекта по оптимизации маршрутов

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

# Константы
INF = 1e9               # условная "бесконечность" для отсутствующих путей
EARTH_RADIUS_M = 6371000 # радиус Земли в метрах

# Геодезические расстояния
def calculate_geodesic_distance(lat1, lon1, lat2, lon2):
    """
    Расстояние между двумя точками на сфере по формуле гаверсинусов.
    Возвращает расстояние в метрах.
    """
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return EARTH_RADIUS_M * c

# Работа с матрицами расстояний
def check_matrix_asymmetry(D, tolerance=1.0):
    """
    Проверяет матрицу расстояний на асимметричность.

    Параметры:
        D (np.ndarray): квадратная матрица расстояний
        tolerance (float): порог различия в метрах (по умолч. 1 м)

    Возвращает:
        max_diff: максимальная абсолютная разница между D[i][j] и D[j][i]
        asym_pairs: количество асимметричных пар
        total_pairs: общее количество пар (без диагонали)
        percent: процент асимметричных пар
    """
    diff = np.abs(D - D.T)
    max_diff = np.max(diff)
    total_pairs = D.size - D.shape[0]
    asym_pairs = np.sum(diff > tolerance)
    percent = (asym_pairs / total_pairs) * 100 if total_pairs > 0 else 0
    return max_diff, asym_pairs, total_pairs, percent

def save_distance_matrix(D, filepath):
    """Сохраняет матрицу расстояний в файл .npy"""
    np.save(filepath, D)

def load_distance_matrix(filepath):
    """Загружает матрицу расстояний из файла .npy"""
    return np.load(filepath)

# Работа с маршрутами
def calculate_route_distance(route, distance_matrix):
    """
    Вычисляет полную длину маршрута по матрице расстояний.

    Параметры:
        route (list): список индексов складов в порядке посещения
        distance_matrix (np.ndarray): матрица расстояний (асимметричная)

    Возвращает:
        float: суммарное расстояние в метрах
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i+1]]
    return total

def route_to_km(route, distance_matrix):
    """Возвращает длину маршрута в километрах."""
    return calculate_route_distance(route, distance_matrix) / 1000.0

def is_valid_tsp_route(route, n_cities, start_end_same=True):
    """
    Проверяет корректность маршрута для TSP/ATSP.

    Параметры:
        route (list): маршрут (список индексов)
        n_cities (int): количество городов (складов)
        start_end_same (bool): должен ли первый и последний совпадать

    Возвращает:
        bool: True если маршрут корректен
    """
    if start_end_same and route[0] != route[-1]:
        return False
    internal = route[:-1] if start_end_same else route
    if len(set(internal)) != n_cities:
        return False
    if max(internal) >= n_cities or min(internal) < 0:
        return False
    return True

def normalize_route(route):
    """
    Приводит маршрут к каноническому виду (без дублирования стартовой точки в конце).
    """
    if route and route[0] == route[-1]:
        return route[:-1]
    return route

# Валидация маршрута на графе
def validate_route_on_graph(graph, route_indices, warehouse_nodes, distance_matrix=None, tolerance=1.0):
    """
    Проверяет, что суммарная длина маршрута, построенного по графу, совпадает
    с длиной, вычисленной по матрице расстояний. Также возвращает полный список
    узлов графа вдоль маршрута и координаты для визуализации.

    Параметры:
        graph (nx.MultiDiGraph): ориентированный граф дорог
        route_indices (list): последовательность индексов складов (начинается и заканчивается start_idx)
        warehouse_nodes (list): список OSM-узлов, соответствующих складам (по порядку)
        distance_matrix (np.ndarray, optional): матрица расстояний для сравнения
        tolerance (float): допустимая разница между графом и матрицей (в метрах)

    Возвращает:
        full_coords (list of tuple): список координат (lat, lon) всех точек маршрута
        graph_length (float): общая длина маршрута по графу (метры)
        is_consistent (bool): True, если разница с матрицей меньше tolerance
    """
    total_length_graph = 0.0
    full_path_nodes = []
    is_consistent = True

    for i in range(len(route_indices) - 1):
        u = warehouse_nodes[route_indices[i]]
        v = warehouse_nodes[route_indices[i+1]]
        try:
            # Кратчайший путь по графу (ориентированный)
            path = nx.shortest_path(graph, u, v, weight='length')
            length = nx.path_weight(graph, path, weight='length')
            total_length_graph += length
            if i == 0:
                full_path_nodes.extend(path)
            else:
                full_path_nodes.extend(path[1:])  # избегаем дублирования узлов
        except nx.NetworkXNoPath:
            print(f"Путь между {u} и {v} не найден")
            return [], float('inf'), False

    # Сравнение с матрицей, если она передана
    if distance_matrix is not None:
        matrix_length = calculate_route_distance(route_indices, distance_matrix)
        diff = abs(total_length_graph - matrix_length)
        if diff > tolerance:
            print(f"Несовпадение длин: граф = {total_length_graph/1000:.2f} км, матрица = {matrix_length/1000:.2f} км (разница {diff:.1f} м)")
            is_consistent = False
        else:
            print(f"Длины совпадают: {total_length_graph/1000:.2f} км (разница {diff:.2f} м)")

    # Координаты для визуализации (широта, долгота)
    coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in full_path_nodes]
    return coords, total_length_graph, is_consistent

def plot_route_on_map(coords, warehouses_df, route_indices, title, filename, figsize=(12, 10)):
    """
    Отрисовывает маршрут на карте с отметками складов.

    Параметры:
        coords (list of tuple): список координат (lat, lon) пути
        warehouses_df (pd.DataFrame): информация о складах (latitude, longitude, name)
        route_indices (list): последовательность индексов складов
        title (str): заголовок графика
        filename (str): путь для сохранения изображения
        figsize (tuple): размер фигуры
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Отрисовка маршрута
    if coords:
        lats = [p[0] for p in coords]
        lons = [p[1] for p in coords]
        ax.plot(lons, lats, 'b-', linewidth=2, label='Маршрут')

    # Отметки складов
    for idx, row in warehouses_df.iterrows():
        if idx in route_indices:
            if idx == route_indices[0]:
                color, marker, size, label = 'red', '*', 150, 'Старт'
            else:
                color, marker, size, label = 'green', 'o', 80, 'Склад'
            ax.scatter(row['longitude'], row['latitude'], c=color, marker=marker, s=size,
                       edgecolors='black', zorder=5, label=label if idx == route_indices[0] else "")
            ax.annotate(row['name'], (row['longitude'], row['latitude']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Убираем дублирование легенды
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Карта сохранена: {filename}")

# Работа со складами
def load_warehouses_from_json(json_path):
    """
    Загружает список складов из JSON файла (формат, созданный вашим геокодером).
    Возвращает pandas DataFrame с колонками: name, latitude, longitude.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for item in data:
        records.append({
            'name': item['name'],
            'latitude': item['latitude'],
            'longitude': item['longitude']
        })
    return pd.DataFrame(records)

# Метрики обучения RL
def compute_rl_training_metrics(episode_distances, window=100):
    """
    Вычисляет метрики процесса обучения RL-агента.

    Параметры:
        episode_distances (list): список длин маршрутов по эпизодам
        window (int): размер окна для вычисления стабильности

    Возвращает:
        dict: словарь с метриками
    """
    if not episode_distances:
        return {}
    arr = np.array(episode_distances)
    best_idx = np.argmin(arr)
    metrics = {
        'best_distance_km': float(arr[best_idx]),
        'best_episode': int(best_idx) + 1,
        'final_distance_km': float(arr[-1]),
        'mean_last_window': float(np.mean(arr[-window:])) if len(arr) >= window else float(np.mean(arr)),
        'std_last_window': float(np.std(arr[-window:])) if len(arr) >= window else float(np.std(arr)),
        'convergence_episode': _find_convergence_episode(arr, threshold=0.01)
    }
    return metrics

def _find_convergence_episode(distances, threshold=0.01):
    """
    Определяет эпизод, после которого улучшение становится меньше порога.
    """
    best_so_far = float('inf')
    stable_count = 0
    for i, d in enumerate(distances):
        if d < best_so_far * (1 - threshold):
            best_so_far = d
            stable_count = 0
        else:
            stable_count += 1
        if stable_count >= 50:  # не улучшается 50 эпизодов подряд
            return i - 50 + 1
    return len(distances)

def save_rl_metrics(metrics, filepath):
    """Сохраняет метрики RL в JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_rl_metrics(filepath):
    """Загружает метрики RL из JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Сохранение результатов экспериментов
def save_results_to_json(results, filepath):
    """Сохраняет результаты экспериментов в JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_results_from_json(filepath):
    """Загружает результаты из JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def results_to_dataframe(results_list):
    """Преобразует список словарей с результатами в pandas DataFrame."""
    return pd.DataFrame(results_list)

# Вспомогательные функции для отладки
def print_route_summary(route, distance_matrix, method_name=""):
    """
    Выводит краткую информацию о маршруте: длина в км и первые/последние точки.
    """
    dist_km = route_to_km(route, distance_matrix)
    route_str = str(route[:5])[:-1] + "..." + str(route[-3:])[1:] if len(route) > 8 else str(route)
    print(f"{method_name}: {dist_km:.2f} км, маршрут: {route_str}")

# Работа с архивами
import zipfile
import os

def zip_folder(folder_path, output_zip=None):
    """
    Создаёт zip-архив папки, сохраняя корневую папку внутри архива.

    Аргументы:
        folder_path (str): путь к папке для архивации.
        output_zip (str, optional): имя выходного архива. По умолчанию folder_path.zip.

    Возвращает:
        str: путь к созданному архиву.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} не является папкой.")

    if output_zip is None:
        output_zip = folder_path.rstrip('/\\') + '.zip'

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Помещаем файл в архив с путём относительно родительской директории папки,
                # чтобы внутри архива оказалась папка folder_path/...
                arcname = os.path.relpath(file_path, start=os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
    print(f"Папка {folder_path} заархивирована в {output_zip}")
    return output_zip

def unzip_all(zip_files, extract_to='.'):
    """
    Распаковывает все zip-архивы из списка напрямую в указанную директорию.

    Аргументы:
        zip_files (list): список имён zip-файлов.
        extract_to (str): директория для распаковки (по умолчанию текущая).
    """
    import zipfile
    import os

    for archive in zip_files:
        if not os.path.exists(archive):
            print(f"Архив {archive} не найден, пропускаем.")
            continue

        print(f"Распаковываем {archive} в {extract_to}")
        with zipfile.ZipFile(archive, 'r') as zipf:
            zipf.extractall(extract_to)
        print(f"Распакован: {archive}")
