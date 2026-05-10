# -*- coding: utf-8 -*-
# preprocess.py

#pip install osmnx

# Однократная подготовка данных: загрузка графа, складов,
# построение асимметричных матриц расстояний для всех сценариев шума,
# сохранение матриц и вспомогательной информации.
# Создаёт файлы как для 12, так и для 29 складов с разными префиксами.

import os
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    INF, save_distance_matrix, load_warehouses_from_json,
    check_matrix_asymmetry, zip_folder, unzip_all
)
from noise_scenarios import generate_noise_scenarios, check_warehouses_accessibility

# Выходная директория
OUTPUT_DIR = "preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Пути к исходным данным (могут быть в raw_data/ или в корне)
RAW_DATA_DIR = "raw_data"
GRAPH_FILE_CANDIDATES = [
    os.path.join(RAW_DATA_DIR, "moscow_region_drive_network.graphml"),
    "moscow_region_drive_network.graphml"
]
WAREHOUSES_JSON_RC_CANDIDATES = [
    os.path.join(RAW_DATA_DIR, "warehouses_rc_rfc_coordinates.json"),
    "warehouses_rc_rfc_coordinates.json"
]
WAREHOUSES_JSON_ALL_CANDIDATES = [
    os.path.join(RAW_DATA_DIR, "all_warehouses_coordinates.json"),
    "all_warehouses_coordinates.json"
]

# Префиксы для имён файлов
PREFIX_12 = ""          # для 12 складов – без префикса
PREFIX_29 = "all_"      # для 29 складов – префикс "all_"

def ensure_data_available():
    if not os.path.exists(RAW_DATA_DIR) and os.path.exists("raw_data.zip"):
        print("Распаковываем raw_data.zip...")
        unzip_all(["raw_data.zip"], extract_to=".")
    # Если граф всё ещё не найден, но есть в корне – find_file справится

# 1. Загрузка графа
def find_file(candidates):
    """Ищет первый существующий файл из списка кандидатов."""
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Ни один из файлов не найден: {candidates}")

def load_graph():
    """Загружает дорожный граф из сохранённого файла."""
    graph_file = find_file(GRAPH_FILE_CANDIDATES)
    graph = ox.load_graphml(graph_file)
    print(f"Граф загружен из {graph_file}: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер")
    return graph

# 2. Загрузка и привязка складов
def load_and_prepare_warehouses(json_path_candidates, graph):
    """Загружает склады из JSON, привязывает к ближайшим узлам графа."""
    json_path = find_file(json_path_candidates)
    df = load_warehouses_from_json(json_path)
    nodes = []
    for idx, row in df.iterrows():
        point = (row['latitude'], row['longitude'])
        nearest = ox.distance.nearest_nodes(graph, point[1], point[0])
        nodes.append(nearest)
    df['node_id'] = nodes
    return df

# 3. Построение матрицы расстояний
def compute_asymmetric_distance_matrix(graph, warehouse_nodes):
    n = len(warehouse_nodes)
    D = np.full((n, n), INF, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0.0
            else:
                try:
                    d = nx.shortest_path_length(graph, warehouse_nodes[i], warehouse_nodes[j], weight='length')
                    D[i][j] = float(d)
                except nx.NetworkXNoPath:
                    pass
    return D

# 4. Сохранение информации о складах
def save_warehouse_info(df, scenario_name, prefix):
    info = {'scenario': scenario_name, 'warehouses': []}
    for idx, row in df.iterrows():
        info['warehouses'].append({
            'id': idx,
            'name': row['name'],
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'node_id': int(row['node_id'])
        })
    safe_name = scenario_name.replace(' ', '_').replace('+', '_')
    out_file = os.path.join(OUTPUT_DIR, f'{prefix}warehouse_info_{safe_name}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Сохранена информация о складах: {out_file}")

# 5. Анализ и визуализация матриц
def analyze_and_visualize_matrices(matrices, prefix, warehouses_df):
    """
    Для набора матриц (словарь {сценарий: матрица}) выводит статистику,
    проверяет асимметрию и строит тепловые карты.
    """
    print(f"Анализ матриц расстояний (префикс '{prefix}')")

    stats = []
    for name, D in matrices.items():
        # Проверка асимметрии
        max_diff, asym_pairs, total_pairs, percent = check_matrix_asymmetry(D, tolerance=1.0)
        # Количество достижимых пар (не INF)
        finite_mask = (D < INF/2) & (D > 0)
        num_finite = np.sum(finite_mask)
        mean_dist = np.mean(D[finite_mask]) / 1000 if num_finite > 0 else float('inf')
        max_dist = np.max(D[finite_mask]) / 1000 if num_finite > 0 else float('inf')
        unreachable = total_pairs - num_finite

        stats.append({
            'Сценарий': name,
            'Среднее расстояние (км)': mean_dist,
            'Макс. расстояние (км)': max_dist,
            'Асимметричных пар (%)': percent,
            'Недостижимых пар': unreachable
        })
        print(f"\n{name}:")
        print(f"  Среднее расстояние: {mean_dist:.2f} км")
        print(f"  Максимальное расстояние: {max_dist:.2f} км")
        print(f"  Асимметричных пар: {asym_pairs} ({percent:.1f}%)")
        print(f"  Недостижимых пар: {unreachable}")

    # Сохраняем статистику в CSV
    stats_df = pd.DataFrame(stats)
    csv_path = os.path.join(OUTPUT_DIR, f'{prefix}matrix_statistics.csv')
    stats_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nСтатистика сохранена: {csv_path}")

    # Тепловые карты для первых трёх сценариев (или всех, если их <=3)
    n_plots = min(3, len(matrices))
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    for ax, (name, D) in zip(axes, list(matrices.items())[:n_plots]):
        D_log = np.where(D < INF/2, np.log1p(D), np.nan)
        sns.heatmap(D_log, ax=ax, cmap='viridis', cbar_kws={'label': 'log(1 + расстояние, м)'})
        ax.set_title(f'{name}')
        short_names = [w[:10] for w in warehouses_df['name']]
        n = len(short_names)
        ax.set_xticks(range(n))          # устанавливаем позиции тиков
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_names, rotation=0, fontsize=8)
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, f'{prefix}distance_heatmaps.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Тепловые карты сохранены: {heatmap_path}")

# 6. Обработка одного набора складов
def process_warehouse_set(json_path_candidates, prefix, description, graph):
    print(f"\nОбработка набора: {description} (префикс '{prefix}')")

    warehouses_df = load_and_prepare_warehouses(json_path_candidates, graph)
    warehouse_nodes = warehouses_df['node_id'].tolist()
    start_warehouse = 0
    print(f"Загружено складов: {len(warehouses_df)}")
    print(f"Начальная точка: {warehouses_df.iloc[start_warehouse]['name']}")

    # Генерация сценариев шума (без визуализации дорог, чтобы не загромождать)
    scenarios = generate_noise_scenarios(
        graph, warehouses_df, warehouse_nodes,
        start_index=start_warehouse, random_seed=42,
        save_visualizations=False
    )

    matrices = {}
    for scenario_name, scenario_graph in scenarios.items():
        print(f"\n  Сценарий: {scenario_name}")
        accessible = check_warehouses_accessibility(
            scenario_graph, warehouse_nodes, warehouses_df['name'].tolist()
        )
        if not accessible:
            print(f"    Не все склады достижимы!")

        dist_matrix = compute_asymmetric_distance_matrix(scenario_graph, warehouse_nodes)
        matrices[scenario_name] = dist_matrix

        safe_name = scenario_name.replace(' ', '_').replace('+', '_')
        matrix_file = os.path.join(OUTPUT_DIR, f'{prefix}distance_matrix_{safe_name}.npy')
        save_distance_matrix(dist_matrix, matrix_file)
        print(f"    Матрица сохранена: {matrix_file}")

        save_warehouse_info(warehouses_df, safe_name, prefix)

    # Анализ и визуализация матриц
    analyze_and_visualize_matrices(matrices, prefix, warehouses_df)

# 7. Основная функция
def preprocess_all():
    ensure_data_available()
    graph = load_graph()

    # 12 складов (РЦ/РФЦ)
    process_warehouse_set(WAREHOUSES_JSON_RC_CANDIDATES, PREFIX_12, "12 складов (РЦ/РФЦ)", graph)

    # 29 складов (все)
    process_warehouse_set(WAREHOUSES_JSON_ALL_CANDIDATES, PREFIX_29, "29 складов (все)", graph)

    print(f"\nПредобработка завершена. Данные в папке '{OUTPUT_DIR}'.")
    # Архивация (если доступна)
    if 'zip_folder' in globals():
        zip_path = zip_folder(OUTPUT_DIR)
        print(f"Архив создан: {zip_path}")
    else:
        print("Функция zip_folder не найдена.")

# 8. Функции для загрузки данных
def load_preprocessed_matrices(use_all=False, scenario_names=None):
    if scenario_names is None:
        scenario_names = ['Без_шума', 'Пробки', 'Блокировки_Пробки', 'Умеренные_условия']
    prefix = PREFIX_29 if use_all else PREFIX_12
    matrices = {}
    for name in scenario_names:
        safe_name = name.replace(' ', '_').replace('+', '_')
        file_path = os.path.join(OUTPUT_DIR, f'{prefix}distance_matrix_{safe_name}.npy')
        if os.path.exists(file_path):
            matrices[name] = np.load(file_path)
        else:
            print(f"Файл не найден: {file_path}")
    return matrices

def load_warehouse_info(use_all=False, scenario_name='Без_шума'):
    prefix = PREFIX_29 if use_all else PREFIX_12
    safe_name = scenario_name.replace(' ', '_').replace('+', '_')
    file_path = os.path.join(OUTPUT_DIR, f'{prefix}warehouse_info_{safe_name}.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data['warehouses'])

# 9. Точка входа
if __name__ == "__main__":
    preprocess_all()
