# -*- coding: utf-8 -*-
# noise_scenarios.py
# Модуль для моделирования шумовых сценариев на дорожной сети:
#   - пробки (умножение длины дорог)
#   - блокировки (очень большая длина)
#   - комбинированные и умеренные сценарии
# Также включает функции визуализации изменённых дорог и анализа влияния шума.

import os
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from collections import Counter
from utils import INF, calculate_route_distance

# Папка для сохранения результатов
OUTPUT_DIR = "noise_scenarios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Базовые функции изменения графа
def add_traffic_jams(graph, jam_roads, traffic_multiplier=10):
    """
    Моделирует пробки на указанных дорогах путём увеличения их длины.
    Возвращает копию графа и количество изменённых рёбер.
    """
    graph_jammed = graph.copy()
    modified = 0
    for u, v, key in jam_roads:
        if graph_jammed.has_edge(u, v, key):
            orig_len = graph_jammed[u][v][key].get('length', 0)
            graph_jammed[u][v][key]['length'] = orig_len * traffic_multiplier
            graph_jammed[u][v][key]['traffic_jam'] = True
            graph_jammed[u][v][key]['original_length'] = orig_len
            modified += 1
    print(f"Добавлены пробки на {modified} дорогах (множитель {traffic_multiplier})")
    return graph_jammed, modified

def block_roads(graph, blocked_roads, block_weight=INF):
    """
    Полностью перекрывает дороги, устанавливая очень большую длину (INF).
    Возвращает копию графа и количество заблокированных рёбер.
    """
    graph_blocked = graph.copy()
    blocked_count = 0
    for u, v, key in blocked_roads:
        if graph_blocked.has_edge(u, v, key):
            graph_blocked[u][v][key]['length'] = block_weight
            graph_blocked[u][v][key]['blocked'] = True
            blocked_count += 1
    print(f"Заблокировано {blocked_count} дорог")
    return graph_blocked, blocked_count

# 2. Поиск дорог по типу или названию
def find_important_roads(graph, road_types=None):
    if road_types is None:
        road_types = ['motorway', 'trunk', 'primary']
    important = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        highway = data.get('highway', '')
        if isinstance(highway, list):
            highway = highway[0] if highway else ''
        if highway in road_types:
            important.append((u, v, key))
    return important

def find_roads_by_name(graph, name_patterns):
    matched = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        name = data.get('name', '')
        if isinstance(name, list):
            name = ' '.join(name)
        name_lower = str(name).lower()
        if any(pattern.lower() in name_lower for pattern in name_patterns):
            matched.append((u, v, key))
    return matched

def select_random_roads(roads_list, percentage):
    if not roads_list:
        return []
    n_to_select = max(1, int(len(roads_list) * percentage))
    n_to_select = min(n_to_select, len(roads_list))
    indices = np.random.choice(len(roads_list), size=n_to_select, replace=False)
    return [roads_list[i] for i in indices]

# 3. Визуализация изменённых дорог
def visualize_modified_roads(original_graph, modified_graph, scenario_name,
                             warehouses_df, warehouse_nodes, start_index=0,
                             save_dir=OUTPUT_DIR):
    """
    Отображает дорожную сеть, подсвечивая пробки (оранжевым) и блокировки (красным).
    Наносит склады: начальный (синий), остальные (зелёные).
    Сохраняет карту в указанную папку.
    """
    fig, ax = plt.subplots(figsize=(15, 12))
    ox.plot_graph(original_graph, ax=ax, node_size=0,
                  edge_linewidth=0.5, edge_color='lightgray',
                  bgcolor='white', show=False, close=False)

    jammed = []
    blocked = []
    for u, v, key, data in modified_graph.edges(keys=True, data=True):
        if data.get('traffic_jam', False):
            jammed.append((u, v, key))
        if data.get('blocked', False):
            blocked.append((u, v, key))

    # Рисуем пробки
    for u, v, key in jammed:
        if modified_graph.has_edge(u, v, key):
            edge_data = modified_graph[u][v][key]
            coords = _get_edge_coords(modified_graph, u, v, edge_data)
            if coords:
                x_vals, y_vals = zip(*coords)
                ax.plot(x_vals, y_vals, color='orange', linewidth=3, alpha=0.8)

    # Рисуем блокировки
    for u, v, key in blocked:
        if modified_graph.has_edge(u, v, key):
            edge_data = modified_graph[u][v][key]
            coords = _get_edge_coords(modified_graph, u, v, edge_data)
            if coords:
                x_vals, y_vals = zip(*coords)
                ax.plot(x_vals, y_vals, color='red', linewidth=4, alpha=0.8)

    # Склады
    for idx, row in warehouses_df.iterrows():
        color = 'blue' if idx == start_index else 'green'
        ax.scatter(row['longitude'], row['latitude'],
                   c=color, s=100, edgecolors='black', zorder=5)
        ax.annotate(row['name'], (row['longitude'], row['latitude']),
                    xytext=(5,5), textcoords='offset points', fontsize=9)

    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=2, label='Обычные дороги'),
        Line2D([0], [0], color='orange', lw=3, label='Пробки'),
        Line2D([0], [0], color='red', lw=4, label='Блокировки'),
        Line2D([0], [0], marker='o', color='blue', markersize=8, label='Начальный склад'),
        Line2D([0], [0], marker='o', color='green', markersize=8, label='Остальные склады')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f'Изменения в сценарии: {scenario_name}', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'modified_roads_{scenario_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Карта изменений сохранена: {save_path}")
    return len(jammed), len(blocked)

def _get_edge_coords(graph, u, v, edge_data):
    """Вспомогательная функция для получения координат ребра."""
    if 'geometry' in edge_data:
        return list(edge_data['geometry'].coords)
    else:
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        return [(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])]

# 4. Анализ влияния шума на матрицы расстояний
def analyze_noise_impact(original_graph, scenario_graphs, warehouse_nodes, warehouse_names):
    """
    Сравнивает матрицы расстояний между складами для разных сценариев.
    Выводит статистику и строит график увеличения средней длины пути.
    """
    print("Анализ влияния шума на матрицы расстояний")

    # Вычисляем матрицы для всех сценариев
    matrices = {}
    for name, graph in scenario_graphs.items():
        matrices[name] = _compute_matrix(graph, warehouse_nodes)

    n = len(warehouse_nodes)
    base_matrix = matrices["Без шума"]

    # Сводная таблица
    stats = []
    for name, matrix in matrices.items():
        if name == "Без шума":
            continue
        diff = matrix - base_matrix
        mask = (base_matrix < INF/2)  # учитываем только достижимые пары
        valid_diff = diff[mask]
        mean_increase = np.mean(valid_diff) / 1000
        max_increase = np.max(valid_diff) / 1000
        # Процент пар, ставших недостижимыми
        unreachable_before = np.sum(base_matrix >= INF/2)
        unreachable_after = np.sum(matrix >= INF/2)
        new_unreachable = unreachable_after - unreachable_before
        stats.append({
            'Сценарий': name,
            'Среднее увеличение (км)': mean_increase,
            'Макс. увеличение (км)': max_increase,
            'Новых недостижимых пар': new_unreachable
        })
        print(f"\n{name}:")
        print(f"  Среднее увеличение расстояния: {mean_increase:.2f} км")
        print(f"  Максимальное увеличение: {max_increase:.2f} км")
        print(f"  Новых недостижимых пар: {new_unreachable}")

    # График увеличения среднего расстояния
    scenarios = [s['Сценарий'] for s in stats]
    means = [s['Среднее увеличение (км)'] for s in stats]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scenarios, means, color=['orange', 'red', 'green'])
    ax.set_ylabel('Среднее увеличение расстояния (км)')
    ax.set_title('Влияние шумовых сценариев на среднюю длину пути между складами')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'noise_distance_increase.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nГрафик сохранён: {save_path}")

    # График доли затронутых дорог по типам (для каждого сценария)
    plot_affected_road_types(original_graph, scenario_graphs)

def _compute_matrix(graph, warehouse_nodes):
    """Вычисляет асимметричную матрицу расстояний."""
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

def plot_affected_road_types(original_graph, scenario_graphs):
    """Строит столбчатую диаграмму доли изменённых дорог по типам для каждого сценария."""
    # Собираем информацию о типах дорог в исходном графе
    type_counts = Counter()
    for u, v, key, data in original_graph.edges(keys=True, data=True):
        hw = data.get('highway', 'other')
        if isinstance(hw, list):
            hw = hw[0] if hw else 'other'
        type_counts[hw] += 1
    total_edges = sum(type_counts.values())

    scenarios_data = {}
    for name, graph in scenario_graphs.items():
        if name == "Без шума":
            continue
        affected_counts = Counter()
        for u, v, key, data in graph.edges(keys=True, data=True):
            if data.get('traffic_jam', False) or data.get('blocked', False):
                hw = data.get('highway', 'other')
                if isinstance(hw, list):
                    hw = hw[0] if hw else 'other'
                affected_counts[hw] += 1
        scenarios_data[name] = affected_counts

    # Отбираем топ-5 типов дорог по общему количеству
    top_types = [t for t, _ in type_counts.most_common(5)]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_types))
    width = 0.2
    for i, (name, counts) in enumerate(scenarios_data.items()):
        percentages = [counts.get(t, 0) / type_counts[t] * 100 for t in top_types]
        ax.bar(x + i*width, percentages, width, label=name)
    ax.set_xticks(x + width * (len(scenarios_data)-1)/2)
    ax.set_xticklabels(top_types)
    ax.set_ylabel('Доля изменённых дорог (%)')
    ax.set_title('Доля дорог, затронутых пробками или блокировками, по типам')
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'affected_road_types.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"График типов дорог сохранён: {save_path}")

# 5. Генерация сценариев
def generate_noise_scenarios(original_graph, warehouses_df, warehouse_nodes,
                             start_index=0, random_seed=42, save_visualizations=True):
    """
    Создаёт четыре сценария дорожной обстановки.
    Возвращает словарь {название: граф}.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    important = find_important_roads(original_graph)
    mkad = find_roads_by_name(original_graph, ['МКАД', 'MKAD', 'Московская кольцевая'])
    highways = find_roads_by_name(original_graph, [
        'Ленинградское', 'Ярославское', 'Дмитровское', 'Рижское',
        'Волоколамское', 'Пятницкое', 'Новорижское', 'Киевское',
        'Минское', 'Можайское', 'Калужское', 'Варшавское'
    ])

    print(f"Найдено важных дорог: {len(important)}")
    print(f"Дорог МКАД: {len(mkad)}, основных шоссе: {len(highways)}")

    scenarios = {}
    scenarios["Без шума"] = original_graph

    # Сценарий 2: только пробки
    jam_roads1 = select_random_roads(important, 0.15)
    graph_jam, _ = add_traffic_jams(original_graph, jam_roads1, traffic_multiplier=8)
    if save_visualizations:
        visualize_modified_roads(original_graph, graph_jam, "Пробки на основных дорогах",
                                 warehouses_df, warehouse_nodes, start_index)
    scenarios["Пробки"] = graph_jam

    # Сценарий 3: блокировки + пробки
    jam_roads2 = select_random_roads(important, 0.12)
    blocked_mkad = select_random_roads(mkad, 0.08)
    blocked_highways = select_random_roads(highways, 0.04)
    all_blocked = blocked_mkad + blocked_highways
    graph_blocked, _ = block_roads(original_graph, all_blocked)
    graph_combined, _ = add_traffic_jams(graph_blocked, jam_roads2, traffic_multiplier=8)
    if save_visualizations:
        visualize_modified_roads(original_graph, graph_combined, "Блокировки и пробки",
                                 warehouses_df, warehouse_nodes, start_index)
    scenarios["Блокировки+Пробки"] = graph_combined

    # Сценарий 4: умеренные условия
    jam_roads3 = select_random_roads(important, 0.08)
    blocked_mkad_mod = select_random_roads(mkad, 0.03)
    blocked_highways_mod = select_random_roads(highways, 0.015)
    all_blocked_mod = blocked_mkad_mod + blocked_highways_mod
    graph_mod_blocked, _ = block_roads(original_graph, all_blocked_mod)
    graph_moderate, _ = add_traffic_jams(graph_mod_blocked, jam_roads3, traffic_multiplier=4)
    if save_visualizations:
        visualize_modified_roads(original_graph, graph_moderate, "Умеренные условия",
                                 warehouses_df, warehouse_nodes, start_index)
    scenarios["Умеренные условия"] = graph_moderate

    return scenarios

# 6. Проверка достижимости
def check_warehouses_accessibility(graph, warehouse_nodes, warehouse_names):
    n = len(warehouse_nodes)
    unreachable = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                nx.shortest_path_length(graph, warehouse_nodes[i], warehouse_nodes[j], weight='length')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                unreachable.append((i, j, warehouse_names[i], warehouse_names[j]))
    if unreachable:
        print(f"Найдено {len(unreachable)} недостижимых пар:")
        for i, j, name_i, name_j in unreachable[:5]:
            print(f"  {name_i} -> {name_j}")
        if len(unreachable) > 5:
            print(f"  ... и ещё {len(unreachable)-5} пар")
        return False
    else:
        print("Все склады достижимы.")
        return True
