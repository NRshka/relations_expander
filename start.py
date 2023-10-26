from collections import defaultdict
from itertools import product
from typing import Dict, List, Set, Tuple

import graphviz  # type: ignore
import numpy as np
from loguru import logger  # type: ignore
from numba import jit  # type: ignore


@jit(nopython=True)
def find_relations_from_point(
    matrix: np.ndarray,
    seen_data: Set[Tuple[int, int]],
    candidates: List[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    clusters: Set[Tuple[int, int]] = set()
    # Используем итеративный алгоритм вместо рекурсивного, чтобы избежать max recursion depth exceeded
    # спасибо Гвидо за ещё один день без tail recursion optimization!
    while len(candidates):
        cur_row, cur_col = candidates.pop()
        ptr = cur_row, cur_col
        seen_data.add(ptr)
        clusters.add(ptr)

        for col in range(matrix.shape[1]):
            if matrix[cur_row, col] == 1:
                ptr = (cur_row, col)
                if ptr not in seen_data:
                    candidates.append(ptr)
                    seen_data.add(ptr)
                    clusters.add(ptr)

        for row in range(matrix.shape[0]):
            if matrix[row, cur_col] == 1:
                ptr = (row, cur_col)
                if ptr not in seen_data:
                    candidates.append(ptr)
                    seen_data.add(ptr)
                    clusters.add(ptr)

    return clusters


def find_relations(matrix: np.ndarray) -> List[Set[Tuple[int, int]]]:
    seen_data: Set[Tuple[int, int]] = set()
    seen_data.add((-1, -1))  # грязный хак, чтобы numba мог оценить размер айтемов в контейнере
    clusters: List[Set[Tuple[int, int]]] = []
    candidates: List[Tuple[int, int]] = []
    # перебираем все точки, которые мы ещё не посещали и для каждой такой точки ищем "знакомых"
    # всех знакомых записываем в посещённые точки
    # точки, имеющие общих знакомых считаются принадлещими одному классу
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                ptr = (i, j)
                if ptr not in seen_data:
                    candidates.append(ptr)
                    clusters.append(find_relations_from_point(matrix, seen_data, candidates))

    return clusters


def set_ones(matrix: np.ndarray, clusters: List[Set[Tuple[int, int]]]) -> np.ndarray:
    for cluster in clusters:
        for i, j in cluster:
            matrix[i, j] = 1
            matrix[j, i] = 1

    return matrix


def find_negatives(
    matrix: np.ndarray,
    class2cluster: Dict[int, Set[int]],
    points2class: Dict[int, int],
):
    seen_points: Set[Tuple[int, int]] = set()
    all_negative_pairs = set()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == -1 and (i, j) not in seen_points:
                i_class = points2class[i]
                j_class = points2class[j]
                negative_pairs = product(class2cluster[i_class], class2cluster[j_class])
                for neg_i, neg_j in negative_pairs:
                    # TODO добавить сохранение таких индексов для последующего анализа ошибок в разметке
                    if matrix[neg_i, neg_j] == 1:
                        logger.warning('Supposed negative pair has positive label!')
                    if matrix[neg_i, neg_j] == 1:
                        logger.warning('Supposed negative pair has positive label!')

                    matrix[neg_i, neg_j] = -1
                    matrix[neg_j, neg_i] = -1
                    all_negative_pairs.add((i, j))

    return matrix, all_negative_pairs


if __name__ == '__main__':
    arr = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, -1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, -1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
    ])
    name2idx = {str(i + 1): i for i in range(arr.shape[0])}
    idx2name = {v: k for k, v in name2idx.items()}

    logger.info('Finding relations...')
    clusters = find_relations(arr)
    graph = graphviz.Graph(format='png')
    written_edges = set()
    class2cluster = defaultdict(set)
    points2class = {}
    for cls_idx, cluster in enumerate(clusters):
        for i, j in cluster:
            class2cluster[cls_idx].add(i)
            class2cluster[cls_idx].add(j)
            points2class[i] = cls_idx
            points2class[j] = cls_idx

            graph.node(idx2name[i], idx2name[i])
            graph.node(idx2name[j], idx2name[j])

            if i != j and (j, i) not in written_edges:
                graph.edge(idx2name[i], idx2name[j], label='Same')
                written_edges.add((i, j))

    arr = set_ones(arr, clusters)
    logger.info('Finding negatives...')
    arr, negative_pairs = find_negatives(arr, class2cluster, points2class)
    for i, j in negative_pairs:
        if i != j and (j, i) not in written_edges:
            graph.edge(idx2name[i], idx2name[j], label='Different')
            written_edges.add((i, j))

    logger.info('Rendering graph...')
    graph.render(directory='graph')

    logger.info('Done.')
