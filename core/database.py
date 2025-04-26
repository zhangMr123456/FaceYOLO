# FAISS索引构建示例
import json
import os
from typing import List, Dict

import faiss
import numpy as np
from faiss import IndexIDMap

from settings import INDEX_PATH, DATAMETA_PATH


def save_faiss_index(index: IndexIDMap, filename=INDEX_PATH):
    faiss.write_index(index, filename)
    print(f"已保存FAISS索引到 {filename}")


def load_faiss_index(filename=INDEX_PATH, new_file=False) -> IndexIDMap:
    if not os.path.exists(filename) or new_file:
        index = faiss.IndexFlatIP(512)
        index_with_ids = faiss.IndexIDMap(index)
        print(f"{filename} 执行初始化 IndexFlatIP + IndexIDMap")
    else:
        index_with_ids = faiss.read_index(filename)
        print(f"已从 {filename} 加载FAISS索引")
    return index_with_ids


def get_exist_keys(datameta_path=DATAMETA_PATH) -> List[str]:
    if not os.path.exists(datameta_path):
        return []
    else:
        with open(datameta_path, 'r') as file:
            datameta_dict = json.load(file)
        return list(datameta_dict.keys())


def write_embedding(emb_dict: Dict[str, List[np.ndarray]], datameta_path=DATAMETA_PATH):
    if not os.path.exists(datameta_path):
        datameta_dict = {}
    else:
        with open(datameta_path, 'r') as file:
            datameta_dict = json.load(file)

    max_index = max((int(key) for key in datameta_dict.keys()), default=0)
    file_paths = list(datameta_dict.values())

    index = load_faiss_index()

    for file_path, embs in emb_dict.items():
        if file_path in file_paths:
            print(f"file_path: {file_path} exists database, continue")
            continue
        else:
            file_paths.append(file_path)

        for emb in embs:
            index.add_with_ids(emb.reshape(1, -1), max_index + 1)
            datameta_dict[max_index + 1] = file_path
            max_index += 1

    with open(datameta_path, 'w') as file:
        json.dump(datameta_dict, file)

    save_faiss_index(index)


def query_embedding(embs, k=10, datameta_path=DATAMETA_PATH):
    if embs is None:
        return []

    if not os.path.exists(datameta_path):
        datameta_dict = {}
    else:
        with open(datameta_path, 'r') as file:
            datameta_dict = json.load(file)

    index = load_faiss_index()
    file_paths = []
    for emb in embs:
        distances, indices = index.search(emb.reshape(1, -1), k=k)
        file_paths.extend([(score, datameta_dict[str(int(idx))])
                           for idx, score in zip(indices[0], distances[0]) if idx > 0])
    return file_paths
