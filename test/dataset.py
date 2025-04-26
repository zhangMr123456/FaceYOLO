import numpy as np
from core.database import load_faiss_index, save_faiss_index

test_index = load_faiss_index()

for i in range(10000, 100000):
    vector = np.random.random((1, 512)).astype('float32')
    test_index.add_with_ids(vector, i)  # 添加向量和 ID

vector = np.random.random((1, 512)).astype('float32')
distances, indices = test_index.search(vector, k=3)
print("distances: ", distances)
print("indices: ", indices)

save_faiss_index(test_index)
