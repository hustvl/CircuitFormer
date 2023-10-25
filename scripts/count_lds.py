import numpy as np
from tqdm import tqdm
import json
def get_bin_idx(x):
    return min(int(x * np.float32(1000)), 1000)

if __name__ == "__main__":
    label_root = 'dataset/CircuitNet/train_congestion/congestion/label/'
    label_list = []
    f = open("data/train.txt", "r")
    line = f.readline()
    while line:
        name = line.split(',')[0].split('/')[-1]
        label_list.append(name)
        line = f.readline()
    f.close()
    max_value = 1000
    value_dict = {x: 0 for x in range(max_value)}
    for i in tqdm(range(len(label_list))):
        label_path = label_root + label_list[i]
        label = np.load(label_path).flatten()
        for j in range(label.shape[0]):
            value_dict[min(get_bin_idx(label[j]),max_value-1)] += 1

    with open('data/lds.txt', 'w', encoding="utf-8") as file:
        alldata = list(value_dict.values())
        file.write(json.dumps(alldata, ensure_ascii=False))
        file.close()



