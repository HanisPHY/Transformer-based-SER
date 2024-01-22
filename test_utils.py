import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

import torch


def plotAttention(attention_weights, labels):
    # Average attention weights for each class
    avg_attention_weights = {}
    label_cnt = defaultdict(int)
    for i, label in enumerate(labels):
        label_cnt[label] += 1
        label_attention = attention_weights[i]
        
        if avg_attention_weights.get(label) is None:
            avg_attention_weights[label] = np.array(np.zeros(label_attention[0][0].shape))
        
        for j in range(len(label_attention)):
            for k in range(len(label_attention[j])):
                print("label_attention[j][k].detach().numpy(): ", j, k, label_attention[j][k].detach().numpy())
                avg_attention_weights[label] += label_attention[j][k].detach().numpy()
                
    for key in label_cnt:
        avg_attention_weights[key] /= label_cnt[key]
        
    # visualize avg_attention_weights for each class
    plt.figure(figsize=(5, 5 * len(avg_attention_weights)))

    cnt = 1
    for key in avg_attention_weights:
        ax = plt.subplot(len(avg_attention_weights), 1, cnt)
        sns.heatmap(avg_attention_weights[key], cmap='Blues', ax=ax)
        ax.set_title(key)
        cnt += 1

    plt.tight_layout()
    plt.show()
        
    return avg_attention_weights


def test_plotAttention():
    # Test case 1
    attention_weights = [
        np.array([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]]),
        np.array([[[[0.9, 1.0], [1.1, 1.2]], [[1.3, 1.4], [1.5, 1.6]]]])
    ]
    print("shape is: ", attention_weights[0].shape)
    attention_weights = [torch.from_numpy(attention_weight) for attention_weight in attention_weights]
    labels = [0, 1]

    ans = plotAttention(attention_weights, labels)
    
    a = np.array([[0.1, 0.2], [0.3, 0.4]]) + np.array([[0.5, 0.6], [0.7, 0.8]])
    b = np.array([[0.9, 1.0], [1.1, 1.2]]) + np.array([[1.3, 1.4], [1.5, 1.6]])
    correct = {0: a, 1: b}
    
    assert np.allclose(ans[0], correct[0])
    assert np.allclose(ans[1], correct[1])
    
def test_plotAttention2():
    import random
    rows = 50
    cols = 50
    random_2d_array_1 = [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]
    random_2d_array_2 = [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]

    attention_weights = [
        np.array([[random_2d_array_1]]),
        np.array([[random_2d_array_2]])
    ]
    print("shape is: ", attention_weights[0].shape)
    attention_weights = [torch.from_numpy(attention_weight) for attention_weight in attention_weights]
    labels = [0, 1]

    ans = plotAttention(attention_weights, labels)
    
test_plotAttention()
test_plotAttention2()