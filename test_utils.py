import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch

def plotAttention(attention_weights, labels):
    # Maximum attention map size            
    max_seq_length = max([att.shape[2] for batch in attention_weights for att in batch])

    sum_attention_maps = defaultdict(lambda: np.zeros((max_seq_length, max_seq_length)))
    count_attention_maps = defaultdict(int)

    # Each head has an attention map
    for i, (label, batch_attention) in enumerate(zip(labels, attention_weights)):
        for attention_map in batch_attention:
            for head_attention in attention_map:
                seq_length = head_attention.shape[-1]
                
                padded_attention = np.zeros((max_seq_length, max_seq_length))
                padded_attention[:seq_length, :seq_length] = head_attention.cpu().detach().numpy()
                
                sum_attention_maps[label] += padded_attention
                count_attention_maps[label] += 1

    avg_attention_weights = {label: sum_attention / count_attention_maps[label] for label, sum_attention in sum_attention_maps.items()}
    
    # visualize avg_attention_weights for each class
    plt.figure(figsize=(5,5))  

    cnt = 1
    for key in avg_attention_weights:
        ax = plt.subplot(len(avg_attention_weights), 1, cnt)
        sns.heatmap(avg_attention_weights[key], cmap='Blues', ax=ax)
        ax.set_title(key)
        cnt += 1

    plt.show()
    
def plotAttention_pca(attention_weights):
    zeroPadded_attention_weights = padZero(attention_weights)

    pca = PCA(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(np.concatenate(zeroPadded_attention_weights, axis=0).reshape(-1, zeroPadded_attention_weights[0][0].shape[-1]**2))
    X_pca = pca.fit_transform(zeroPadded_attention_weights)
    # print(tsne.kl_divergence_)
    
def plotAttention_tsne(attention_weights):
    zeroPadded_attention_weights = padZero(attention_weights)

    tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(np.concatenate(zeroPadded_attention_weights, axis=0).reshape(-1, zeroPadded_attention_weights[0][0].shape[-1]**2))
    X_tsne = tsne.fit_transform(zeroPadded_attention_weights)
    print(tsne.kl_divergence_)
    
def test_plotAttention_pca_1():
    attention_weights = [
        np.array([[[[0.1], [0.3]], [[0.5], [0.7]]]]),
        np.array([[[[0.9, 1.0, 1.1], [1.1, 1.2, 1.3], [0.9, 1.0, 1.1]], [[1.3, 1.4, 1.5], [1.5, 1.6, 1.7], [0.9, 1.0, 1.1]]]])
    ]
    
    attention_weights = [torch.from_numpy(attention_weight) for attention_weight in attention_weights]
    expected_output = [
        np.array([[[0.1, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[0.5, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        np.array([[[0.9, 1.0, 0.0, 0.0], [1.1, 1.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[1.3, 1.4, 0.0, 0.0], [1.5, 1.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    ]
    plotAttention_pca(attention_weights)    

def padZero(attention_weights):
    max_seq_length = max(head.shape[2] for batch in attention_weights for head in batch)
    padded_attention_weights = []
    for att_lst in attention_weights:
        padded_att_lst = []
        for batch_attention in att_lst:
            padded_batch_attention = []
            for head_attention in batch_attention:
                padded_head_attention = []
                for att in head_attention:
                    seq_length = att.shape[-1]
                    padded_attention = np.zeros((max_seq_length, max_seq_length))
                    padded_attention[:seq_length, :seq_length] = att
                    padded_head_attention.append(np.array(padded_attention))
                padded_batch_attention.append(np.array(padded_head_attention))
            padded_att_lst.append(np.array(padded_batch_attention))
        padded_attention_weights.append(np.array(padded_att_lst))
        
    print("padded_attention_weights: ", padded_attention_weights)
    return np.array(padded_attention_weights)

def test_plotAttention_tsne_1():
    attention_weights = [
        np.array([[[[0.1], [0.3]], [[0.5], [0.7]]]]),
        np.array([[[[0.9, 1.0, 1.1], [1.1, 1.2, 1.3], [0.9, 1.0, 1.1]], [[1.3, 1.4, 1.5], [1.5, 1.6, 1.7], [0.9, 1.0, 1.1]]]])
    ]
    
    attention_weights = [torch.from_numpy(attention_weight) for attention_weight in attention_weights]
    expected_output = [
        np.array([[[0.1, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[0.5, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        np.array([[[0.9, 1.0, 0.0, 0.0], [1.1, 1.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[1.3, 1.4, 0.0, 0.0], [1.5, 1.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    ]
    plotAttention_tsne(attention_weights)
    
def test_plotAttention_tsne_2():
    attention_weights = [
        np.array([[[[0.1]], [[0.5]]]]),
        np.array([[[[0.9, 1.0, 1.1], [1.1, 1.2, 1.3], [0.9, 1.0, 1.1]], [[1.3, 1.4, 1.5], [1.5, 1.6, 1.7], [0.9, 1.0, 1.1]]]])
    ]
    
    attention_weights = [torch.from_numpy(attention_weight) for attention_weight in attention_weights]
    expected_output = [
        np.array([[[0.1, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[0.5, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        np.array([[[0.9, 1.0, 0.0, 0.0], [1.1, 1.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[1.3, 1.4, 0.0, 0.0], [1.5, 1.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    ]
    padZero(attention_weights)
    
def test_padZero_1():
    attention_weights = [
        np.array([[[[0.1], [0.3]], [[0.5], [0.7]]]]),
        np.array([[[[0.9, 1.0, 1.1], [1.1, 1.2, 1.3], [0.9, 1.0, 1.1]], [[1.3, 1.4, 1.5], [1.5, 1.6, 1.7], [0.9, 1.0, 1.1]]]])
    ]
    
    attention_weights = [torch.from_numpy(attention_weight) for attention_weight in attention_weights]
    expected_output = [
        np.array([[[0.1, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[0.5, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        np.array([[[0.9, 1.0, 0.0, 0.0], [1.1, 1.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
                    [[1.3, 1.4, 0.0, 0.0], [1.5, 1.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    ]
    padZero(attention_weights)
    # assert np.array_equal(padZero(attention_weights), expected_output)

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
    
# test_plotAttention()
# test_plotAttention2()

# test_padZero()

# test_plotAttention_tsne_1()
# test_plotAttention_tsne_2()

test_plotAttention_pca_1()