from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json

from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


def data_distribution(df, classes_names):
    classes = np.unique(df['label'].values) # class
    values = [df['label'].values.tolist().count(class_) for class_ in classes] # frequency
    plt.figure( figsize=(10 , 6)  , dpi=100 )
    plt.bar(classes , values)
    plt.xticks(classes, classes_names , size=10)
    plt.xlabel('Class', size=12)
    plt.ylabel('Frequency', size=12)
    plt.title('Class Distribution of Dataset', size=13)
    plt.show()


def plot_training(loss_list, metric_list, title):
    # %matplotlib inline
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5) )
    fig.subplots_adjust(wspace=.2)
    plotLoss(ax1, np.array(loss_list), title)
    plotAccuracy(ax2, np.array(metric_list), title)
    plt.show()


def plotLoss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label="Train_loss")
    ax.plot(loss_list[:, 1], label="Validation_loss")
    ax.set_title("Loss Curves - " + title, fontsize=12)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})
    ax.grid()


def plotAccuracy(ax, metric_list, title):
    ax.plot(metric_list[:], label="validation_Accuracy")
    ax.set_title("Accuracy Curve - " + title, fontsize=12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})
    ax.grid()


def report(labels, preds, encoder):
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    # Decode labels (ids to class name)
    preds = encoder.inverse_transform(preds)
    labels = encoder.inverse_transform(labels)
    # Calculate accuracy for each class
    class_accuracies = []
    for class_ in encoder.classes_:
        class_acc = np.mean(preds[labels == class_] == class_)
        class_accuracies.append(class_acc)

    print( list(zip(encoder.classes_,class_accuracies)))
    print(classification_report(labels, preds, labels = encoder.classes_))
    plot_cnf_matrix(cm , encoder.classes_)

def padZero(attention_weights):
    # attention_weights has 6 dimensions: (num_iter_dataloader, num_layers, batch_size, num_heads, sequence_length, sequence_length)
    max_seq_length = 0
    
    for iter_dataloader in attention_weights:
        for iter, layer in enumerate(iter_dataloader):
            # Use attention weights of the final layer
            if iter == len(iter_dataloader) - 1:
                seq_length = layer.shape[-1]
                if seq_length > max_seq_length:
                    max_seq_length = seq_length
                
    padded_attention_weights = []
    print("padZero function")
    for iter_dataloader in attention_weights:
        padded_att_lst = []
        for iter, layer in enumerate(iter_dataloader):
            print(iter, len(iter_dataloader) - 1)
            if iter == len(iter_dataloader) - 1:
                padded_batch_attention = []
                for batch in layer:
                    padded_head_attention = []
                    # enumerate over heads. Each head has an attention map
                    for head in batch:
                        seq_length = head.shape[-1]
                        padded_attention = np.zeros((max_seq_length, max_seq_length))
                        padded_attention[:seq_length, :seq_length] = head
                        padded_head_attention.append(padded_attention)
                    padded_batch_attention.append(padded_head_attention)
                padded_att_lst.append(padded_batch_attention)
        padded_attention_weights.append(padded_att_lst)
        
    zeroPaddedAttention = np.array(padded_attention_weights)
    print("Saving array -------------------")
    saveArray(zeroPaddedAttention, "zeroPaddedAttention.npy")
    return zeroPaddedAttention
    
# attention_weights: list of attention weights for each sa0mple

# attentions (tuple(torch.FloatTensor):  each item in the attention_weights
# Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
def plotAttention(attention_weights, test_labels, encoder):
    labels = encoder.inverse_transform(test_labels)
    print("len(labels) are: ", len(labels))
    
    paddedAttentionWeights = padZero(attention_weights)
    print(paddedAttentionWeights.shape)
    flattendAttentionWeights = arrayTo2D(paddedAttentionWeights)
    
    # # t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # # X_tsne = tsne.fit_transform(np.concatenate(zeroPadded_attention_weights, axis=0).reshape(-1, zeroPadded_attention_weights[0][0].shape[-1]**2))
    # X_tsne = tsne.fit_transform(padded_attention_weights)
    # print(tsne.kl_divergence_)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(flattendAttentionWeights)
    
    # df = pd.DataFrame(data = X_pca, 
    #               columns = ['PC1', 'PC2'])
    # print("dataframe of X_pca is: ", df)

    # target = pd.Series(labels, name='target')
    # result_df = pd.concat([df, target], axis=1)
    # print("result_df is: ", result_df)
    
    # print("labels are: ", labels)
    
    # colors = ['r', 'g', 'b', 'y']
    # plt.figure(figsize=(8, 6))
    # for target, color in zip(target, colors):
    #     indicesToKeep = result_df['target'] == target
    #     plt.scatter(result_df.loc[indicesToKeep, 'PC1'], 
    #            result_df.loc[indicesToKeep, 'PC2'], 
    #            c = color, 
    #            s = 50)
    # plt.legend(labels)
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)  # alpha for transparency
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Padded Attention Weights')
    plt.grid(True)
    plt.show()
    
    # visualize PCA
    # draw the scatter plot and use a different color for each class
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='PC1', y='PC2', hue='target', data=result_df, palette='viridis')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA of Padded Attention Weights')
    # plt.grid(True)
    # plt.show()
    
    
    
    # avg_attention_weights = {label: sum_attention / count_attention_maps[label] for label, sum_attention in sum_attention_maps.items()}
    
    # # visualize avg_attention_weights for each class
    # print("-" * 50, "Average attention heat map")
    # nrows = 4
    # ncols = 1
    # fig_width = 6
    # fig_height = nrows * 6
    
    # fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    # for ax, (key, data) in zip(axs, avg_attention_weights.items()):
    #     sns.heatmap(data, cmap='Blues', ax=ax)
    #     ax.set_title(key)

    # plt.tight_layout()
    # plt.show()

def plot_cnf_matrix(cm , classes):
    print("confusion matrix:")
    print(cm)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm_display.plot()
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    cm_df = pd.DataFrame(cm,classes,classes)                      
    plt.figure(figsize=(10,10))  
    sns.heatmap(cm_df , annot=True , cmap='Blues', fmt='g')
    plt.show()

# helping method
def saveArray(array_to_save, file_name):
    np.save(file_name, array_to_save)
    print(f"Array saved to {file_name}")
    
def loadArray(file_name):
    print(f"Array loaded from {file_name}")
    return np.load(file_name)

def saveList(list_to_save, file_name):
    with open('list.json', 'w') as file:
        json.dump(list_to_save, file)
    print(f"List saved to {file_name}")
    
def loadList(file_name):
    print(f"List loaded from {file_name}")
    with open(file_name, 'r') as file:
        return json.load(file)
    
def arrayTo2D(array):
    size = 1
    for dim in array.shape[1:]:
        size *= dim
    flattened_maps = array.reshape(-1, size)

    print("flattened_maps.shape: ", flattened_maps.shape)
    return flattened_maps
