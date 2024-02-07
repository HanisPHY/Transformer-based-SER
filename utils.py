from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
import numpy as np
import pandas as pd

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

    
# attention_weights: list of attention weights for each sample

# attentions (tuple(torch.FloatTensor):  each item in the attention_weights
# Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
def plotAttention(attention_weights, labels, encoder):
    labels = encoder.inverse_transform(labels)
    # Maximum attention map size
    max_seq_length = max(att.shape[2] for batch in attention_weights for att in batch)

    sum_attention_maps = defaultdict(lambda: np.zeros((max_seq_length, max_seq_length)))
    count_attention_maps = defaultdict(lambda: np.zeros((max_seq_length, max_seq_length)))

    # (batch_size, num_heads, sequence_length, sequence_length)
    for i, (label, attentions) in enumerate(zip(labels, attention_weights)):
        for batch_attention in attentions:
            for head_attention in batch_attention:
                for att in head_attention:
                    seq_length = att.shape[-1]
                    
                    padded_attention = np.zeros((max_seq_length, max_seq_length))
                    padded_attention[:seq_length, :seq_length] = att
                    
                    non_padding_entries = np.ones((max_seq_length, max_seq_length))
                    
                    sum_attention_maps[label] += padded_attention
                    count_attention_maps[label] += non_padding_entries

    avg_attention_weights = {label: sum_attention / count_attention_maps[label] for label, sum_attention in sum_attention_maps.items()}
    
    # visualize avg_attention_weights for each class
    print("-" * 50, "Average attention heat map")
    nrows = 4
    ncols = 1
    fig_width = 6
    fig_height = nrows * 6
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    for ax, (key, data) in zip(axs, avg_attention_weights.items()):
        sns.heatmap(data, cmap='Blues', ax=ax)
        ax.set_title(key)

    plt.tight_layout()
    plt.show()

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
