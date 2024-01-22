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
    # Average attention weights for each class
    avg_attention_weights = {}
    label_cnt = defaultdict(int)
    labels = encoder.inverse_transform(labels)
    for i, label in enumerate(labels):
        label_cnt[label] += 1
        label_attention = attention_weights[i]
        
        if avg_attention_weights.get(label) is None:
            avg_attention_weights[label] = np.array(np.zeros(label_attention[0][0].shape))
        
        for j in range(len(label_attention)):
            for k in range(len(label_attention[j])):
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
