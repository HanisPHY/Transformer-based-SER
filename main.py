
from transformers import Wav2Vec2FeatureExtractor
from model import get_model
from dataset import load_data , split_data , get_data_loaders
import torch
from torch import nn
import numpy as np
import random
from config import *
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from utils import *


def get_classes_weight(labels):
    classes_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                       classes=np.unique(labels),
                                                       y=np.array(labels))
    return torch.tensor(classes_weight, dtype=torch.float)


def collect(outputs, labels, predictions, true_labels, attention = False, attention_weights = [], output_atten_weight = []):
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    if len(predictions) == 0:
        predictions = preds
        true_labels = labels
        if attention:
            attention_weights = output_atten_weight.cpu().numpy()
    else:
        predictions = np.concatenate((predictions, preds))
        true_labels = np.concatenate((true_labels, labels))
        
        if attention:
            attention_weights = np.concatenate((attention_weights, output_atten_weight.cpu().numpy()))

    return predictions, true_labels, attention_weights


def train(model, dataloader, optimizer, criterion, epoch, device):
    # put the model on train mode
    model.train()
    losses, predictions, true_labels = [], [], []

    for iter, (inputs, labels) in enumerate(dataloader):


        inputs = inputs.to(device)
        print("Train: The shape of the raw input is: ", inputs.shape)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect predictions and true labels
        predictions, true_labels, _ = collect(outputs, labels, predictions, true_labels, attention=False)

        if iter % round((len(dataloader) / 5)) == 0:
            print(f'\r[Epoch][Batch] = [{epoch + 1}][{iter}] -> Loss = {np.mean(losses):.4f} ')

    return np.mean(losses), accuracy_score(true_labels, predictions), predictions , true_labels


def evaluate(model, dataloader, criterion, device, output_attention=False):
    # put the model on evaluation mode
    model.eval()
    losses, predictions, true_labels, attention_weights = [], [], [], []

    for iter, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        print("Evaluation: The shape of the raw input is: ", inputs.shape)
        labels = labels.to(device)

        outputs, atten_weight = model(inputs, output_attention=output_attention)
        
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Collect predictions and true labels
        predictions, true_labels, attention_weights = collect(outputs, labels, predictions, true_labels, attention=True, attention_weights=attention_weights, output_atten_weight=atten_weight)

    return np.mean(losses), accuracy_score(true_labels, predictions) , predictions , true_labels, attention_weights



def trainModel(data_path, check_point, lr, epocks, weight_decay, sch_gamma, sch_step,
               title='', train_bs=2  , plot_data_dist=False):
    

    # load data
    df, label_encoder = load_data(data_path)
    print(df)
    
    class_distribution = df['label'].value_counts()
    print("Class distribution in the full dataset:")
    print(class_distribution)

    num_classes = len(label_encoder.classes_)
    print('Data loaded successfully') ; print('-' * 50)

    # Plot data distribution
    if plot_data_dist : data_distribution(df, label_encoder.classes_)

    # Split data to train, validation, and test sets
    train_val_data, test_data = split_data(df, stratify = df['label'])
    print('-' * 50)
    class_distribution = train_val_data['label'].value_counts()
    print("Class distribution in the train_val_data:")
    print(class_distribution)

    class_distribution = test_data['label'].value_counts()
    print("Class distribution in the test_data:")
    print(class_distribution)
    print('-' * 50)
    train_data, val_data = split_data(train_val_data, stratify = train_val_data['label'])
    
    class_distribution = train_data['label'].value_counts()
    print("Class distribution in the train_data:")
    print(class_distribution)
    
    class_distribution = val_data['label'].value_counts()
    print("Class distribution in the val_data:")
    print(class_distribution)
    print('-' * 50)
    
    print('Number of train samples =', len(train_data) )
    print('Number of validation samples =', len(val_data) )
    print('Number of test samples =', len(test_data) ) ; print('-' * 50)

    print('Loading FeatureExtractor ...', end ='')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(check_point)
    # print out the input shape of the model (the extracted features, the output of Wav2Vec2FeatureExtractor)
    print('Input shape of the model =', feature_extractor.sampling_rate, 'samples') ; print('-' * 50)
    print('\rFeatureExtractor loaded successfully') ; print('-' * 50)

    # Create data loaders
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(train_data , val_data, test_data, train_bs, feature_extractor)
    print('Number of train batches =', len(train_dataloader))
    print('Number of validaion batches =', len(val_dataloader) ) ; print('-' * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'is available' ) ; print('-' * 50)

    print('Loading model ...', end ='')
    model = get_model(check_point, num_classes, device)
    print('\rModel loaded successfully') ; print('-' * 50)

    # Determine the type of : optimizer, scheduling and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    # fc_weights = get_classes_weight(train_data.label.values).to(device)
    # criterion = nn.CrossEntropyLoss(weight=fc_weights)
    criterion = nn.CrossEntropyLoss()

    print('Start Training ....',  end ='' )
    best_acc = 0 ; loss_list, acc_list = [], []
    for epock in range(epocks):
        train_loss, trian_acc , _ , _, inputWeights = train(model, train_dataloader, optimizer, criterion, epock, device)
        val_loss , val_acc , _ , _, _ = evaluate(model, val_dataloader, criterion, device)
        # scheduler.step()
        loss_list.append([train_loss, val_loss])
        acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best-model.pt')
        # print(f'\tTrain -> Loss = {train_loss:.4f} /  accuracy = {trian_acc:.4f}')
        # print(f'\tValidation -> Loss = {val_loss:.4f} /  accuracy = {val_acc:.4f}')
        plot_training(np.array(loss_list), np.array(acc_list), title)

    model = get_model(check_point, num_classes, device)
    model.load_state_dict(torch.load('best-model.pt'))
    test_loss, test_acc, test_preds, test_labels, attention_weights = evaluate(model, test_dataloader , criterion, device, output_attention=True)
    print('-' * 30, '\nBest model on test set -> Loss =', test_loss, f'Accuracy = {test_acc * 100:.2f} %')
    
    plotAttention(attention_weights, test_labels, label_encoder)
    report(test_labels, test_preds, label_encoder)



if __name__ == '__main__':
    print("Main program")
    random_seed=3 
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    trainModel(DATASET_PATH, HUBERT, LR, EPOCHS, WEIGHT_DECAY, SCH_GAMMA, SCH_STEP)
