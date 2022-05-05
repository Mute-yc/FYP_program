import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from sklearn.utils.multiclass import type_of_target
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
       # print('**************************************')
       # print(type(data.y))
       #print(type(output))
       #print((output>1).any())
       # print((output<0).any())
        loss = loss_fn(output, data.y.view(-1, 1).float()) #loss function 
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
        
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    loss_accumulate = 0.0
    count = 0.0
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            loss = loss_fn(total_preds, total_labels)

            loss_accumulate += loss
            count += 1

            y_label = total_labels.flatten().tolist()
            y_pred = total_preds.flatten().tolist()
        loss = loss_accumulate / count
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)

        precision = tpr / (tpr + fpr)

        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

        thred_optim = thresholds[5:][np.argmax(f1[5:])]

        print("optimal threshold: " + str(thred_optim))

        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

        auc_k = auc(fpr, tpr)
        print("AUROC:" + str(auc_k))
        print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

        cm1 = confusion_matrix(y_label, y_pred_s)
        print('Confusion Matrix : \n', cm1)
        print('Recall : ', recall_score(y_label, y_pred_s))
        print('Precision : ', precision_score(y_label, y_pred_s))

        total1 = sum(sum(cm1))
        #####from confusion matrix calculate accuracy
        accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
        print('Accuracy : ', accuracy1)

        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        print('Sensitivity : ', sensitivity1)

        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        print('Specificity : ', specificity1)

        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                  outputs), y_pred, loss.item()


    # return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = [['davis','kiba','chembl','bindingdb', 'AGC'][int(sys.argv[1])]] 
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
max_auc = 0
# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_7_noFDADrug_target_unseen_target_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_7_unseen_target_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_7_noFDADrug_target_unseen_target_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_7_unseen_target_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        #m = nn.Sigmod()
        #logits = torch.squeeze(m(score))
        loss_fn = nn.BCEWithLogitsLoss()  # classification function
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            auroc, auprc, f1, logits, loss = predicting(model, device, test_loader)
            if auroc > max_auc:
                # model_max = copy.deepcopy(model)
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,[auroc, auprc, f1, logits, loss])))
                # max_auc = auc
                print('Validation at Epoch ' + str(epoch + 1) + ' , AUROC: ' + str(auroc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))
                best_epoch = epoch+1
                print('auc improved at epoch ', best_epoch)
            #G,P = predicting(model, device, test_loader)
            #ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
            #if ret[1]<best_mse:
            #    torch.save(model.state_dict(), model_file_name)
            #    with open(result_file_name,'w') as f:
            #        f.write(','.join(map(str,ret)))
            #    best_epoch = epoch+1
            #    best_mse = ret[1]
            #    best_ci = ret[-1]
            #    print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
            else:
                print(auroc,'No improvement since epoch ', best_epoch)
