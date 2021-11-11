import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error 

class Result():
    def __init__(self):
        pass

    def rebuild_loss_pearson(self, path, epoch_num):
        epoch_loss_list = []
        epoch_pearson_list = []
        min_train_loss = 100
        min_train_id = 0
        for i in range(1, epoch_num + 1):
            train_df = pd.read_csv(path + '/TrainingPred_' + str(i) + '.txt', delimiter=',')
            score_list = list(train_df['Score'])
            pred_list = list(train_df['Pred Score'])
            epoch_loss = mean_squared_error(score_list, pred_list)
            epoch_loss_list.append(epoch_loss)
            epoch_pearson = train_df.corr(method = 'pearson')
            epoch_pearson_list.append(epoch_pearson['Pred Score'][0])
            if epoch_loss < min_train_loss:
                min_train_loss = epoch_loss
                min_train_id = i
        print('-------------BEST MODEL ID:' + str(min_train_id) + '-------------')
        print('BEST MODEL TRAIN LOSS: ', min_train_loss)
        print('BEST MODEL PEARSON CORR: ', epoch_pearson_list[min_train_id - 1])
        # print('\n-------------EPOCH TRAINING PEARSON CORRELATION LIST: -------------')
        # print(epoch_pearson_list)
        # print('\n-------------EPOCH TRAINING MSE LOSS LIST: -------------')
        # print(epoch_loss_list)
        epoch_pearson_array = np.array(epoch_pearson_list)
        epoch_loss_array = np.array(epoch_loss_list)
        np.save(path + '/pearson.npy', epoch_pearson_array)
        np.save(path + '/loss.npy', epoch_loss_array)
        return min_train_id

    def plot_loss_pearson(self, path, epoch_num):
        epoch_pearson_array = np.load(path + '/pearson.npy')
        epoch_loss_array = np.load(path + '/loss.npy')
        x = range(1, epoch_num + 1)
        plt.figure(1)
        plt.title('Training Loss and Pearson Correlation in ' + str(epoch_num) + ' Epochs') 
        plt.xlabel('Train Epochs') 
        plt.figure(1)
        plt.subplot(211)
        plt.plot(x, epoch_loss_array) 
        plt.subplot(212)
        plt.plot(x, epoch_pearson_array)
        plt.show()

    def plot_train_real_pred(self, path, best_model_num, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        pred_dl_input_df = pd.read_csv(path + '/TrainingPred_' + best_model_num + '.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Training Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = '.', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train'
        path = './datainfo/plot/%s' % (file_name) + '.png'
        unit = 1
        if os.path.exists('./datainfo/plot') == False:
            os.mkdir('./datainfo/plot')
        while os.path.exists(path):
            path = './datainfo/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
        
    def plot_test_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        pred_dl_input_df = pd.read_csv(path + '/TestPred.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Test Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score', color='green',
                    style = '.', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test'
        path = './datainfo/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = './datainfo/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)



path = './datainfo/result/epoch_50'
path = './datainfo/result/epoch_50'
epoch_num = 50
min_train_id = Result().rebuild_loss_pearson(path, epoch_num)
Result().plot_loss_pearson(path, epoch_num)

epoch_time = '50'
best_model_num = str(min_train_id)
Result().plot_train_real_pred(path, best_model_num, epoch_time)
Result().plot_test_real_pred(path, epoch_time)