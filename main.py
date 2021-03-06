from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

from download_stock_data import download_and_save_stock_data
from create_features import get_random_forest_features, get_nn_features, indexize
from batch_generator import BatchGenerator
from rnn import RNN

###############################################
# Globals
###############################################

train_dir = '/media/sander/samsungssd/tradosaurus/train_data/'
test_dir = '/media/sander/samsungssd/tradosaurus/test_data/'

n_history = 100
n_future = 1
offset = 100
price_pred = True

    
###############################################
# Download and save stock data
###############################################

#download_and_save_stock_data()


###############################################
# Load RNN features and train RNN
###############################################

get_nn_features(n_history, n_future, offset, train_dir, test_dir, price_pred)

batch_size = 32
generator = BatchGenerator(train_dir, test_dir, batch_size, n_history, n_future)


summary_frequency=100
num_nodes=64
num_layers=5
num_unrollings = n_history-n_future
batch_generator=generator
input_shape = 3
only_retrain_output=False
output_keep_prob = 1

cell=tf.nn.rnn_cell.GRUCell

nn = RNN(summary_frequency, num_nodes, num_layers, num_unrollings, n_future,
             batch_generator, input_shape, only_retrain_output, output_keep_prob,
             cell)
#nn.load('models/checkpoint_1dag.ckpt') 
nn.train(20000)
#nn.save('models/checkpoint_1dag.ckpt')
#nn.save('models/checkpoint_5dagen.ckpt')
nn.plot_loss()

###############################################
# Eval RNN
###############################################

# Scatter predicted vs actual price change for batch of stocks
generator = BatchGenerator(train_dir, test_dir, 1000, n_history, n_future)
x_batch, y_batch = generator.next_batch('test')

preds = []
labels = []
for i in range(1000):
    x, y = x_batch[i], y_batch[i]
    y = y.reshape([n_history-n_future*2, 1]) 
    x = x.reshape([1, 3, n_history-n_future*2]) 
    #x[:,0,:]=0.
    #x[:,1,:]=0.
    #x[:,2,:]=0.
    preds.extend(nn.predict(x))
    labels.extend(y)
plt.scatter(preds, labels, alpha=.4)
plt.legend(['Predicted vs actual stock price change'])

# Determine correlation
preds = np.array(preds).reshape([len(preds)])
labels = np.array(labels).reshape([len(labels)])
pearsonr(preds, labels)

# Determine mean difference beteween high low and normal cats
np.mean(labels[np.where(preds>3)])
np.mean(labels[np.where(preds<-3)])
np.mean(labels[np.where(abs(preds)<3)])

# Determine acc
score=np.zeros(len(labels))
score[preds>0]+=.5
score[labels>0]+=.5
acc = len(score[score!=.5])/float(len(score))
acc


# Plot predicted vs actual price change for a single stocks
x_batch, y_batch = generator.next_batch('test')
x, y = x_batch[0], y_batch[0]
y = y.reshape([n_history-n_future*2, 1])
x = x.reshape([1, 3, n_history-n_future*2])
preds = nn.predict(x)[:,0]


#plt.plot(x[0,0,:], color='g', alpha=.4)
plt.plot([0]*(n_history-n_future), color='k', alpha=.4)
plt.plot(y, color='r', alpha=.4)
plt.plot(preds, color='b', alpha=.4)
plt.legend(['Zero-line', 'Actual price change', 'Predicted price change'])

############################################################
# Prediction pipeline
############################################################

# Get last n_history days for each stock as list from stocks.csv
stocks = pd.read_csv('data/stocks.csv')

features = []
preds = []
aandelen = []
price_hist_list = []

for aandeel in tqdm(stocks.Aandeel.unique().tolist()):
    
    temp_stocks = stocks[stocks.Aandeel==aandeel]
    temp_stocks.sort_values('Date', inplace=True)
    temp_stocks = temp_stocks[-n_history:]
    if len(temp_stocks) == n_history:
                        
        price_history_normal = np.array(temp_stocks.Close)
        price_history = indexize(np.array(temp_stocks.Close))
        volume_history = indexize(np.array(temp_stocks.Volume))
        exchange_history = indexize(np.array(temp_stocks.exchange_total))
        
        x = np.concatenate([price_history, volume_history, exchange_history], axis=0).reshape([3, n_history])
        
        x_next = x[:,n_future:]                
        
        x_diff = x_next-x[:,:-n_future]
                                       
        x_diff = x_diff.reshape([1, 3, n_history-n_future])
        
        pred = nn.predict(x_diff).squeeze()
        
        preds.append(pred[-1])
        aandelen.append(aandeel)
        price_hist_list.append(price_history_normal) 
        features.append(x)
    else:
        continue
    
preds = np.array(preds)
features = np.array(features)
price_hist_list = np.array(price_hist_list).squeeze()

len(np.where(preds>.5)[0])
len(np.where(preds<-.8)[0])

prices_good = np.mean(price_hist_list[np.where(preds>.5)[0]], axis=0)
prices_bad = np.mean(price_hist_list[np.where(preds<-.8)[0]], axis=0)

plt.plot(prices_good, color='g')
plt.plot(prices_bad, color='r')

volumes_good = np.mean(features[:,1,:][np.where(preds>.5)[0]], axis=0)
volumes_bad = np.mean(features[:,1,:][np.where(preds<-.8)[0]], axis=0)

plt.plot(volumes_good, color='g')
plt.plot(volumes_bad, color='r')

exchanges_good = np.mean(features[:,2,:][np.where(preds>.5)[0]], axis=0)
exchanges_bad = np.mean(features[:, 2,:][np.where(preds<-.8)[0]], axis=0)
  
plt.plot(exchanges_good, color='g')
plt.plot(exchanges_bad, color='r')

preds_zipped = zip(aandelen, preds)
preds_zipped.sort(key = lambda t: t[1])

print 'worst:'
for i,j in preds_zipped[:10]:
    print i,j

print 'best:'    
for i,j in preds_zipped[-12:]:
    print i,j
    
    


###############################################
# Load random forest features
###############################################

features_train, labels_train, price_history_train, features_test, labels_test, price_history_test=\
get_random_forest_features(n_future, n_history, offset)


###############################################
# Fit random forest
###############################################

model = RandomForestRegressor()
model.fit(features_train, labels_train)
model.score(features_train, labels_train)
model.score(features_test, labels_test)
plt.scatter(model.predict(features_train), labels_train, alpha=.4)
plt.scatter(model.predict(features_test), labels_test, alpha=.4)
importance=model.feature_importances_


temp = np.concatenate([features_test, labels_test.reshape([len(labels_test),1])], axis=1)
temp_df = pd.DataFrame(temp)
col_list = ['col'+str(n) for n in range(len(temp_df.columns))]
col_list[-1]='profit'
temp_df.columns = col_list

temp_df['pred'] = model.predict(features_test)

preds_good=np.where(temp_df['pred']>.2)[0]
preds_bad=np.where(temp_df['pred']<-.2)[0]
preds_norm = np.where(abs(temp_df['pred'])<.2)[0]

prices_good = np.mean(price_history_test[preds_good], axis=0)
prices_bad = np.mean(price_history_test[preds_bad], axis=0)
prices_norm = np.mean(price_history_test[preds_norm], axis=0)

plt.plot(prices_good, color='g')
plt.plot(prices_bad, color='r')
plt.plot(prices_norm, color='b')

temp_df[temp_df['pred']>.3]['profit'].describe()
temp_df[temp_df['pred']<-.3]['profit'].describe()
temp_df[abs(temp_df['pred'])<.3]['profit'].describe()


###############################################
# Fit tradosaurus
###############################################

from tradosaurus import tradosaurus

num_features = features_train.shape[1]
mini_batch_size = 500
features_train_tradosaurus = np.concatenate([features_train, labels_train.reshape([len(labels_train),1])], axis=1)
features_test_tradosaurus = np.concatenate([features_test, labels_test.reshape([len(labels_test),1])], axis=1)

saurus = tradosaurus([128, 64, 32], 10, .01, .99, num_features, 0, .01, mini_batch_size)
saurus.evolve(500, features_train_tradosaurus, features_test_tradosaurus)

weights = saurus.return_weights()
actions, reward = saurus.trade(features_test_tradosaurus, weights, mini_batch=False)
actions = np.array(actions)

plt.scatter(actions, labels_test)

good_preds=np.where(actions>.01)[0]
bad_preds=np.where(actions<.0)[0]
norm_preds = np.where(abs(actions-1)<.002)[0]


prices_good = np.mean(price_history_test[good_preds], axis=0)
prices_bad = np.mean(price_history_test[bad_preds], axis=0)
prices_norm = np.mean(price_history_test[norm_preds], axis=0)

plt.plot(prices_good, color='g')
plt.plot(prices_bad, color='r')
plt.plot(prices_norm, color='b')

    
