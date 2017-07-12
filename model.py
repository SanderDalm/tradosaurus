from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import matplotlib.pyplot as plt

from create_features import get_random_forest_features, get_nn_features
from batch_generator import BatchGenerator
from rnn import RNN

###############################################
# Load random forest features
###############################################

features_train, labels_train, price_history_train, features_test, labels_test, price_history_test=\
get_random_forest_features(1, 30, 10)

###############################################
# Fit random forest
###############################################

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(features_train, labels_train)
model.score(features_train, labels_train)
model.score(features_test, labels_test)
plt.scatter(model.predict(features_train), labels_train)
plt.scatter(model.predict(features_test), labels_test)
importance=model.feature_importances_ 


temp = np.concatenate([features_test, labels_test.reshape([len(labels_test),1])], axis=1)
temp_df = pd.DataFrame(temp)
col_list = ['col'+str(x) for x in range(len(temp_df.columns))]
col_list[-1]='profit'
temp_df.columns = col_list

temp_df['pred'] = model.predict(features_test)

preds_good=np.where(temp_df['pred']>.3)[0]
preds_bad=np.where(temp_df['pred']<-.3)[0]
preds_norm = np.where(abs(temp_df['pred'])<.3)[0]


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

###############################################
# Load RNN features
###############################################

train_dir = '/media/sander/samsungssd/tradosaurus/train_data/'
test_dir = '/media/sander/samsungssd/tradosaurus/test_data/'
batch_size = 50

get_nn_features(100, 10, train_dir, test_dir)
generator = BatchGenerator(train_dir, test_dir, batch_size)

###############################################
# Train RNN
###############################################

