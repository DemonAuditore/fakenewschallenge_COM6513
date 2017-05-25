from sklearn.neural_network import MLPClassifier
from utils.score import report_score
import os
import numpy as np
import sys
import re

current_path = os.path.dirname(os.path.abspath(__file__))
train_data_folder = os.path.join(current_path, 'data', 'train')
dev_data_folder = os.path.join(current_path, 'data', 'dev')
test_data_folder = os.path.join(current_path, 'data', 'test')


# ======================================================================================================================
# read train data
# ======================================================================================================================
datapoint_perc = 0.5
train_data_list = []
train_label_list = []
train_file_name = os.listdir(train_data_folder)
train_file_path_list = [os.path.join(train_data_folder, x) for x in train_file_name]
for i, train_file_path in enumerate(train_file_path_list):
    stance = re.findall(r'#([A-Za-z]+)#', train_file_name[i])[0]
    feature_array = np.load(train_file_path)
    train_data_list.append(feature_array)
    train_label_list.append(stance)
    if i == len(train_file_path_list) * datapoint_perc:
        break
# ======================================================================================================================


# ======================================================================================================================
# read dev data
# ======================================================================================================================
dev_data_list = []
dev_label_list = []
dev_file_name = os.listdir(dev_data_folder)
dev_file_path_list = [os.path.join(dev_data_folder, x) for x in dev_file_name]
for i, dev_file_path in enumerate(dev_file_path_list):
    stance = re.findall(r'#([A-Za-z]+)#', dev_file_name[i])[0]
    feature_array = np.load(dev_file_path)
    dev_data_list.append(feature_array)
    dev_label_list.append(stance)
# ======================================================================================================================


# ======================================================================================================================
# read test data
# ======================================================================================================================
test_data_list = []
test_label_list = []
test_file_name = os.listdir(test_data_folder)
test_file_path_list = [os.path.join(test_data_folder, x) for x in test_file_name]
for i, test_file_path in enumerate(test_file_path_list):
    stance = re.findall(r'#([A-Za-z]+)#', test_file_name[i])[0]
    feature_array = np.load(test_file_path)
    test_data_list.append(feature_array)
    test_label_list.append(stance)
# ======================================================================================================================


# train
#
# for numNode in range(10):
#     for numHL in range(10):
# topo_size = ((numNode+3)*100, numHL+8)

# baseline system
# topo_size = (100, 1)
# improved system
# topo_size = (1000, 9)
# improved system not suitable
topo_size = (300, 8)
print('size of mlpc: {}'.format(topo_size))
mlpc = MLPClassifier(hidden_layer_sizes=topo_size, random_state= 19940807)
mlpc = mlpc.fit(train_data_list, train_label_list)

# dev
predicted = mlpc.predict(dev_data_list)

# # test
# predicted = mlpc.predict(test_data_list)


# # print score
# dev
actual = dev_label_list
# # test
# actual = test_label_list
report_score(actual, predicted)

sys.exit()