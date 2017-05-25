import os
import sys
import numpy as np
from utils.dataset import DataSet
from features import *
from utils.generate_test_splits import split

def get_feature_for_stance(details):
    feature = []
    head_line = details['Headline']
    body_id = details['Body ID']
    stance = details['Stance']
    article_body = dataset.articles[body_id]

    feature.extend(word_overlap_features(head_line, article_body))
    feature.extend(get_word2vector_f(head_line,article_body))
    feature.extend(refuting_features(head_line, article_body))
    feature.extend(polarity_features(head_line, article_body))

    return stance, feature

# save every data point
def np_save(i, stance, feature, data_type):
    file_name = "{}_#{}#".format(i, stance)
    if data_type == 'train':
        file_path = os.path.join(train_data_folder,file_name)
    elif data_type == 'dev':
        file_path = os.path.join(dev_data_folder, file_name)
    elif data_type == 'test':
        file_path = os.path.join(test_data_folder, file_name)

    np.save(file_path, feature)

# =============================================================
# set path
current_path = os.path.dirname(os.path.abspath(__file__))
train_data_folder = os.path.join(current_path, 'data', 'train')
dev_data_folder = os.path.join(current_path, 'data', 'dev')
test_data_folder = os.path.join(current_path, 'data', 'test')
#

# get data
dataset = DataSet()
data_splits = split(dataset)
training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']
#




data_type = 'train'
for i, details in enumerate(training_data):
        stance, feature = get_feature_for_stance(details)
        np_save(i, stance, feature, data_type)


data_type = 'dev'
for i, details in enumerate(dev_data):
        stance, feature = get_feature_for_stance(details)
        np_save(i, stance, feature, data_type)


data_type = 'test'
for i, details in enumerate(test_data):
        stance, feature = get_feature_for_stance(details)
        np_save(i, stance, feature, data_type)



