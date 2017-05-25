
import matplotlib.pyplot as plt
import os
import re
import sys
import numpy as np

from sklearn.neural_network import MLPClassifier

model_name_impro_1 = 'improve_(300, 8).txt'
model_name_impro_2 = 'improve_(1000, 9).txt'
model_name_base = 'base_(100, 1).txt'

# # improved model
# file_name_list = [ model_name_impro_1, model_name_impro_2]
# legend_list = ['improved model (300, 8)', 'improved model (1000, 9)']

# baseline model
file_name_list = [model_name_base]
legend_list = ['baseline model']

current_path = os.path.dirname(os.path.abspath(__file__))
learning_curve_folder = os.path.join(current_path, 'learning_curve')
learning_curve_path_list = [os.path.join(learning_curve_folder, x) for x in file_name_list]

loss_array_list = [np.loadtxt(learning_curve_path) for learning_curve_path in learning_curve_path_list]
for loss_array in loss_array_list:

    print('loss_array: {}'.format(loss_array))
    plt.plot(loss_array)

plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('learning curve')
plt.legend(legend_list)

plt.savefig('aaa')
plt.show('aaa')
















