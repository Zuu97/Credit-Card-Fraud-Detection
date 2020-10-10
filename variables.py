import os
data_path = 'creditcard.csv'

dense1 = 1024
dense2 = 512
dense3 = 128
dense4 = 64
keep_prob = 0.3

num_classes = 1

learning_rate = 0.0001
batch_size = 128
num_epoches = 10
validation_split = 0.2

model_weights = 'weights/disease_prediction.h5'
acc_img = "visualization/accuracy_comparison.png"
loss_img = "visualization/loss_comparison.png"
confusion_matrix_img = "visualization/confusion_matrix.png"