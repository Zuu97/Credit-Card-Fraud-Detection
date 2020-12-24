import os
data_path = 'creditcard.csv'

# DNN parameters
dense1 = 1024
dense2 = 512
dense3 = 128
dense4 = 64
keep_prob = 0.3

# 1Dcnn parameters
imput_features = 28
input_shape = (1, imput_features)
conv1_cnn = 1200
conv2_cnn = 400
conv3_cnn = 400
conv4_cnn = 200
gru_cnn = 200
keep_prob = 0.5

num_classes = 1

seed = 42
learning_rate = 0.0001
batch_size = 32
num_epoches = 50
validation_split = 0.15
test_split = 0.15
negative_to_positive_ratio = 2.5

model_weights = 'weights/fraud_detection.h5'
model_converter = "weights/fraud_detection.tflite"
acc_img = "visualization/accuracy_comparison.png"
loss_img = "visualization/loss_comparison.png"
confusion_matrix_img = "visualization/confusion_matrix.png"