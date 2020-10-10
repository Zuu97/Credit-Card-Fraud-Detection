import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*
'''  Use following command to run the script
                python model.py
'''
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9975):
            print("\nReached 99.5% train accuracy.So stop training!")
            self.model.stop_training = True

class FraudDetection(object):
    def __init__(self):
        X, Y = get_data()
        self.X = X
        self.Y = Y
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))
        print("No: of Output classes : {}".format(len(set(self.Y))))

    def classifier(self):
        n_features = self.X.shape[1]
        inputs = Input(shape=(n_features,))
        x = Dense(dense1, activation='relu')(inputs)
        x = Dense(dense2, activation='relu')(x)
        x = Dense(dense2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(num_classes, activation='sigmoid')(x)
        self.model = Model(inputs, outputs)

    def train(self):
        callbacks = myCallback()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy'],
        )
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split,
                            # callbacks= [callbacks]
                            )
        self.plot_metrics()
        self.save_model()

    def plot_metrics(self):
        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']
        plt.plot(np.arange(1,num_epoches+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,num_epoches+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(loss_img)
        plt.legend()
        plt.show()

        acc_train = self.history.history['accuracy']
        acc_val = self.history.history['val_accuracy']
        plt.plot(np.arange(1,num_epoches+1), acc_train, 'r', label='Training Accuracy')
        plt.plot(np.arange(1,num_epoches+1), acc_val, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(acc_img)
        plt.legend()
        plt.show()

    def load_model(self, model_weights):
        loaded_model = load_model(model_weights)
        loaded_model.compile(
                        loss='binary_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy'],
                        )
        self.model = loaded_model

    def save_model(self):
        self.model.save(model_weights)

    def run(self):
        if os.path.exists(model_weights):
            print("Loading the model !!!")
            self.load_model(model_weights)
        else:
            print("Training the model !!!")
            self.classifier()
            self.train()

if __name__ == "__main__":
    model = FraudDetection()
    model.run()