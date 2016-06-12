#! -*- coding: utf-8 -*-

import keras
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from animeface import AnimeFaceDataset

class Schedule(object):
	def __init__(self, init=0.01):
		self.init = init
	def __call__(self, epoch):
		lr = self.init
		for i in xrange(1, epoch+1):
			if i%5==0:
				lr *= 0.5
		return lr

def get_schedule_func(init):
	return Schedule(init)

def build_deep_cnn(num_classes=3):
	model = Sequential()

	model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(256, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(256, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	return model

if __name__ == '__main__':
	dataset = AnimeFaceDataset()
	dataset.load_data_target()
	x = dataset.data
	y = dataset.target
	n_class = len(set(y))
	perm = np.random.permutation(len(y))
	x = x[perm]
	y = y[perm]

	model = build_deep_cnn(n_class)
	model.summary()
	init_learning_rate = 1e-2
	opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])
	early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
	lrs = LearningRateScheduler(get_schedule_func(init_learning_rate))

	hist = model.fit(x, y, 
					 batch_size=128, 
					 nb_epoch=50, 
					 validation_split=0.1, 
					 verbose=1, 
					 callbacks=[early_stopping, lrs])

	json_string = model.to_json()
	with open("./model_arch.json", "w") as f:
		f.write(json_string)
	model.save_weights("./model_weights.h5")

	df = pd.DataFrame(hist.history)
	df.index += 1
	df.index.name = "epoch"
	df[["acc", "val_acc"]].plot(linewidth=2)
	plt.savefig("acc_history.pdf")
	df[["loss", "val_loss"]].plot(linewidth=2)
	plt.savefig("loss_history.pdf")
	










