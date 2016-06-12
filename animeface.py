#! -*- coding: utf-8 -*-

import os
import six.moves.cPickle as pickle
import numpy as np
try:
	import cv2 as cv
except:
	pass
from progressbar import ProgressBar

class AnimeFaceDataset:
	def __init__(self):
		self.data_dir_path = u"./animeface-character-dataset/thumb/"
		self.data = None
		self.target = None
		self.n_types_target = -1
		self.dump_name = u'animedata'
		self.image_size = 32

	def get_dir_list(self):
		tmp = os.listdir(self.data_dir_path)
		if tmp is None:
			return None
		ret = []
		for x in tmp:
			if os.path.isdir(self.data_dir_path+x):
				if len(os.listdir(self.data_dir_path+x)) >= 2:
					ret.append(x)
		return sorted(ret)

	def get_class_id(self, fname):
		dir_list = self.get_dir_list()
		dir_name = filter(lambda x: x in fname, dir_list)
		return dir_list.index(dir_name[0])

	def get_class_name(self, id):
		dir_list = self.get_dir_list()
		return dir_list[id]

	def load_data_target(self):
		if os.path.exists(self.dump_name+".pkl"):
			print "load from pickle"
			self.load_dataset()
			print "done"
		else:
			dir_list = self.get_dir_list()
			ret = {}
			self.target = []
			self.data = []
			print("now loading...")
			pb = ProgressBar(min_value=0, max_value=len(dir_list)).start()
			for i, dir_name in enumerate(dir_list):
				pb.update(i)
				file_list = os.listdir(self.data_dir_path+dir_name)
				for file_name in file_list:
					root, ext = os.path.splitext(file_name)
					if ext == u'.png':
						abs_name = self.data_dir_path+dir_name+'/'+file_name
						# read class id i.e., target
						class_id = self.get_class_id(abs_name)
						self.target.append(class_id)
						# read image i.e., data
						image = cv.imread(abs_name)
						image = cv.resize(image, (self.image_size, self.image_size))
						image = image.transpose(2,0,1)
						image = image/255.
						self.data.append(image)
			pb.finish()
			print("done.")
			self.data = np.array(self.data, np.float32)
			self.target = np.array(self.target, np.int32)

			self.dump_dataset()

	def dump_dataset(self):
		pickle.dump((self.data,self.target), open(self.dump_name+".pkl", 'wb'), -1)

	def load_dataset(self):
		self.data, self.target = pickle.load(open(self.dump_name+".pkl", 'rb'))


if __name__ == '__main__':
	dataset = AnimeFaceDataset()
	dataset.load_data_target()


