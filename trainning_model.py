import numpy as np
from PIL import Image
import os, cv2

def train_classifier() :
	datasets = 'dataset'
	(images, labels, names, id) = ([], [], {}, 0)

	for (sub_dirs, dirs, image) in os.walk(datasets):
		for sub_dir in dirs:
			print(sub_dir)
			names[id] = sub_dir
			subjectpath = os.path.join(datasets, sub_dir)
			for filename in os.listdir(subjectpath):
				path = subjectpath + '/' + filename
				label = id
				images.append(cv2.imread(path, 0))
				labels.append(int(label))
			id += 1 # plus to more folder

	(images, labels) = [np.array(list) for list in [images, labels]]
	classifier = cv2.face.LBPHFaceRecognizer_create()
	classifier.train(images, labels)
	classifier.write("classifier_v1_1.xml")
#
train_classifier()