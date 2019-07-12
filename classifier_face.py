import cv2
import os
from matplotlib import pyplot as plt
faceCascade = cv2.CascadeClassifier("model/haarcascade_frontalface_altold_6.xml")
# fullbodyCascade = cv2.CascadeClassifier("model/haarcascade_fullbody.xml")
# eyeglassesCascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml")

datasets = 'dataset'

(images, labels, names, id) = ([], [], {}, 0)

for (sub_dirs, dirs, image) in os.walk(datasets):
	for sub_dir in dirs:
		names[id] = sub_dir
		subjectpath = os.path.join(datasets, sub_dir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			label = id
			images.append(cv2.imread(path, 0))
			labels.append(int(label))
		id += 1 # plus to more folder

def get_histrogram(img):
	color = ('b', 'g', 'r')
	for i, col in enumerate(color):
		hist = cv2.calcHist([img], [i], None, [256], [0, 256])
		plt.plot(hist, color=col)
		print(hist)
	# 	plt.xlim([0, 256])
	# plt.show()

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):  #วาดกรอบ แสดงข้อความ ตามตำแหน่งที่ตรวจจับ
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
	features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors) #detect face
	coords = []
	for (x, y, w, h) in features :
		id,_ = clf.predict(gray[y:y+h,x:x+w]) # id, _ <- คือ ค่า confident
		cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
		cv2.rectangle(img,  (x,y+h+35), (x+w, y+h), color, cv2.FILLED)

		if int(format(round(100 - _))) > 70:
			id = names[id]
			_ = "{0}%".format(round(100 - _))
		else :
			id = "unknow"
			_ = "{0}%".format(round(100 - _))
		get_histrogram(img)

		cv2.putText(img, str(id), (x,y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2) #set Text
		cv2.putText(img, str(_), (x+w-60,y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2) #set Text

		coords = [x, y, w, h]
	return img,coords

def detect(img, faceCascade, clf) : #ตรวจจับใบหน้า

	img, coords = draw_boundary(img, faceCascade, 1.1, 10, (50,252,50), clf)
	# img, coords = draw_boundary(img, fullbodyCascade, 1.1, 10, (50,252,50), clf)
	# img, coords = draw_boundary(img, eyeglassesCascade, 1.1, 10, (50,252,50), clf)
	return img


# cam = cv2.VideoCapture("vdo/pasu_jn_je1.mov")
cam = cv2.VideoCapture(0)
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier_v1_1.xml")
while (True) :
	ret,frame = cam.read() # จับเฟรมภาพ
	if ret:
		frame = detect(frame, faceCascade, clf)
		cv2.imshow('frame', frame)
		if (cv2.waitKey(1) & 0xFF== ord('q')) :
			break
	else :
		break
cam.release()
cv2.destroyAllWindows()
