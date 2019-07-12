import cv2
import os

# while True:
# 	try:
# 		input_id = int(input('\n Enter id : '))
# 	except ValueError:
# 		print("id must be integer")
# 		continue
# 	else:
# 		break

# face_id = str(input_id)
while True:
	input_name = input('Enter your name : ')
	if not input_name.strip():
		print("*** Please enter your name. ***")
		continue
	else:
		break
face_name = str(input_name)

print(" ")
print("detecting face of : " + face_name)

dataset = 'dataset'
sub_dataset = str(face_name)

path_dir = os.path.join(dataset, sub_dataset)

if not os.path.isdir(path_dir):
	os.mkdir(path_dir)

path = [os.path.join(path_dir, f) for f in os.listdir(path_dir)]

faceCascade = cv2.CascadeClassifier("model/haarcascade_frontalface_altold_6.xml")
# fullbodyCascade = cv2.CascadeClassifier("model/haarcascade_fullbody.xml")
# eyeglassesCascade = cv2.CascadeClassifier("model/haarcascade_eye_tree_eyeglasses.xml")


def create_dataset(img, img_id, path_dir):
	cv2.imwrite('%s/%s.jpg' % (path_dir, img_id), img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
	features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors) #detect face
	coords = []
	for (x, y, w, h) in features :
		cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
		cv2.putText(img, text, (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2) #set Text
		coords = [x, y, w, h]
	return img,coords

def detect(img, faceCascade, img_id, path_dir) :
	img, coords = draw_boundary(img, faceCascade, 1.1, 10, (0,0,255), "Face")
	if len(coords) == 4:
		coords_x = coords[0]
		coords_y = coords[1]
		coords_w = coords[2]
		coords_h = coords[3]
		result = img[coords_y : coords_y + coords_h , coords_x : coords_x + coords_w ]
		create_dataset(result, img_id, path_dir)
	return img

# cam = cv2.VideoCapture("vdo/pasu3.mov")
cam = cv2.VideoCapture(0)
# cam = cv2.imgread("img/jn1.jpg")
img_id = 0
if len(path) != 0:
	img_id = len(path) +1

while (True) :
	ret,frame = cam.read()
	if ret:
		frame = detect(frame, faceCascade, img_id, path_dir)
		cv2.imshow('frame', frame)
		img_id += 1
		if (cv2.waitKey(1) & 0xFF== ord('q')) :
			break
	else :
		break
cam.release()
cv2.destroyAllWindows()
