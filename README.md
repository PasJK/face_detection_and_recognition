# face_detection_and_recognition
First install
- python3.xx
- opencv
--------------------------------------------------------
How to use
1.run file - detect_create_dataset.py
2.Enter your name (dataset name) and press enter.
3.after finished , open file model_training.py and set your model.xml name in line 24 [ classifier.write("classifier_v1_1.xml") ]
4.after created a model, open file classifier_face.py
5.change your model name in line 66 [ clf.read("classifier_v1_1.xml") ] and run.
6.wait a minutes program will show camera and detect face.
