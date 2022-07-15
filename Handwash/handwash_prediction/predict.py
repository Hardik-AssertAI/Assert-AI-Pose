from pickle import GLOBAL
import cv2
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import numpy as np
import statistics
from statistics import mode

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



global batch_size
batch_size = 2
img_width = 224
img_height = 224
global IMG_SIZE
IMG_SIZE = (img_height, img_width)
N_CHANNELS = 3
global IMG_SHAPE
IMG_SHAPE = IMG_SIZE + (N_CHANNELS,)
N_CLASSES = 7
classes=['0','1','2','3','4','5','6']
class_names=['wrists','palm to palm','back of hands', 'between fingers','back of fingers','base of thumbs','fingernails']

# global weights_dict
# weights_dict= {0: 0.5061350327302445, 1: 1.2660703348102422, 2: 0.7916907083167528, 3: 1.5546910866910866, 4: 1.2442803591461484, 5: 1.3146400591400735, 6: 1.3093784706511455}

test_data_dir = './test/'
global model
model = load_model("./kaggle-single-framefinal-model/")

global pred
pred=0
pred_list=[]

def get_datasets(test_data_dir, batch_size=None):
    # if batch_size is None:
    #     batch_size = classify_dataset.batch_size

    # IMG_SIZE = classify_dataset.IMG_SIZE

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=IMG_SIZE,
        label_mode='categorical',
        crop_to_aspect_ratio=False,
        batch_size=batch_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    return test_ds

def measure_performance(ds,pred_list):
    global pred
    for images, labels in ds:
        images=images.reshape((1,224,224,3))
        predicted = model.predict(images)
        for y_p, _ in zip(predicted, labels):
            pred=(int(np.argmax(y_p)))
            print(pred)
            if(len(pred_list)>=10):
                pred_list.pop(0)
            pred_list.append(pred)
            print(pred_list)

vid=cv2.VideoCapture("./test_vids/6.mp4")
while(True):
    ret,frame=vid.read()
    if not ret :
        break
    cv2.imwrite('./test/0/1.jpeg', frame)
    ds=get_datasets(test_data_dir)
    measure_performance(ds,pred_list)
    outp=mode(pred_list)
    disp=class_names[outp]
    cv2.putText(frame, disp, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('prediction', frame)
    if cv2.waitKey(1000//30) & 0xFF == ord('q'):    
        break
vid.release()
cv2.destroyAllWindows()
