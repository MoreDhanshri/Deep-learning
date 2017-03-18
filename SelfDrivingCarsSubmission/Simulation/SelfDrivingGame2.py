import pygame
import glob
import numpy as np
import scipy.misc
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

#read data.txt
with open("/home/dhanshri/Documents/Deep learning/Midterm Project/Self Driving/coding stage 1/Autopilot-TensorFlow-master/driving_dataset/data.txt") as f:
    for line in f:
        xs.append("/home/dhanshri/Documents/Deep learning/Midterm Project/Self Driving/coding stage 1/Autopilot-TensorFlow-master/driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)


xs_val=xs[-int(len(xs)*0.05):]
ys_val=ys[-int(len(xs)*0.05):]

X_val=[]
for i in range(len(xs_val)):
    X_val.append(scipy.misc.imresize(scipy.misc.imread(xs_val[i]),[66, 200]).astype('float32')/255)
    
Y_val=[]

for i in range(len(ys_val)):
    Y_val.append(ys_val[i])

X_val=np.array(X_val)
Y_val=np.array(Y_val)

json_file = open('/home/dhanshri/Documents/DeepLearning/Midterm Project/Self Driving/coding stage 2/TrainedModel/SelfDrivingSmallNetwork/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/dhanshri/Documents/DeepLearning/Midterm Project/Self Driving/coding stage 2/TrainedModel/SelfDrivingSmallNetwork/model.h5")
print("Loaded model from disk")

#compile loaded model
loaded_model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')



pygame.init()
size = (256, 455)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

for i in range(len(X_val)):
#for i in range(len(filenames)):
    angle = X_val[i]
    true_angle = loaded_model.predict(Y_val[i])
    
    # add image to screen
    img = pygame.image.load(xs_val[i])
    screen.blit(img, (0, 0))
    
    # add text
    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    screen.blit(pred_txt, (10, 280))
    screen.blit(true_txt, (10, 300))

    # draw steering wheel
    radius = 50
    pygame.draw.circle(screen, WHITE, [320, 300], radius, 2) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [320 + int(x), 300 - int(y)], 7)
    
    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5) 
    
    #pygame.display.update()
    pygame.display.flip()
