
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, LeakyReLU
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import optimizers

lines = []
with open('../sim_data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

correction = 0.2
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	for i in range(3):
            		path = '../sim_data/IMG/' + batch_sample[i].split('/')[-1]
            		image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
            		tmp = image[:,:,1].reshape(160, 320, 1)
            		images.append(tmp)
            		tmp2 = cv2.flip(tmp,1).reshape(160, 320, 1)
            		images.append(tmp2)

            	angle = float(batch_sample[3])
            	temp = [angle, -1.*angle, angle+correction, -1.*(angle+correction), angle-correction, -1.*(angle-correction)]
            	angles.extend(temp)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 1)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Lambda(lambda image:K.tf.image.resize_images(image, size=(128, 128))))

model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
model.add(LeakyReLU())
model.add(Conv2D(24, (5, 5), strides=(3, 3), padding="same"))
model.add(LeakyReLU())
model.add(Conv2D(36, (5, 5), strides=(3, 3), padding="same"))
model.add(LeakyReLU())
model.add(Conv2D(48, (5, 5), strides=(3, 3), padding="same"))
model.add(LeakyReLU())
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(LeakyReLU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(LeakyReLU())
model.add(Dense(1))

model.compile(optimizer='Adam', loss='mse')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
model.save('model.h5')