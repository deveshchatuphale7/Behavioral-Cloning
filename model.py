import os
import cv2
from keras.layers import Input, Lambda
from keras.models import Model
import tensorflow as tf

def load_image(log_path, filename):
    filename = filename.strip()
    if filename.startswith('IMG'):
        filename = log_path+'/'+filename
    else:
        # load it relative to where log file is now, not whats in it
        filename = log_path+'/IMG/'+PurePosixPath(filename).name
    img = cv2.imread(filename)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# randomily change the image brightness
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # brightness - referenced Vivek Yadav post
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0

    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# crop camera image to fit nvidia model input shape
def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]

    # y_start = 60+random.randint(-10, 10)
    # x_start = int(width/2)-int(crop_width/2)+random.randint(-40, 40)
    y_start = 60
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


# referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def jitter_image_rotation(image, steering):
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering


# if driving in a straight line remove extra rows
def filter_driving_straight(data_df, hist_items=5):
    print('filtering straight line driving with %d frames consective' %
          hist_items)
    steering_history = deque([])
    drop_rows = []

    for idx, row in data_df.iterrows():
        # controls = [getattr(row, control) for control in vehicle_controls]
        steering = getattr(row, 'steering')

        # record the recent steering history
        steering_history.append(steering)
        if len(steering_history) > hist_items:
            steering_history.popleft()

        # if just driving in a straight
        if steering_history.count(0.0) == hist_items:
            drop_rows.append(idx)

    # return the dataframe minus straight lines that met criteria
    return data_df.drop(data_df.index[drop_rows])


# jitter random camera image, adjust steering and randomise brightness
def jitter_camera_image(row, log_path, cameras):
    steering = getattr(row, 'steering')

    # use one of the cameras randomily
    camera = cameras[random.randint(0, len(cameras)-1)]
    steering += steering_adj[camera]

    image = load_image(log_path, getattr(row, camera))
    image, steering = jitter_image_rotation(image, steering)
    image = randomise_image_brightness(image)

    return image, steering


# create a training data generator for keras fit_model
def gen_train_data(log_path='./data', log_file='driving_log.csv', skiprows=1,
                   cameras=cameras, filter_straights=False,
                   crop_image=True, batch_size=128):

    # load the csv log file
    print("Cameras: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,
                          names=column_names, skiprows=skiprows)

    # filter out straight line stretches
    if filter_straights:
        data_df = filter_driving_straight(data_df)

    data_count = len(data_df)

    print("Log with %d rows." % (len(data_df)))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]

            image, steering = jitter_camera_image(row, log_path, cameras)

            # flip 50% randomily that are not driving straight
            if random.random() >= .5 and abs(steering) > 0.1:
                image = cv2.flip(image, 1)
                steering = -steering

            if crop_image:
                image = crop_camera(image)

            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))


# create a valdiation data generator for keras fit_model
def gen_val_data(log_path='/u200/Udacity/behavioral-cloning-project/data',
                 log_file='driving_log.csv', camera=camera_centre[0],
                 crop_image=True, skiprows=1,
                 batch_size=128):

    # load the csv log file
    print("Camera: ", camera)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,
                          names=column_names, skiprows=skiprows)
    data_count = len(data_df)
    print("Log with %d rows."
          % (data_count))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]
            steering = getattr(row, 'steering')

            # adjust steering if not center
            steering += steering_adj[camera]

            image = load_image(log_path, getattr(row, camera))

            if crop_image:
                image = crop_camera(image)

            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))



        # Build the Final Test Neural Network in Keras Here
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
