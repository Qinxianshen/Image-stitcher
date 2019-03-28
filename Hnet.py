from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, ELU, BatchNormalization, Lambda, merge, MaxPooling2D, Input, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.utils import plot_model
from keras.optimizers import Adam,SGD
from keras.callbacks import Callback, RemoteMonitor
import keras.backend as K

from glob import glob
from matplotlib import pyplot as plt
from numpy.linalg import inv
import cv2
import random
import numpy as np
import time


def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))

def homography_regression_model():
    input_shape=(128, 128, 2)
    input_img = Input(shape=input_shape)
     
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv1")(input_img)
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv3")(x)
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv4")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
   
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv5")(x)
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv6")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)
    
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv7")(x)
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv8")(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dropout(0.75, noise_shape=None, seed=None)(x)
    x = Dense(1024, name='FC1')(x)
    out = Dense(8, name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    
    model.compile(optimizer=Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=euclidean_distance)
    return model



#visualizing the model
model = homography_regression_model()
model.summary()



# function for training and test
def get_train(path = "./ms_coco_test_images/*.jpg", num_examples = 1280):
    # hyperparameters
    rho = 32
    patch_size = 128
    height = 240
    width = 320

    loc_list = glob(path)
    X = np.zeros((num_examples,128, 128, 2))  # images
    Y = np.zeros((num_examples,8))
    for i in range(num_examples):
        # select random image from tiny training set
        index = random.randint(0, len(loc_list)-1)
        img_file_location = loc_list[index]
        color_image = plt.imread(img_file_location)
        color_image = cv2.resize(color_image, (width, height))
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

        # create random point P within appropriate bounds
        y = random.randint(rho, height - rho - patch_size)  # row?
        x = random.randint(rho, width - rho - patch_size)  # col?
        # define corners of image patch
        top_left_point = (x, y)
        bottom_left_point = (patch_size + x, y)
        bottom_right_point = (patch_size + x, patch_size + y)
        top_right_point = (x, patch_size + y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

        # compute H
        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inverse = inv(H)
        inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
        warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

        # grab image patches
        original_patch = gray_image[y:y + patch_size, x:x + patch_size]
        warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
        # make into dataset
        training_image = np.dstack((original_patch, warped_patch))
#	plt.imshow(warped_patch) 
#	plt.title('train_images image')
#	plt.show()
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        X[i, :, :] = training_image
        Y[i, :] = H_four_points.reshape(-1)        
    return X,Y

def get_generator(path = "./ms_coco_test_images/*.jpg", num_examples = 256):
    while 1:
        # hyperparameters
        rho = 32
        patch_size = 128
        height = 240
        width = 320

        loc_list = glob(path)
        X = np.zeros((num_examples,128, 128, 2))  # images
        Y = np.zeros((num_examples,8))
        for i in range(num_examples):
            # select random image from tiny training set
            index = random.randint(0, len(loc_list)-1)
            img_file_location = loc_list[index]
            color_image = plt.imread(img_file_location)
            color_image = cv2.resize(color_image, (width, height))
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

            # create random point P within appropriate bounds
            y = random.randint(rho, height - rho - patch_size)  # row
            x = random.randint(rho, width - rho - patch_size)  # col
            # define corners of image patch
            top_left_point = (x, y)
            bottom_left_point = (patch_size + x, y)
            bottom_right_point = (patch_size + x, patch_size + y)
            top_right_point = (x, patch_size + y)
            four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
            perturbed_four_points = []
            for point in four_points:
                perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

            # compute H
            H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
            H_inverse = inv(H)
            inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
            warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

            # grab image patches
            original_patch = gray_image[y:y + patch_size, x:x + patch_size]
            warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
            # make into dataset
            training_image = np.dstack((original_patch, warped_patch))
            H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
            X[i, :, :] = training_image
            Y[i, :] = H_four_points.reshape(-1)        
        yield (X,Y)

def get_test(path):
    rho = 32
    patch_size = 128
    height = 240
    width = 320
    #random read image
    loc_list = glob(path)
    index = random.randint(0, len(loc_list)-1)
    img_file_location = loc_list[index]
    color_image = plt.imread(img_file_location)
    color_image = cv2.resize(color_image,(width,height))
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    #points
    y = random.randint(rho, height - rho - patch_size)  # row
    x = random.randint(rho,  width - rho - patch_size)  # col
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    four_points_array = np.array(four_points)
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
        
    #compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)
    inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
    # grab image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
    # make into dataset
    training_image = np.dstack((original_patch, warped_patch))
    val_image = training_image.reshape((1,128,128,2))
    
    return color_image, H_inverse,val_image,four_points_array,four_points


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):

	# Get width and height of input images	
	w1,h1 = img1.shape[:2]
	w2,h2 = img2.shape[:2]

	# Get the canvas dimesions
	img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
	img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


	# Get relative perspective of second image
	img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

	# Resulting dimensions
	result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

	# Getting images together
	# Calculate dimensions of match points
	[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	
	# Create output array after affine transformation 
	transform_dist = [-x_min,-y_min]
	transform_array = np.array([[1, 0, transform_dist[0]], 
								[0, 1, transform_dist[1]], 
								[0,0,1]]) 

	# Warp images to get the resulting image
	result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
									(x_max-x_min, y_max-y_min))
	result_img[transform_dist[1]:w1+transform_dist[1], 
				transform_dist[0]:h1+transform_dist[0]] = img1

	# Return the result
	return result_img

##############           train

#train_number = 200
#t0 = time.time()
#for i in range(train_number):
#    t1 = time.time()
#    train_images,train_labels = get_train(path = "./test_images/*.jpg", num_examples = 1280)      
#    model = homography_regression_model()
#    print("loading model weights")
#    model.load_weights('my_model_weights.h5')
#    print("training ......")
#    model.fit(train_images,train_labels,epochs=1, batch_size=64)
#    print("saving model weights")
#    model.save_weights('my_model_weights.h5')
#    K.clear_session()
#    t2 = time.time()
#    print("training_number:"+str(i)+"   spend time:"+str(t2-t1)+"s" + "    total time:" + str(t2-t0)+"s")

#train_images,train_labels = get_train(path = "./test_images/*.jpg", num_examples = 5)
#print(train_images.shape)
#print(len(train_labels))



#######################################voc image test 
K.clear_session()
model = homography_regression_model()
model.load_weights('my_model_weights.h5')

color_image, H_matrix,val_image,four_points_array,four_points = get_test("./test_images/*.jpg")
four_points_array_ = four_points_array.reshape((1,4,2))
rectangle_image = cv2.polylines(color_image, four_points_array_, 1, (0,0,255),2)
warped_image = cv2.warpPerspective(rectangle_image, H_matrix, (color_image.shape[1], color_image.shape[0]))
labels = model.predict(val_image)
K.clear_session()
labels_ = np.int32(labels.reshape((4,2)))
perturbed_four = np.subtract(four_points_array,labels_)
print(perturbed_four)
print("perturbed_four------")
print(four_points_array)
print("four_points_array------")
print(labels_)
print("labels_---------------")
perturbed_four_ = perturbed_four.reshape((1,4,2))
warped_image = cv2.polylines(warped_image, perturbed_four_, 1, (255,0,0),2)
 
plt.imshow(rectangle_image) 
plt.title('original image')  
plt.show()



plt.imshow(warped_image) 
plt.title('warped_image image')
plt.show()


######################################### image stiiching


# Stitch the images together using homography matrix

four_points=np.float32(four_points)
perturbed_four_points = perturbed_four_.reshape((4,2))
print(four_points.shape)
print(perturbed_four_points.shape)
H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
H_inverse = inv(H)
result_image = get_stitched_image(rectangle_image, warped_image, H_inverse)

plt.imshow(result_image) 
plt.title('result_image image')
plt.show()





