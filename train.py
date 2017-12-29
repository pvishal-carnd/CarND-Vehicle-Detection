import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

spatialParams = {}
spatialParams['enabled'] = True
spatialParams['size'] = (16, 16)
spatialParams['clfSize'] = (64, 64)

colorParams = {}
colorParams['enabled'] = True
colorParams['nBins'] = 16
colorParams['binsRange'] = (0, 256)

hogParams = {}
hogParams['enabled'] = True
hogParams['colorSpace'] = 'YCrCb'
hogParams['orient'] = 9
hogParams['pixPerCell'] = 8
hogParams['cellsPerBlock'] = 2
hogParams['hogChannel'] = 0 # Can be 0, 1, 2, or "ALL"


#spatialParams = {}
#spatialParams['enabled'] = True
#spatialParams['size'] = (16, 16)
#spatialParams['clfSize'] = (64, 64)
#
#colorParams = {}
#colorParams['enabled'] = True
#colorParams['nBins'] = 32
#colorParams['binsRange'] = (0, 256)
#
#hogParams = {}
#hogParams['enabled'] = True
#hogParams['colorSpace'] = 'LUV'
#hogParams['orient'] = 8
#hogParams['pixPerCell'] = 8
#hogParams['cellsPerBlock'] = 2
#hogParams['hogChannel'] = 0 # Can be 0, 1, 2, or "ALL"

def convertColor(rgbImg, cspace):
    # apply color conversion if other than 'RGB'
    if cspace == 'RGB':
        outImg = np.copy(rgbImg)
    elif cspace == 'HSV':
        outImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        outImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        outImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        outImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        outImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2YCrCb)

    return outImg/np.max(outImg)

# Define a function to compute binned color features
def bin_spatial(img, params):
    size = params['size']
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, params):
    nbins = params['nBins']
    bins_range = params['binsRange']
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, params, vis=False, feature_vec=True):
    orient = params['orient']
    pix_per_cell = params['pixPerCell']
    cell_per_block = params['cellsPerBlock']
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  block_norm='L1', visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       block_norm='L1', visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
def extract_features(img, spatialParams, colorParams, hogParams):

    hog_channel = hogParams['hogChannel']

    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(img, spatialParams)

    # Apply color_hist() also with a color space option now
    hist_features = color_hist(img, colorParams)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(get_hog_features(
                                img[:,:,channel],
                                hogParams,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(
                                img[:,:,hog_channel],
                                hogParams,
                                vis=False, feature_vec=True)

    return spatial_features, hist_features, hog_features


def featuresFromImgList(imgs, spatialParams, colorParams, hogParams):

    cspace = hogParams['colorSpace']

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        img = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        img = convertColor(img, cspace)

        spatial_features, hist_features, hog_features = extract_features(img, spatialParams,
                                                        colorParams, hogParams)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features


if __name__ == '__main__':

    # set paths
    carsPath    = './data/*/vehicles/*/*.png'
    notCarsPath = './data/*/non-vehicles/*/*.png'

    # Read the car images
    cars = glob.glob(carsPath)
    notcars = glob.glob(notCarsPath)

    #colorspace = settings['colorspace']
    #spatial = settings['spatial']
    #histbin = settings['histbin']
    #orient = settings['orient']
    #pix_per_cell = settings['pix_per_cell']
    #cell_per_block = settings['cell_per_block']
    #hog_channel = settings['hog_channel']

    car_features    = featuresFromImgList(cars, spatialParams, colorParams, hogParams)
    notcar_features = featuresFromImgList(notcars, spatialParams, colorParams, hogParams)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    xScaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = xScaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = 1;
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:', spatialParams['size'],
        'and', colorParams['nBins'],'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('Training time: ', round(t2-t, 2), 's')
    # Check the score of the SVC
    print('Test Accuracy: ', round(svc.score(X_test, y_test), 4))
    s = pickle.dump({'model':svc, 'scaler': xScaler, 'spatialParams':spatialParams,
                    'colorParams':colorParams, 'hogParams':hogParams},
                    open('model.pkl', 'wb'))

    svc = None

    # Check the prediction time for a single sample
    loaded = pickle.load(open('model.pkl', 'rb'))
    svc = loaded['model']

    t  = time.time()
    n_predict = 10
    print('Random lables:', n_predict, 'labels: ', y_test[0:n_predict])
    print('Predictions:  ', svc.predict(X_test[0:n_predict]))
    t2 = time.time()

    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

