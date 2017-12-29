import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle
import cv2
from scipy.ndimage.measurements import label
from train import extract_features, convertColor

def genSameSizeWindows(imgSize, x_start_stop=None, y_start_stop=None,
                  xy_window=(32, 32), xy_overlap=(0.5, 0.5)):

    if x_start_stop == None:
        x_start_stop = [0, imgSize[1]]

    if y_start_stop == None:
        y_start_stop = [0, imgSize[0]]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def genWindowList(sizeList, imgSize):
    output = []
    for winSize, yLims in sizeList:
        windows = genSameSizeWindows(imgSize, x_start_stop=None, y_start_stop=yLims,
                                    xy_window=winSize, xy_overlap=(0.5, 0.5))
        output.extend(windows)
    return output

def searchOverWindows(img, windows, clf, scaler,
                      spatialParams, colorParams, hogParams):

    # A list to store all positive windows
    posWindows = []

    # Iterate over all windows in the input image
    for win in windows:
        # Extract pixels and resize
        # TODO: Don't hardcode (64,64)
        winImg = cv2.resize(img[win[0][1]:win[1][1], win[0][0]:win[1][0]], (64, 64))
        features = extract_features(winImg, spatialParams, colorParams, hogParams)

        # Have the scaler scale the features
        scFeatures = scaler.transform(np.concatenate(features).reshape(1, -1))

        # Have the classifier make the prediction
        prediction = clf.predict(scFeatures)
        if prediction:
            posWindows.append(win)

    return posWindows

def updateHeatMap(heatmap, windows):
    # Iterate through list of bboxes
    for win in windows:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[win[0][1]:win[1][1], win[0][0]:win[1][0]] += 1

    # Return updated heatmap
    return heatmap


def thresholdHeatmap(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def drawLabels(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


if __name__ == '__main__':

    ## Read the saved model

    loaded = pickle.load(open('model.pkl', 'rb'))
    clf = loaded['model']
    scaler = loaded['scaler']
    spatialParams = loaded['spatialParams']
    colorParams   = loaded['colorParams']
    hogParams     = loaded['hogParams']

    ## Test window creation
    cspace = hogParams['colorSpace']
    testImg = mpimg.imread('test_images/test1.jpg')
    procImg = convertColor(testImg, cspace)
    imgSize = procImg.shape[:2]
    windowSizes = [
           ((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[400, 578]),
           ((192, 192),[450, 700]),
           ((256, 256),[450, 700])
      ]

    windows = genWindowList(windowSizes, imgSize)

    posWindows = searchOverWindows(procImg, windows, clf, scaler,
            spatialParams, colorParams, hogParams)

    heatmap = np.zeros_like(testImg[:,:,0]).astype(np.float)
    updateHeatMap(heatmap, posWindows)
    #thresholdHeatmap(heatmap, 1)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    drawImg = drawLabels(np.copy(testImg), labels)

    fig = plt.figure()
    plt.subplot(1,3,1)
    for p1, p2 in itertools.chain(posWindows):
        cv2.rectangle(testImg, p1, p2, (15,15,200), 4)
    plt.imshow(testImg)
    plt.subplot(1,3,2)
    plt.imshow(heatmap)
    plt.subplot(1,3,3)
    plt.imshow(drawImg)
    plt.show()


