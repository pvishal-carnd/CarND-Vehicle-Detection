from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import itertools
import pickle
import cv2
from scipy.ndimage.measurements import label
from train import extract_features, convertColor

state = {}
detectParams = {}
detectParams['windowSizes'] = [
                              ((64, 64),  [400, 500]),
                               ((96, 96),  [400, 500]),
                               ((128, 128),[400, 578]),
                               ((192, 192),[450, 700]),
                               ((256, 256),[450, 700])
                              ]
detectParams['windowOverlap'] = (0.5, 0.5)
detectParams['heatmapCacheSize'] = 10
detectParams['heatmapThreshold'] = 10

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

def genWindowList(sizeList, imgSize, overlaps):
    output = []
    for winSize, yLims in sizeList:
        windows = genSameSizeWindows(imgSize, x_start_stop=None, y_start_stop=yLims,
                                    xy_window=winSize, xy_overlap=overlaps)
        output.extend(windows)
    return output

def predictBinary(clf, features):
    return clf.predict(features)

def predictWithMargin(clf, features, threshold):
    margin = clf.decision_function(features)
    return margin > threshold


def searchOverWindows(img, windows, clf, scaler,
                      spatialParams, colorParams, hogParams):

    clfSize = spatialParams['clfSize']
    # A list to store all positive windows
    positives = []

    # Iterate over all windows in the input image
    for win in windows:
        # Extract pixels and resize
        winImg = cv2.resize(img[win[0][1]:win[1][1], win[0][0]:win[1][0]], clfSize)
        features = extract_features(winImg, spatialParams, colorParams, hogParams)

        # Have the scaler scale the features
        scFeatures = scaler.transform(np.concatenate(features).reshape(1, -1))

        # Have the classifier make the prediction
        #prediction = predictBinary(clf, scFeatures)
        prediction = predictWithMargin(clf, scFeatures, 0.7)
        if prediction:
            positives.append(win)

    return positives

#def updateHeatMap(windows, heatmap=None):
#    # Iterate through list of bboxes
#    for win in windows:
#        # Add += 1 for all pixels inside each bbox
#        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
#        heatmap[win[0][1]:win[1][1], win[0][0]:win[1][0]] += 1
#
#    # Return updated heatmap
#    return heatmap

def genHeatmap(windows, imgSize):
    heatmap = np.zeros(imgSize, dtype=np.float)

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


def processFrame(img, intermediates=True):

    cspace = state['hogParams']['colorSpace']
    cImg = convertColor(img, cspace)
    positives = searchOverWindows(cImg, state['windows'], state['clf'],
            state['scaler'], state['spatialParams'], state['colorParams'],
            state['hogParams'])

    #print(len(positives))
    #heatmap = state['heatmap']
    heatmapCurrent = genHeatmap(positives, state['imgSize'])
    state['heatmaps'].append(heatmapCurrent)
    heatmap = thresholdHeatmap(sum(state['heatmaps']), detectParams['heatmapThreshold'])

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    drawImg = drawLabels(np.copy(img), labels)

    if intermediates:
        heatmap = heatmap/np.max(heatmap)
        winImg = np.copy(img)
        for p1, p2 in itertools.chain(positives):
            cv2.rectangle(winImg, p1, p2, (15,15,200), 3)

        imgTopLeft  = img/np.max(img)
        imgTopRight = winImg/np.max(winImg)
        imgBotLeft  = np.dstack(( heatmap, heatmap, heatmap ))
        imgBotRight = drawImg/np.max(drawImg)
        outFrame = np.vstack((
                        np.hstack( (imgTopLeft, imgTopRight) ),
                        np.hstack( (imgBotLeft, imgBotRight) )
                        ))
        outFrame = (outFrame*255).astype(np.uint8)
    else:
        outFrame = drawImg

    return outFrame


if __name__ == '__main__':

    ## Read the saved model and save it to a local cache
    loaded = pickle.load(open('model.pkl', 'rb'))
    state['clf']           = loaded['model']
    state['scaler']        = loaded['scaler']
    state['spatialParams'] = loaded['spatialParams']
    state['colorParams']   = loaded['colorParams']
    state['hogParams']     = loaded['hogParams']
    state['imgSize']       = (720, 1280)

    state['windows'] = genWindowList(detectParams['windowSizes'],
                        state['imgSize'], detectParams['windowOverlap'])
    #state['heatmap'] = np.zeros(state['imgSize'], dtype=np.float)
    state['heatmaps'] = deque(maxlen=detectParams['heatmapCacheSize'])

    #videoIn = VideoFileClip('./test_video.mp4')
    videoIn = VideoFileClip('./project_video.mp4')
    videoOut = videoIn.fl_image(processFrame)
    videoOut.write_videofile('out.mp4', audio=False)

    ## Test window creation
    #testImg = mpimg.imread('test_images/test1.jpg')
    #outImg = processFrame(testImg, intermediates=True)

    #fig = plt.figure()
    #plt.imshow(outImg)
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()


