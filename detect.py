import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2

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
        output.append(windows)
    return output

if __name__ == '__main__':


    ## Test window creation
    testImg = mpimg.imread('test_images/test1.jpg')
    imgSize = testImg.shape[:2]
    pyramid = [
            ((64, 64),  [400, 500]),
            ((96, 96),  [400, 500]),
            ((128, 128),[450, 600]),
              ]
    windows = genWindowList(pyramid, imgSize)

    #print(len(list(itertools.chain(*windows))))
    print(imgSize)
    for p1, p2 in itertools.chain(*windows):
        cv2.rectangle(testImg, p1, p2, (15,15,200), 4)
    plt.imshow(testImg)
    plt.show()

    #print(len(windows))

