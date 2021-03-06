# Vehicle Detection and Tracking

[//]: # (Image References)
[image1]: ./images/confmat.png
[image2]: ./images/slidingwindows.png
[image3]: ./images/detections.png
[image4]: ./images/heatmap.png
[image5]: ./images/project_video_out.gif

## Introduction

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream  and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

This pipeline is then used to draw bounding boxes around the cars in the video. Following is a sneak peak into the results of the project. 

![Result GIF][image5]

### Files 
1. ```train.py```: Code to evaluate features, train the classifier and save it
2. ```detect.py```: Code to load the train classifier and detect cars on video streams
3. ```project_video_out.mp4```: The output video. Can be  found [here](./project_video_out.mp4)
4. ```project_video_out_diag.mp4```: The output video with intermediate results for debugging. Can be found [here](./project_video_out_diag.mp4)
5. ```writeup.md```: This file

## Data Exploration

The training data for this project was provided by Udacity. This dataset consists of labeled images taken from the  GTI vehicle image database [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html), and the [KITTI](http://www.cvlibs.net/datasets/kitti/) vision benchmark suite, and some examples extracted from the project video itself. 

* 64x64 images
* 8792 positive images and 9666 negative 
* Slightly unbalanced with about 10% more negatives than positives. This is not a problem considering that are more "non-car" cases in reality than cars.


## Designing and training a classifier

The dataset was generated from video sequences. This means that there is a high-temporal correlation between images in the dataset and a random train/test split does not really present too many new images in the test phase. While this most probably caused our classifier to overfit, we have not looked into correcting this. We have continued to use a 80-20 train/test split for training and evaluating our classifier.


The following features features were used for our classifier

* Spatial features, that are basically all the pixels of the image resized to an appropriate size
* Color histogram of each color channel
* HOG or Histogram of Oriented Gradients

###  Choice of parameters and channels for HOG

A lot of experimentation went into finalizing the HOG parameters and colorspaces. While the initial tuning was targeted towards accuracy numbers, there had to be re-tuning of the classifier towards the end to reduce the number of false positives - at the cost of the total accuracy. 

The RGB colorspace was discarded early on due to its sensitivity to lighting. Experiments were done with other colorspaces and while there was a very little difference between their performance, the YCrCb colorspace worked the best. This can not be generalized as it is entirely possible for other non-RGB colorspaces to work better for a different set of parameters. The following is a summary of the parameters. The variable names are self-explanatory.

```
spatialParams['enabled']   = True
spatialParams['size']      = (16, 16)
spatialParams['clfSize']   = (64, 64)

colorParams['enabled']     = True
colorParams['nBins']       = 16
colorParams['binsRange']   = (0, 256)

hogParams['enabled']       = True
hogParams['colorSpace']    = 'YCrCb'
hogParams['orient']        = 9
hogParams['pixPerCell']    = 8
hogParams['cellsPerBlock'] = 2
hogParams['hogChannel']    = 0 # Y in this case

```

### Training an SVM 

We used the `LinearSVC` taken from the `scikit-learn` package to train our classifier. We also tried the RBF kernel but that took significantly more time to train and evaluate and a linear kernel gave us sufficient accuracy. 

While the accuracy was the classifier was close to 97%, there were still a large number of false positives. Some parts of the side rails and lane markings were systematically being detected as positives with a high degree of confidence. We describe an approach to combat those in a later section.      

The classifier is significantly fast to evaluate taking much less than a millisecond to predict.

![Confusion matrix][image1]

Clearly, our classifier has 100% recall but we do have false positives.

### Sliding Window Search

To use a trained classifier to detect cars in an image, it is necessary to divide the input image into multiple small windows and pass each of them to the classifier. In addition, we have to attempt doing this for multiple window sizes because a car could be of any size. To reduce the number of windows evaluate, we do the following,

1. We evalute four window sizes: 64x64, 96x96, 128x128 and 192x192. 
2. For each of the above sizes, we limit the area of search considering the physical possibiliy of finding cars of those sizes in different areas of the image. Here is the list that specifies this:
```
detectParams['windowSizes'] = [((64, 64),  [400, 500]),
                                  ((96, 96),  [400, 500]),
                                  ((128, 128),[400, 578]),
                                  ((192, 192),[450, 700])]
```  
The following image shows the multiscale sliding windows.
![Sliding Windows][image2]

### False positives 
Let us start with inspecting the results of our sliding window search. We see that all the cars, that are large enough in size, are detected by our pipeline. This is according to what we saw in our confusion matrix.

![Detections][image3]

Also like we see in the confusion matrix, we have false positives. Certain parts of side rails, lane markings and some tree borders were consistently been shown as positives. To reject them, the following strategy was used.

1. **Re-traning the classifier:**  Initially, accuracy was the only criterion for tuning the classifier and training parameters. Later in the project, rate of false positives was also considered as a criteria. 
2. **Thresholding the decision function:** For an SVM classifier, the distance of a feature point from the seperating hyperplane can be considered as a metric for prediction confidence. Therefore, points that are too close to this line were rejected. This helped reject some false positives. Implementation of both binary and thresholded prediction schemes is shown below:

	```
	def predictBinary(clf, features):
    	return clf.predict(features)
	
	def predictWithMargin(clf, features, threshold):
    	margin = clf.decision_function(features)
	    return margin > threshold
	
	```

3. **Heatmaps:**, for which we devote the entire next section.

#### Rejecting false-positives with integrated heatmaps

We use heatmaps to reject false-positives and consolidate multiple detection of the same object.

We do the following:

1. We initialize a deque of `maxlength = 10` frames. A deque allows us to keep a moving list of last `n` objects. If a new object is added after the deque is full, the oldest object is forgotten. 
	```state['heatmaps'] = deque(maxlen=detectParams['heatmapCacheSize'])```
2. For the current frame, we start with a heatmap of all zeros.
2. For all the positive windows in the current frame, we add 1 to the value of all the pixels in that window. This if our heatmap for the current frame.
	```
	def genHeatmap(windows, imgSize):
	    heatmap = np.zeros(imgSize, dtype=np.float)
	
	    for win in windows:
	        # Add += 1 for all pixels inside each bbox
	        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
	        heatmap[win[0][1]:win[1][1], win[0][0]:win[1][0]] += 1
	
	    # Return updated heatmap
	    return heatmap
	
	``` 
3. We append this heatmap to the deque.
4. We sum all heatmaps in the deque and threshold them. This gives us pixels that have been consistently been detected as positives.
	```
    heatmapCurrent = genHeatmap(positives, state['imgSize'])
    state['heatmaps'].append(heatmapCurrent)
    heatmap = thresholdHeatmap(sum(state['heatmaps']), detectParams['heatmapThreshold'])
	```
5. We use this thresholded heatmap to label pixels using `scipy.ndimage.measurements.label()` and draw bounding boxes.

Following are the heatmaps for the set of images shown above. The false positives are consistent and some of them are also seen across frame. We therefore had to integrate the heatmaps over 10 frames and keep a high acceptance threshold.
 
![Heatmaps][image4]


```
detectParams['heatmapCacheSize'] = 10
detectParams['heatmapThreshold'] = 10
```

This concludes the project. The output video is [here](project_video_out.mp4). A diagnostic version of this video with intermediate processing frames is also available [here](project_video_out_diag.mp4)


## Discussion

1. The rate of false-postives is still very high. Due to the systematic nature of the side rails and some lane markings being shown as positives, there is a case for hard negative mining. 
2. The bounding boxes are wobbly and a little unstable. This can be improved with further filtering and tracking. Specifically tracking, with a constant velocity model, would give very stable estimates of the car positions through frames even if measured positions and bounding boxes are noisy.  
3. The speed of detection is still very low. In comparison, YOLO object detection works with a larger class of objects with FPS greater than 60. Most of the time is used up in evaluating HOG features. A way to improve speed would be to compute the HOG features only once for the entire region of interest and then select the right feature vectors. In addition, evaluation of critical code in C++ and parallelizing (this is an embarrassingly parallel task) are also possible options. 




