
# Building a Vehicle Detection Pipeline using HOG Feaures

#### Kevin Wu, Udacity Self Driving Car Nanodegree Program, Term 1

In this notebook I walk through how I trained a linear SVM to detect images of cars and how I built an image processing pipeline to detect vehicles from videos taken on the road. 

### Histogram of Oriented Gradients (HOG)

I use the sklearn's `hog` function to extract a histogram of oriented gradients from an image.
The parameters used for this feature extraction step include:
- pixels per cell: the n x n cell size over which each gradient histogram is computed
- cells per block: number of cells over which gradients are normalized
- orient: The number of orientation bins in the histogram of gradients


```python
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

```

Let's load in the training images and visualize some examples of HOG features. 


```python
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

%matplotlib inline

cars = glob.glob('vehicles/*/*.png', recursive=True)
notcars = glob.glob('non-vehicles/*/*.png', recursive=True)

print ("Number of car images: ", len(cars))
print ("Number of non-car images: ", len(notcars))

## Print some sample images
car_img = mpimg.imread(cars[0])
notcar_img = mpimg.imread(notcars[0])

fig = plt.figure()
a1=fig.add_subplot(1,2,1)
imgplot = plt.imshow(car_img)
a1.set_title('Car')
a2=fig.add_subplot(1,2,2)
imgplot = plt.imshow(notcar_img)
a2.set_title('Not a Car')
```

    Number of car images:  8792
    Number of non-car images:  8968





    <matplotlib.text.Text at 0x112954128>




![png](output_6_2.png)


I used a combination of intuition and trial-and-error to pick the parameters for calculating HOG features. To develop an intuition for how the various parameters change the output I plotted the results on a random subsample of image data. 

In general, I found that the gradients in the RGB channel were  to be very informative, as the color channels caused the gradient information to be encoded in different channels depending on the color of the car. I had more luck with color spaces that transformed the image into a luma component (Y). I chose YCrCb because it is the most ubiquitous luma-chroma encoding scheme in use today. 


```python
import cv2
import numpy as np

## FINAL PARAMETERS
########################################################################################
COLOR_SPACE = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 9  # HOG orientations
PIX_PER_CELL = 8 # HOG pixels per cell
CELL_PER_BLOCK = 4 # HOG cells per block
HOG_CHANNEL = "ALL"
########################################################################################

def get_hog_images(image, 
                  color_space=COLOR_SPACE, 
                  orient=ORIENT, 
                  pix_per_cell=PIX_PER_CELL,
                  cell_per_block=CELL_PER_BLOCK):

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      
        
    hog_images = []
    for hog_channel in range(feature_image.shape[-1]):
        features, hog_image = get_hog_features(feature_image[:,:,hog_channel], 
                                        orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        hog_images.append(hog_image)
    
    return feature_image, hog_images
    
```

I used the following code to visualize car images from the training data in their new color spaces, as well as the HOG features generated from those channels, to gain intuition as to which combination of features worked best. 

In YCrCb colorspace (and YUV as well), we can visualize how the information content drops off sharply after the first color channel. Initially, I only used the Y channel HOG features in the training data, but I found that using all 3 channeles still improved the test accuracy by ~1%; even though the Cr and Cb channeles among car and non-car images seem largely similar, they still contain important features. 

I did some manual trial-and-error to tweak the other parameters: pixels per cell, orient, and cells per block. The latter two had little effect on the model accuracy. Decreasing pixels per cell led to a slight decrease in test accuracy, while increasing it led to nearly identical performance. I kept `PIX_PER_CELL` at 8 to speed up training and help prevent overfitting. 


```python
import matplotlib.gridspec as gridspec

n_examples = 3

car_idx = np.random.choice(range(len(cars)), n_examples)
notcar_idx = np.random.choice(range(len(notcars)), n_examples)

gs1 = gridspec.GridSpec(n_examples*2, 7)
gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes. 
plt.figure(figsize=(12,12))
for j in range(n_examples):
    car_img = mpimg.imread(cars[car_idx[j]])
    ax0 = plt.subplot(gs1[2*j*7])
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_aspect('equal')
    plt.imshow(car_img)

    
    feature_image, hog_car_imgs = get_hog_images(car_img)
    for i in range(3):
        ax1 = plt.subplot(gs1[2*j*7+2*i+1])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.imshow(feature_image[:,:,i])
        
        ax2 = plt.subplot(gs1[2*j*7+2*i+2])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')
        plt.imshow(hog_car_imgs[i], cmap='gray')
        
    notcar_img = mpimg.imread(notcars[notcar_idx[j]])
    ax0 = plt.subplot(gs1[(2*j+1)*7])
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_aspect('equal')
    plt.imshow(notcar_img)
    
    feature_image, hog_car_imgs = get_hog_images(notcar_img)
    
    
    for i in range(3):
        ax3 = plt.subplot(gs1[(2*j+1)*7+2*i+1])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_aspect('equal')
        plt.imshow(feature_image[:,:,i])
        
        ax4 = plt.subplot(gs1[(2*j+1)*7+2*i+2])
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_aspect('equal')
        plt.imshow(hog_car_imgs[i], cmap='gray')
        
plt.savefig('hog_features.png')
```

There are other ways of describing our image data besides using HOG. The first is by computing a set of RGB color histograms for each image, such as following:

<img src="notebook_files/rgb-histogram-plot.jpg">


```python
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    

```

Finally, we use shrunken down versions of the raw image data (a technique we call here "spatial binning").


```python

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
```

Given my HOG feature vectors, I found that the inclusion of these RGB features did improve the performance of my vehicle detection pipeline (~4 percentage points in test accuracy improvements in the classifier!).

For the new feature extraction techniques, we have two additional parameters for spatial binning size (the size of the shrunked down image), and number of bins in our color histogram.

I found that increasing the size of our spatial bins from 32x32 helped but only marginally. In my final model, I use 32x32 to speed up training/prediction times. 


```python
SPATIAL_SIZE = (32, 32) # Spatial binning dimensions
HIST_BINS = 32    # Number of histogram bins
```

### Classification using SVM

I use the `extract_features` function we defined in the classroom modules to generate a vector of HOG, spatial, and color features given a set of parameters. 

Here are all the hyperparameters I used, in one place:


```python
## HOG
COLOR_SPACE = 'YCrCb' 
ORIENT = 9  
PIX_PER_CELL = 8
CELL_PER_BLOCK = 4 
HOG_CHANNEL = "ALL"

## SPATIAL BINNING
SPATIAL_SIZE = (32, 32)

## COLOR_HISTOGRAMS
HIST_BINS = 32 

## FEATURES TO USE
USE_HOG = True
USE_SPAT = True
USE_COL_HIST = True
```


```python
import time

def convert_color(image, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            raise Exception("Unrecognized color space.")
    else: 
        feature_image = np.copy(image)      
    return feature_image

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        feature_image = convert_color(image, color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
        
    return features


start = time.time()

car_features = extract_features(cars, color_space=COLOR_SPACE, 
                        spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                        orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                        cell_per_block=CELL_PER_BLOCK, 
                        hog_channel=HOG_CHANNEL, spatial_feat=USE_SPAT, 
                        hist_feat=USE_COL_HIST, hog_feat=USE_HOG)

notcar_features = extract_features(notcars, color_space=COLOR_SPACE, 
                        spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                        orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                        cell_per_block=CELL_PER_BLOCK, 
                        hog_channel=HOG_CHANNEL, spatial_feat=USE_SPAT, 
                        hist_feat=USE_COL_HIST, hog_feat=USE_HOG)

end = time.time()

print ("Feature extraction time: {0} seconds".format(round(end-start, 2)))
```

    Feature extraction time: 81.62 seconds


Next, I trained a classifier on our car/non-car image data using two classifiers: linear support vector machine and a multi-layer perceptron.  

Since both classifiers produced accuracies > 90% using the default parameters, I did not do any hyperparameter optimization for this step of the pipelin. 

The default parameters for the Linear SVM include the following: 
- Regularization: L2 
- Objective function: Squared hinge loss

The default parameters fo the MLP include the following: 
- Hidden layers: 1
- Neurons in hidden layer: 100
- Activations: ReLU
The MLP is trained using stochastic gradient descent via the Adam optimizer. 



```python
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
## Create matrix of training data 
X = np.vstack((car_features, notcar_features)).astype(np.float64) 
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

## Training labels. 1: Car, 0: no car
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', ORIENT,'orientations', PIX_PER_CELL,
    'pixels per cell and', CELL_PER_BLOCK,'cells per block')
print('Feature vector length:', len(X_train[0]))
print ("")

# Use a linear SVC and Multilayer perceptron
svc = LinearSVC()
mlp = MLPClassifier()

# Check the training time for the SVC
t1=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t1, 2), 'Seconds to train SVC...')
print("")

## Check training time for MLP
t1=time.time()
mlp.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t1, 2), 'Seconds to train MLP...')
print("")

## Print out metrics of our model
y_pred = svc.predict(X_test)
print ("SVM CLASSIFIER METRICS:")
print('Test Accuracy of SVC = ', round(accuracy_score(y_pred, y_test), 4))
print('Test Precision of SVC = ', round(precision_score(y_pred, y_test), 4))
print('Test Recall of SVC = ', round(recall_score(y_pred, y_test), 4))
print("")

y_pred = mlp.predict(X_test)
print ("MULTILAYER PERCEPTRON METRICS:")
print('Test Accuracy of MLP = ', round(accuracy_score(y_pred, y_test), 4))
print('Test Precision of MLP = ', round(precision_score(y_pred, y_test), 4))
print('Test Recall of MLP = ', round(recall_score(y_pred, y_test), 4))
```

    Using: 9 orientations 8 pixels per cell and 4 cells per block
    Feature vector length: 13968
    
    38.23 Seconds to train SVC...
    
    98.79 Seconds to train MLP...
    
    SVM CLASSIFIER METRICS:
    Test Accuracy of SVC =  0.9907
    Test Precision of SVC =  0.9909
    Test Recall of SVC =  0.9903
    
    MULTILAYER PERCEPTRON METRICS:
    Test Accuracy of MLP =  0.9941
    Test Precision of MLP =  0.9943
    Test Recall of MLP =  0.9937


The primary purpose of printing out the metrics (accuracy, precision, recall) here was to try to improve the final video output. Given that our pipeline involves predicting on each 8x8 subsection over a 500x1000(ish) pixel graph, a one percentage point difference in these metrics actually means a noticeable change in the number of boxes being drawn on our camera image. 

Since evaluating our classifier on the final video output takes a long time, I used the classifier metrics as a benchmark in deciding which sets of hyperparameters to even bother testing on the final output video. 

Because the final step of our pipeline filters out false positives, I was willing to tolerate a slightly lower precision than recall metric (there is no backup mechanism if our classifier fails to identify a car). 


### Sliding Window Search

Now that we have a pretty robust image classifier for vehicle detection, we want to implement a sliding search mechanism to run our model over subsets of a larger image. 

To do this, I divide the larger image into 8x8 pixel cells, and a sliding window search that skips by 2 pixels (meaning two adjacent windows overlap by 75%). 

Since the HOG feature extraction step is computationally expensive, we precompute the gradients over the entire image before stepping through each subwindow.

Since the next step involves aggregating overlapping bounding boxes later, I didn't really spend any time tweaking the parameters for our cell/box size or step size. 

<img src="notebook_files/hog-sub.jpg">

The following function implements the steps of our vehicle detection pipeline so far:
- Image cropping and scaling (more on this later)
- Color space transformation
- HOG calculation
- Sliding window search
    - spatial binning (if applicable)
    - color histogram (if applicable)
    - classifier prediction 


```python
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, hog_channel, color_space):
    boxes = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            features = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
                
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            
            if spatial_feat:
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                features.append(spatial_features)
            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
                features.append(hist_features)
            if hog_feat:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                
                if hog_channel=="ALL":
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else: 
                    hog_features = [hog_feat1, hog_feat2, hog_feat3]
                    hog_features = hog_features[hog_channel]
                    
                features.append(hog_features)
                
            a = np.array(np.concatenate(features)).reshape(1, -1)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(a)    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
                
    return draw_img, boxes
```

A few more parameters: we use `Y_START` and `Y_STOP` to define the latitudes of the image we're interested in; given that it's a fixed camera, no need to tune these here - I'll just use what's provided in the example. 

We also have a parameter called scale to shrink/magnify the subsections of our image before applying the classifier. A larger scale, a.k.a. magnification, will decrease the number of bounding boxes, and vice versa with a smaller scale. A smaller scale also significantly increases the time it takes to process a single frame, so I picked a scale that would be able to pick as many boxes as possible without sacrificing significant performance. 1.5 was the sweet spot. 


```python
Y_START = 400
Y_STOP = 656
SCALE = 1.5
```

Let's see what these boxes look like.


```python
test_imgs = [mpimg.imread(jpg) for jpg in glob.glob('test_images/*.jpg', recursive=True)]

gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.1, hspace=0.2) # set the spacing between axes. 
plt.figure(figsize=(12,4))
boxed_imgs = []
boxes = []
for i in range(6):
    ax1 = plt.subplot(gs1[i])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    output_img, box_coord = find_cars(test_imgs[i], Y_START, Y_STOP, SCALE, mlp, X_scaler, 
                           ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS,
                           spatial_feat=USE_SPAT, hist_feat=USE_COL_HIST, hog_feat=USE_HOG, 
                           hog_channel=HOG_CHANNEL, color_space = COLOR_SPACE)
    plt.imshow(output_img, cmap='gray')
    boxes.append(box_coord)
    boxed_imgs.append(output_img)
    plt.axis('off')
    
    mpimg.imsave("output_images/raw_boxes_{0}.png".format(i), output_img)
```


![png](output_30_0.png)


### Boundary Box Post-Processing

Not bad. In the final part of the image processing pipeline, we want to aggregate overlapping bounding boxes, combine predictions from previous frames, and use this information to remove false positives to the best of our ability. 

To do this, I create a heatmap from the overlapping bounding boxes, filter out highlighted areas below a certain threshold, and then convert the heatmap back to a single bounding box. The `THRESHOLD` parameter is the cutoff (inclusive) for false positives. 


```python
THRESHOLD = 1
```


```python
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def remove_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] -= 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
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

```

Here is the heatmap process at work on individual frames, from left to right:


```python
gs1 = gridspec.GridSpec(len(boxed_imgs), 4)
gs1.update(wspace=0.1, hspace=0.2) # set the spacing between axes. 
plt.figure(figsize=(10,10))
for i in range(len(boxed_imgs)):
    ax0 = plt.subplot(gs1[i*4])
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.imshow(test_imgs[i])
    
    ax1 = plt.subplot(gs1[i*4+1])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    heat = np.zeros_like(boxed_imgs[i][:,:,0]).astype(np.float)
    heat = add_heat(heat,boxes[i])
    
    plt.imshow(np.clip(heat, 0, 255))
    
    ax2 = plt.subplot(gs1[i*4+2])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,THRESHOLD)
    heatmap = np.clip(heat, 0, 255)
    plt.imshow(heatmap)
    
    ax3 = plt.subplot(gs1[i*4+3])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(test_imgs[i], labels)
    plt.imshow(draw_img)
    mpimg.imsave("output_images/processed_boxes_{0}.png".format(i), draw_img)
    plt.axis('off')
```


![png](output_35_0.png)


Below is the code I used to generate the consecutive image frames from the test video. I next test my pipeline on the video frames to watch the heatmap aggregation and filtering process at work. 


```python
# vidcap = cv2.VideoCapture('test_video.mp4')
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#     success,image = vidcap.read()
#     cv2.imwrite("vid_images/frame%03d.jpg" % count, image)
#     count += 1

vid_imgs = [mpimg.imread(i) for i in glob.glob('vid_images/*.jpg', recursive=True)][15:]
```


```python
heatmap = None
heatarray = []

gs1 = gridspec.GridSpec(len(vid_imgs), 5)
gs1.update(wspace=0.1, hspace=0.2) # set the spacing between axes. 
plt.figure(figsize=(15,30))
for i in range(len(vid_imgs)):
    image = vid_imgs[i]
    output_img, box_coord = find_cars(image, Y_START, Y_STOP, SCALE, svc, X_scaler, 
                               ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS,
                                True, True, True, 
                                HOG_CHANNEL, COLOR_SPACE)
    
    ax0 = plt.subplot(gs1[i*5])
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.imshow(output_img)
    
    heat = np.zeros_like(output_img[:,:,0]).astype(np.float)
    heat = add_heat(heat,box_coord)
    
    ax1 = plt.subplot(gs1[i*5+1])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.imshow(np.clip(heat, 0, 255))

    #if heatmap is None:
    #    heatmap = heat
    #else:
    #    heatmap = np.add(heatmap, heat)
    
    heatarray.append(heat)
    #print (len(heatarray))
    if len(heatarray) > 5:
        #heatmap = np.subtract(heatmap, heatarray[0])
        del heatarray[0]
    
    heatmap = np.zeros_like(output_img[:,:,0]).astype(np.float)
    for h in heatarray:
        heatmap = np.add(heatmap, h)
    
    heatmap = np.clip(heatmap, 0, 255)
    
    ax2 = plt.subplot(gs1[i*5+2])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.imshow(heatmap)
    
    filtered_heatmap = apply_threshold(heatmap, THRESHOLD)
    
    ax3 = plt.subplot(gs1[i*5+3])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    plt.imshow(np.clip(filtered_heatmap, 0, 255))
    
    labels = label(filtered_heatmap)
    final_img = draw_labeled_bboxes(image, labels)
    
    ax4 = plt.subplot(gs1[i*5+4])
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    plt.imshow(final_img)
    
```


![png](output_38_0.png)


Finally, we create a `FrameProcessor` object to retain information from previous frames, and then combine all the components of our vehicle detection pipeline into a single function and apply it to a video input. 

The last addition to our pipeline here is to add up the heatmaps of the last N frames, and adjust our thresholding function accordingly.


```python

class FrameProcessor(object):
    
    def __init__(self, y_start, y_stop, scale, classifier, x_scaler,
                orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, 
                spatial_features, hist_features, hog_features, hog_channel, hog_cspace, 
                 memory, threshold):
        
        ## Parameters
        self.y_start = y_start
        self.y_stop = y_stop
        self.scale = scale
        self.classifier = classifier
        self.x_scaler = x_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_features = spatial_features
        self.hist_features = hist_features
        self.hog_features = hog_features
        self.hog_channel = hog_channel
        self.hog_cspace = hog_cspace
        
        self.heatarray = []
        self.memory = memory
        self.threshold = threshold
        self.heatmap = None

    def process_image(self, image):   
        output_img, box_coord = find_cars(image, self.y_start, self.y_stop, self.scale, self.classifier, self.x_scaler, 
                               self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins,
                               self.spatial_features, self.hist_features, self.hog_features, 
                               self.hog_channel, self.hog_cspace)

        heat = np.zeros_like(output_img[:,:,0]).astype(np.float)
        heat = add_heat(heat,box_coord)
        
        self.heatarray.append(heat)
        
        if len(self.heatarray) > self.memory:
            del self.heatarray[0]

        heatmap = np.zeros_like(output_img[:,:,0]).astype(np.float)
        for h in self.heatarray:
            heatmap = np.add(heatmap, h)
        
        heatmap = np.clip(heatmap, 0, 255)
        
        heatmap = apply_threshold(heatmap, self.threshold)
        labels = label(heatmap)
        final_img = draw_labeled_bboxes(image, labels)

        return final_img

    

```

For the last stage of the pipeline, I chose to add the last 10 frames and threshold all regions with less than 6 detections. I found that was able to filter out the most false positives without also significantly affecting the true positive count.  

Finally, I chose to use the MLP classifier over the linear SVM due to its (slightly) higher performance on our test data from earlier.


```python
frameProcessor = FrameProcessor(Y_START, Y_STOP, SCALE, mlp, X_scaler, 
                               ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS,
                                True, True, True, 
                                HOG_CHANNEL, COLOR_SPACE, memory=10, threshold=6)
```


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(frameProcessor.process_image) 
clip.write_videofile(output, audio=False)


```

    [MoviePy] >>>> Building video project_video_output.mp4
    [MoviePy] Writing video project_video_output.mp4


    
      0%|          | 0/1261 [00:00<?, ?it/s][A
      0%|          | 1/1261 [00:00<16:33,  1.27it/s][A
      0%|          | 2/1261 [00:01<16:21,  1.28it/s][A
      0%|          | 3/1261 [00:02<16:00,  1.31it/s][A
      0%|          | 4/1261 [00:02<15:42,  1.33it/s][A
      0%|          | 5/1261 [00:03<16:08,  1.30it/s][A
      0%|          | 6/1261 [00:04<16:08,  1.30it/s][A
      1%|          | 7/1261 [00:05<16:05,  1.30it/s][A
      1%|          | 8/1261 [00:06<15:56,  1.31it/s][A
      1%|          | 9/1261 [00:06<16:17,  1.28it/s][A
      1%|          | 10/1261 [00:07<16:11,  1.29it/s][A
      1%|          | 11/1261 [00:08<15:56,  1.31it/s][A
      1%|          | 12/1261 [00:09<16:01,  1.30it/s][A
      1%|          | 13/1261 [00:09<15:58,  1.30it/s][A
      1%|          | 14/1261 [00:10<16:03,  1.29it/s][A
      1%|          | 15/1261 [00:11<15:52,  1.31it/s][A
      1%|▏         | 16/1261 [00:12<15:41,  1.32it/s][A
      1%|▏         | 17/1261 [00:13<15:51,  1.31it/s][A
      1%|▏         | 18/1261 [00:13<15:37,  1.33it/s][A
      2%|▏         | 19/1261 [00:14<15:16,  1.36it/s][A
      2%|▏         | 20/1261 [00:15<15:08,  1.37it/s][A
      2%|▏         | 21/1261 [00:16<15:56,  1.30it/s][A
      2%|▏         | 22/1261 [00:16<16:01,  1.29it/s][A
      2%|▏         | 23/1261 [00:17<16:43,  1.23it/s][A
      2%|▏         | 24/1261 [00:18<18:17,  1.13it/s][A
      2%|▏         | 25/1261 [00:19<17:32,  1.17it/s][A
      2%|▏         | 26/1261 [00:20<16:37,  1.24it/s][A
      2%|▏         | 27/1261 [00:20<15:53,  1.29it/s][A
      2%|▏         | 28/1261 [00:21<15:26,  1.33it/s][A
      2%|▏         | 29/1261 [00:22<15:04,  1.36it/s][A
      2%|▏         | 30/1261 [00:23<14:52,  1.38it/s][A
      2%|▏         | 31/1261 [00:23<14:42,  1.39it/s][A
      3%|▎         | 32/1261 [00:24<14:32,  1.41it/s][A
      3%|▎         | 33/1261 [00:25<14:23,  1.42it/s][A
      3%|▎         | 34/1261 [00:25<14:23,  1.42it/s][A
      3%|▎         | 35/1261 [00:26<14:16,  1.43it/s][A
      3%|▎         | 36/1261 [00:27<14:14,  1.43it/s][A
      3%|▎         | 37/1261 [00:27<14:10,  1.44it/s][A
      3%|▎         | 38/1261 [00:28<14:06,  1.45it/s][A
      3%|▎         | 39/1261 [00:29<14:03,  1.45it/s][A
      3%|▎         | 40/1261 [00:29<14:01,  1.45it/s][A
      3%|▎         | 41/1261 [00:30<14:00,  1.45it/s][A
      3%|▎         | 42/1261 [00:31<13:58,  1.45it/s][A
      3%|▎         | 43/1261 [00:32<14:10,  1.43it/s][A
      3%|▎         | 44/1261 [00:32<14:08,  1.43it/s][A
      4%|▎         | 45/1261 [00:33<14:04,  1.44it/s][A
      4%|▎         | 46/1261 [00:34<14:01,  1.44it/s][A
      4%|▎         | 47/1261 [00:34<13:58,  1.45it/s][A
      4%|▍         | 48/1261 [00:35<13:56,  1.45it/s][A
      4%|▍         | 49/1261 [00:36<13:56,  1.45it/s][A
      4%|▍         | 50/1261 [00:36<14:00,  1.44it/s][A
      4%|▍         | 51/1261 [00:37<13:58,  1.44it/s][A
      4%|▍         | 52/1261 [00:38<13:56,  1.45it/s][A
      4%|▍         | 53/1261 [00:38<14:01,  1.44it/s][A
      4%|▍         | 54/1261 [00:39<14:18,  1.41it/s][A
      4%|▍         | 55/1261 [00:40<14:12,  1.41it/s][A
      4%|▍         | 56/1261 [00:41<14:14,  1.41it/s][A
      5%|▍         | 57/1261 [00:41<14:28,  1.39it/s][A
      5%|▍         | 58/1261 [00:42<15:44,  1.27it/s][A
      5%|▍         | 59/1261 [00:43<15:37,  1.28it/s][A
      5%|▍         | 60/1261 [00:44<15:17,  1.31it/s][A
      5%|▍         | 61/1261 [00:45<15:06,  1.32it/s][A
      5%|▍         | 62/1261 [00:45<14:50,  1.35it/s][A
      5%|▍         | 63/1261 [00:46<14:38,  1.36it/s][A
      5%|▌         | 64/1261 [00:47<14:43,  1.36it/s][A
      5%|▌         | 65/1261 [00:47<14:31,  1.37it/s][A
      5%|▌         | 66/1261 [00:48<14:22,  1.39it/s][A
      5%|▌         | 67/1261 [00:49<14:31,  1.37it/s][A
      5%|▌         | 68/1261 [00:50<14:51,  1.34it/s][A
      5%|▌         | 69/1261 [00:50<14:35,  1.36it/s][A
      6%|▌         | 70/1261 [00:51<14:17,  1.39it/s][A
      6%|▌         | 71/1261 [00:52<14:10,  1.40it/s][A
      6%|▌         | 72/1261 [00:52<14:13,  1.39it/s][A
      6%|▌         | 73/1261 [00:53<14:11,  1.39it/s][A
      6%|▌         | 74/1261 [00:54<14:09,  1.40it/s][A
      6%|▌         | 75/1261 [00:55<14:13,  1.39it/s][A
      6%|▌         | 76/1261 [00:55<14:14,  1.39it/s][A
      6%|▌         | 77/1261 [00:56<14:12,  1.39it/s][A
      6%|▌         | 78/1261 [00:57<14:36,  1.35it/s][A
      6%|▋         | 79/1261 [00:58<14:55,  1.32it/s][A
      6%|▋         | 80/1261 [00:58<15:09,  1.30it/s][A
      6%|▋         | 81/1261 [00:59<14:56,  1.32it/s][A
      7%|▋         | 82/1261 [01:00<14:43,  1.33it/s][A
      7%|▋         | 83/1261 [01:01<14:37,  1.34it/s][A
      7%|▋         | 84/1261 [01:01<14:34,  1.35it/s][A
      7%|▋         | 85/1261 [01:02<14:20,  1.37it/s][A
      7%|▋         | 86/1261 [01:03<14:23,  1.36it/s][A
      7%|▋         | 87/1261 [01:04<14:11,  1.38it/s][A
      7%|▋         | 88/1261 [01:04<14:00,  1.40it/s][A
      7%|▋         | 89/1261 [01:05<13:49,  1.41it/s][A
      7%|▋         | 90/1261 [01:06<13:43,  1.42it/s][A
      7%|▋         | 91/1261 [01:06<13:39,  1.43it/s][A
      7%|▋         | 92/1261 [01:07<13:34,  1.44it/s][A
      7%|▋         | 93/1261 [01:08<13:31,  1.44it/s][A
      7%|▋         | 94/1261 [01:08<13:30,  1.44it/s][A
      8%|▊         | 95/1261 [01:09<13:41,  1.42it/s][A
      8%|▊         | 96/1261 [01:10<13:36,  1.43it/s][A
      8%|▊         | 97/1261 [01:11<13:35,  1.43it/s][A
      8%|▊         | 98/1261 [01:11<13:30,  1.44it/s][A
      8%|▊         | 99/1261 [01:12<13:27,  1.44it/s][A
      8%|▊         | 100/1261 [01:13<13:28,  1.44it/s][A
      8%|▊         | 101/1261 [01:13<13:26,  1.44it/s][A
      8%|▊         | 102/1261 [01:14<13:36,  1.42it/s][A
      8%|▊         | 103/1261 [01:15<13:36,  1.42it/s][A
      8%|▊         | 104/1261 [01:15<13:44,  1.40it/s][A
      8%|▊         | 105/1261 [01:16<14:01,  1.37it/s][A
      8%|▊         | 106/1261 [01:17<14:20,  1.34it/s][A
      8%|▊         | 107/1261 [01:18<14:15,  1.35it/s][A
      9%|▊         | 108/1261 [01:19<14:35,  1.32it/s][A
      9%|▊         | 109/1261 [01:19<14:20,  1.34it/s][A
      9%|▊         | 110/1261 [01:20<14:03,  1.37it/s][A
      9%|▉         | 111/1261 [01:21<14:02,  1.37it/s][A
      9%|▉         | 112/1261 [01:21<14:03,  1.36it/s][A
      9%|▉         | 113/1261 [01:22<14:40,  1.30it/s][A
      9%|▉         | 114/1261 [01:23<14:51,  1.29it/s][A
      9%|▉         | 115/1261 [01:24<15:20,  1.24it/s][A
      9%|▉         | 116/1261 [01:25<15:08,  1.26it/s][A
      9%|▉         | 117/1261 [01:25<14:47,  1.29it/s][A
      9%|▉         | 118/1261 [01:26<14:16,  1.33it/s][A
      9%|▉         | 119/1261 [01:27<13:57,  1.36it/s][A
     10%|▉         | 120/1261 [01:28<13:41,  1.39it/s][A
     10%|▉         | 121/1261 [01:28<13:29,  1.41it/s][A
     10%|▉         | 122/1261 [01:29<13:20,  1.42it/s][A
     10%|▉         | 123/1261 [01:30<13:12,  1.44it/s][A
     10%|▉         | 124/1261 [01:30<13:08,  1.44it/s][A
     10%|▉         | 125/1261 [01:31<13:08,  1.44it/s][A
     10%|▉         | 126/1261 [01:32<13:08,  1.44it/s][A
     10%|█         | 127/1261 [01:32<13:07,  1.44it/s][A
     10%|█         | 128/1261 [01:33<13:03,  1.45it/s][A
     10%|█         | 129/1261 [01:34<13:01,  1.45it/s][A
     10%|█         | 130/1261 [01:34<12:59,  1.45it/s][A
     10%|█         | 131/1261 [01:35<12:58,  1.45it/s][A
     10%|█         | 132/1261 [01:36<12:56,  1.45it/s][A
     11%|█         | 133/1261 [01:36<12:55,  1.45it/s][A
     11%|█         | 134/1261 [01:37<12:52,  1.46it/s][A
     11%|█         | 135/1261 [01:38<12:53,  1.46it/s][A
     11%|█         | 136/1261 [01:39<12:53,  1.46it/s][A
     11%|█         | 137/1261 [01:39<12:57,  1.45it/s][A
     11%|█         | 138/1261 [01:40<12:53,  1.45it/s][A
     11%|█         | 139/1261 [01:41<12:53,  1.45it/s][A
     11%|█         | 140/1261 [01:41<12:53,  1.45it/s][A
     11%|█         | 141/1261 [01:42<12:55,  1.44it/s][A
     11%|█▏        | 142/1261 [01:43<12:56,  1.44it/s][A
     11%|█▏        | 143/1261 [01:43<12:52,  1.45it/s][A
     11%|█▏        | 144/1261 [01:44<12:58,  1.44it/s][A
     11%|█▏        | 145/1261 [01:45<13:18,  1.40it/s][A
     12%|█▏        | 146/1261 [01:46<13:24,  1.39it/s][A
     12%|█▏        | 147/1261 [01:46<13:17,  1.40it/s][A
     12%|█▏        | 148/1261 [01:47<13:21,  1.39it/s][A
     12%|█▏        | 149/1261 [01:48<13:24,  1.38it/s][A
     12%|█▏        | 150/1261 [01:49<14:22,  1.29it/s][A
     12%|█▏        | 151/1261 [01:49<14:19,  1.29it/s][A
     12%|█▏        | 152/1261 [01:50<14:32,  1.27it/s][A
     12%|█▏        | 153/1261 [01:51<15:30,  1.19it/s][A
     12%|█▏        | 154/1261 [01:52<15:56,  1.16it/s][A
     12%|█▏        | 155/1261 [01:53<15:54,  1.16it/s][A
     12%|█▏        | 156/1261 [01:54<15:43,  1.17it/s][A
     12%|█▏        | 157/1261 [01:55<15:40,  1.17it/s][A
     13%|█▎        | 158/1261 [01:55<15:35,  1.18it/s][A
     13%|█▎        | 159/1261 [01:56<15:05,  1.22it/s][A
     13%|█▎        | 160/1261 [01:57<14:39,  1.25it/s][A
     13%|█▎        | 161/1261 [01:58<14:13,  1.29it/s][A
     13%|█▎        | 162/1261 [01:58<13:52,  1.32it/s][A
     13%|█▎        | 163/1261 [01:59<13:28,  1.36it/s][A
     13%|█▎        | 164/1261 [02:00<13:15,  1.38it/s][A
     13%|█▎        | 165/1261 [02:01<13:06,  1.39it/s][A
     13%|█▎        | 166/1261 [02:01<13:04,  1.40it/s][A
     13%|█▎        | 167/1261 [02:02<12:54,  1.41it/s][A
     13%|█▎        | 168/1261 [02:03<13:04,  1.39it/s][A
     13%|█▎        | 169/1261 [02:03<13:00,  1.40it/s][A
     13%|█▎        | 170/1261 [02:04<12:59,  1.40it/s][A
     14%|█▎        | 171/1261 [02:05<13:00,  1.40it/s][A
     14%|█▎        | 172/1261 [02:06<13:00,  1.40it/s][A
     14%|█▎        | 173/1261 [02:06<13:26,  1.35it/s][A
     14%|█▍        | 174/1261 [02:07<13:22,  1.35it/s][A
     14%|█▍        | 175/1261 [02:08<13:47,  1.31it/s][A
     14%|█▍        | 176/1261 [02:09<14:00,  1.29it/s][A
     14%|█▍        | 177/1261 [02:09<14:18,  1.26it/s][A
     14%|█▍        | 178/1261 [02:10<14:25,  1.25it/s][A
     14%|█▍        | 179/1261 [02:11<14:56,  1.21it/s][A
     14%|█▍        | 180/1261 [02:12<15:14,  1.18it/s][A
     14%|█▍        | 181/1261 [02:13<15:43,  1.14it/s][A
     14%|█▍        | 182/1261 [02:14<15:36,  1.15it/s][A
     15%|█▍        | 183/1261 [02:15<14:46,  1.22it/s][A
     15%|█▍        | 184/1261 [02:15<14:15,  1.26it/s][A
     15%|█▍        | 185/1261 [02:16<14:14,  1.26it/s][A
     15%|█▍        | 186/1261 [02:17<14:22,  1.25it/s][A
     15%|█▍        | 187/1261 [02:18<14:25,  1.24it/s][A
     15%|█▍        | 188/1261 [02:19<14:44,  1.21it/s][A
     15%|█▍        | 189/1261 [02:19<14:20,  1.25it/s][A
     15%|█▌        | 190/1261 [02:20<14:12,  1.26it/s][A
     15%|█▌        | 191/1261 [02:21<13:48,  1.29it/s][A
     15%|█▌        | 192/1261 [02:22<13:41,  1.30it/s][A
     15%|█▌        | 193/1261 [02:22<13:37,  1.31it/s][A
     15%|█▌        | 194/1261 [02:23<13:39,  1.30it/s][A
     15%|█▌        | 195/1261 [02:24<13:43,  1.29it/s][A
     16%|█▌        | 196/1261 [02:25<13:44,  1.29it/s][A
     16%|█▌        | 197/1261 [02:26<13:43,  1.29it/s][A
     16%|█▌        | 198/1261 [02:26<13:49,  1.28it/s][A
     16%|█▌        | 199/1261 [02:27<13:38,  1.30it/s][A
     16%|█▌        | 200/1261 [02:28<13:44,  1.29it/s][A
     16%|█▌        | 201/1261 [02:29<14:23,  1.23it/s][A
     16%|█▌        | 202/1261 [02:30<14:13,  1.24it/s][A
     16%|█▌        | 203/1261 [02:30<13:36,  1.30it/s][A
     16%|█▌        | 204/1261 [02:31<14:13,  1.24it/s][A
     16%|█▋        | 205/1261 [02:32<14:11,  1.24it/s][A
     16%|█▋        | 206/1261 [02:33<15:08,  1.16it/s][A
     16%|█▋        | 207/1261 [02:34<14:59,  1.17it/s][A
     16%|█▋        | 208/1261 [02:34<14:19,  1.23it/s][A
     17%|█▋        | 209/1261 [02:35<13:40,  1.28it/s][A
     17%|█▋        | 210/1261 [02:36<13:36,  1.29it/s][A
     17%|█▋        | 211/1261 [02:37<13:45,  1.27it/s][A
     17%|█▋        | 212/1261 [02:37<13:15,  1.32it/s][A
     17%|█▋        | 213/1261 [02:38<13:19,  1.31it/s][A
     17%|█▋        | 214/1261 [02:39<14:14,  1.22it/s][A
     17%|█▋        | 215/1261 [02:40<13:54,  1.25it/s][A
     17%|█▋        | 216/1261 [02:41<13:21,  1.30it/s][A
     17%|█▋        | 217/1261 [02:41<12:57,  1.34it/s][A
     17%|█▋        | 218/1261 [02:42<12:37,  1.38it/s][A
     17%|█▋        | 219/1261 [02:43<12:30,  1.39it/s][A
     17%|█▋        | 220/1261 [02:43<12:29,  1.39it/s][A
     18%|█▊        | 221/1261 [02:44<12:38,  1.37it/s][A
     18%|█▊        | 222/1261 [02:45<12:40,  1.37it/s][A
     18%|█▊        | 223/1261 [02:46<12:52,  1.34it/s][A
     18%|█▊        | 224/1261 [02:46<13:10,  1.31it/s][A
     18%|█▊        | 225/1261 [02:47<13:13,  1.31it/s][A
     18%|█▊        | 226/1261 [02:48<13:19,  1.29it/s][A
     18%|█▊        | 227/1261 [02:49<12:57,  1.33it/s][A
     18%|█▊        | 228/1261 [02:49<12:44,  1.35it/s][A
     18%|█▊        | 229/1261 [02:50<13:07,  1.31it/s][A
     18%|█▊        | 230/1261 [02:51<12:56,  1.33it/s][A
     18%|█▊        | 231/1261 [02:52<12:43,  1.35it/s][A
     18%|█▊        | 232/1261 [02:52<12:49,  1.34it/s][A
     18%|█▊        | 233/1261 [02:53<12:47,  1.34it/s][A
     19%|█▊        | 234/1261 [02:54<12:32,  1.37it/s][A
     19%|█▊        | 235/1261 [02:55<12:19,  1.39it/s][A
     19%|█▊        | 236/1261 [02:55<12:10,  1.40it/s][A
     19%|█▉        | 237/1261 [02:56<12:08,  1.41it/s][A
     19%|█▉        | 238/1261 [02:57<12:08,  1.40it/s][A
     19%|█▉        | 239/1261 [02:57<12:05,  1.41it/s][A
     19%|█▉        | 240/1261 [02:58<12:02,  1.41it/s][A
     19%|█▉        | 241/1261 [02:59<12:03,  1.41it/s][A
     19%|█▉        | 242/1261 [03:00<12:02,  1.41it/s][A
     19%|█▉        | 243/1261 [03:00<11:55,  1.42it/s][A
     19%|█▉        | 244/1261 [03:01<11:53,  1.43it/s][A
     19%|█▉        | 245/1261 [03:02<11:50,  1.43it/s][A
     20%|█▉        | 246/1261 [03:02<11:52,  1.42it/s][A
     20%|█▉        | 247/1261 [03:03<11:50,  1.43it/s][A
     20%|█▉        | 248/1261 [03:04<11:48,  1.43it/s][A
     20%|█▉        | 249/1261 [03:04<11:46,  1.43it/s][A
     20%|█▉        | 250/1261 [03:05<11:42,  1.44it/s][A
     20%|█▉        | 251/1261 [03:06<11:41,  1.44it/s][A
     20%|█▉        | 252/1261 [03:06<11:39,  1.44it/s][A
     20%|██        | 253/1261 [03:07<11:36,  1.45it/s][A
     20%|██        | 254/1261 [03:08<11:36,  1.45it/s][A
     20%|██        | 255/1261 [03:09<12:02,  1.39it/s][A
     20%|██        | 256/1261 [03:09<11:55,  1.41it/s][A
     20%|██        | 257/1261 [03:10<11:50,  1.41it/s][A
     20%|██        | 258/1261 [03:11<11:44,  1.42it/s][A
     21%|██        | 259/1261 [03:11<11:40,  1.43it/s][A
     21%|██        | 260/1261 [03:12<11:37,  1.44it/s][A
     21%|██        | 261/1261 [03:13<11:36,  1.44it/s][A
     21%|██        | 262/1261 [03:14<11:35,  1.44it/s][A
     21%|██        | 263/1261 [03:14<11:34,  1.44it/s][A
     21%|██        | 264/1261 [03:15<11:32,  1.44it/s][A
     21%|██        | 265/1261 [03:16<11:30,  1.44it/s][A
     21%|██        | 266/1261 [03:16<11:33,  1.43it/s][A
     21%|██        | 267/1261 [03:17<11:44,  1.41it/s][A
     21%|██▏       | 268/1261 [03:18<12:19,  1.34it/s][A
     21%|██▏       | 269/1261 [03:19<13:00,  1.27it/s][A
     21%|██▏       | 270/1261 [03:20<12:56,  1.28it/s][A
     21%|██▏       | 271/1261 [03:20<13:25,  1.23it/s][A
     22%|██▏       | 272/1261 [03:21<14:11,  1.16it/s][A
     22%|██▏       | 273/1261 [03:22<13:52,  1.19it/s][A
     22%|██▏       | 274/1261 [03:23<13:27,  1.22it/s][A
     22%|██▏       | 275/1261 [03:24<13:07,  1.25it/s][A
     22%|██▏       | 276/1261 [03:25<13:28,  1.22it/s][A
     22%|██▏       | 277/1261 [03:25<13:04,  1.25it/s][A
     22%|██▏       | 278/1261 [03:26<13:21,  1.23it/s][A
     22%|██▏       | 279/1261 [03:27<13:29,  1.21it/s][A
     22%|██▏       | 280/1261 [03:28<13:31,  1.21it/s][A
     22%|██▏       | 281/1261 [03:29<13:06,  1.25it/s][A
     22%|██▏       | 282/1261 [03:29<12:38,  1.29it/s][A
     22%|██▏       | 283/1261 [03:30<12:16,  1.33it/s][A
     23%|██▎       | 284/1261 [03:31<12:13,  1.33it/s][A
     23%|██▎       | 285/1261 [03:31<12:09,  1.34it/s][A
     23%|██▎       | 286/1261 [03:32<12:08,  1.34it/s][A
     23%|██▎       | 287/1261 [03:33<11:50,  1.37it/s][A
     23%|██▎       | 288/1261 [03:34<11:46,  1.38it/s][A
     23%|██▎       | 289/1261 [03:34<11:57,  1.35it/s][A
     23%|██▎       | 290/1261 [03:35<11:58,  1.35it/s][A
     23%|██▎       | 291/1261 [03:36<12:01,  1.35it/s][A
     23%|██▎       | 292/1261 [03:37<11:48,  1.37it/s][A
     23%|██▎       | 293/1261 [03:37<11:38,  1.39it/s][A
     23%|██▎       | 294/1261 [03:38<11:30,  1.40it/s][A
     23%|██▎       | 295/1261 [03:39<11:26,  1.41it/s][A
     23%|██▎       | 296/1261 [03:39<11:24,  1.41it/s][A
     24%|██▎       | 297/1261 [03:40<11:32,  1.39it/s][A
     24%|██▎       | 298/1261 [03:41<11:46,  1.36it/s][A
     24%|██▎       | 299/1261 [03:42<12:14,  1.31it/s][A
     24%|██▍       | 300/1261 [03:43<12:22,  1.29it/s][A
     24%|██▍       | 301/1261 [03:43<12:14,  1.31it/s][A
     24%|██▍       | 302/1261 [03:44<12:22,  1.29it/s][A
     24%|██▍       | 303/1261 [03:45<12:12,  1.31it/s][A
     24%|██▍       | 304/1261 [03:46<12:13,  1.30it/s][A
     24%|██▍       | 305/1261 [03:46<12:18,  1.29it/s][A
     24%|██▍       | 306/1261 [03:47<12:28,  1.28it/s][A
     24%|██▍       | 307/1261 [03:48<12:27,  1.28it/s][A
     24%|██▍       | 308/1261 [03:49<12:15,  1.30it/s][A
     25%|██▍       | 309/1261 [03:49<12:08,  1.31it/s][A
     25%|██▍       | 310/1261 [03:50<12:01,  1.32it/s][A
     25%|██▍       | 311/1261 [03:51<11:56,  1.33it/s][A
     25%|██▍       | 312/1261 [03:52<11:44,  1.35it/s][A
     25%|██▍       | 313/1261 [03:52<11:39,  1.36it/s][A
     25%|██▍       | 314/1261 [03:53<11:42,  1.35it/s][A
     25%|██▍       | 315/1261 [03:54<11:41,  1.35it/s][A
     25%|██▌       | 316/1261 [03:55<11:46,  1.34it/s][A
     25%|██▌       | 317/1261 [03:55<11:44,  1.34it/s][A
     25%|██▌       | 318/1261 [03:56<12:07,  1.30it/s][A
     25%|██▌       | 319/1261 [03:57<12:18,  1.27it/s][A
     25%|██▌       | 320/1261 [03:58<12:08,  1.29it/s][A
     25%|██▌       | 321/1261 [03:59<11:58,  1.31it/s][A
     26%|██▌       | 322/1261 [03:59<11:43,  1.34it/s][A
     26%|██▌       | 323/1261 [04:00<11:28,  1.36it/s][A
     26%|██▌       | 324/1261 [04:01<11:17,  1.38it/s][A
     26%|██▌       | 325/1261 [04:01<11:09,  1.40it/s][A
     26%|██▌       | 326/1261 [04:02<11:02,  1.41it/s][A
     26%|██▌       | 327/1261 [04:03<11:00,  1.41it/s][A
     26%|██▌       | 328/1261 [04:03<10:56,  1.42it/s][A
     26%|██▌       | 329/1261 [04:04<10:53,  1.43it/s][A
     26%|██▌       | 330/1261 [04:05<10:54,  1.42it/s][A
     26%|██▌       | 331/1261 [04:06<10:50,  1.43it/s][A
     26%|██▋       | 332/1261 [04:06<11:05,  1.40it/s][A
     26%|██▋       | 333/1261 [04:07<11:15,  1.37it/s][A
     26%|██▋       | 334/1261 [04:08<11:28,  1.35it/s][A
     27%|██▋       | 335/1261 [04:09<11:38,  1.33it/s][A
     27%|██▋       | 336/1261 [04:09<11:40,  1.32it/s][A
     27%|██▋       | 337/1261 [04:10<11:34,  1.33it/s][A
     27%|██▋       | 338/1261 [04:11<11:47,  1.30it/s][A
     27%|██▋       | 339/1261 [04:12<11:38,  1.32it/s][A
     27%|██▋       | 340/1261 [04:12<11:26,  1.34it/s][A
     27%|██▋       | 341/1261 [04:13<11:18,  1.36it/s][A
     27%|██▋       | 342/1261 [04:14<11:06,  1.38it/s][A
     27%|██▋       | 343/1261 [04:14<10:56,  1.40it/s][A
     27%|██▋       | 344/1261 [04:15<10:58,  1.39it/s][A
     27%|██▋       | 345/1261 [04:16<10:58,  1.39it/s][A
     27%|██▋       | 346/1261 [04:17<10:59,  1.39it/s][A
     28%|██▊       | 347/1261 [04:17<11:06,  1.37it/s][A
     28%|██▊       | 348/1261 [04:18<11:10,  1.36it/s][A
     28%|██▊       | 349/1261 [04:19<11:19,  1.34it/s][A
     28%|██▊       | 350/1261 [04:20<11:24,  1.33it/s][A
     28%|██▊       | 351/1261 [04:20<11:20,  1.34it/s][A
     28%|██▊       | 352/1261 [04:21<11:23,  1.33it/s][A
     28%|██▊       | 353/1261 [04:22<11:24,  1.33it/s][A
     28%|██▊       | 354/1261 [04:23<11:21,  1.33it/s][A
     28%|██▊       | 355/1261 [04:23<11:26,  1.32it/s][A
     28%|██▊       | 356/1261 [04:24<11:34,  1.30it/s][A
     28%|██▊       | 357/1261 [04:25<11:49,  1.27it/s][A
     28%|██▊       | 358/1261 [04:26<11:38,  1.29it/s][A
     28%|██▊       | 359/1261 [04:27<11:24,  1.32it/s][A
     29%|██▊       | 360/1261 [04:27<11:25,  1.31it/s][A
     29%|██▊       | 361/1261 [04:28<11:19,  1.32it/s][A
     29%|██▊       | 362/1261 [04:29<11:21,  1.32it/s][A
     29%|██▉       | 363/1261 [04:30<11:22,  1.32it/s][A
     29%|██▉       | 364/1261 [04:30<11:10,  1.34it/s][A
     29%|██▉       | 365/1261 [04:31<11:23,  1.31it/s][A
     29%|██▉       | 366/1261 [04:32<11:07,  1.34it/s][A
     29%|██▉       | 367/1261 [04:32<10:52,  1.37it/s][A
     29%|██▉       | 368/1261 [04:33<10:52,  1.37it/s][A
     29%|██▉       | 369/1261 [04:34<10:50,  1.37it/s][A
     29%|██▉       | 370/1261 [04:35<10:45,  1.38it/s][A
     29%|██▉       | 371/1261 [04:35<10:34,  1.40it/s][A
     30%|██▉       | 372/1261 [04:36<10:39,  1.39it/s][A
     30%|██▉       | 373/1261 [04:37<10:59,  1.35it/s][A
     30%|██▉       | 374/1261 [04:38<11:08,  1.33it/s][A
     30%|██▉       | 375/1261 [04:38<11:01,  1.34it/s][A
     30%|██▉       | 376/1261 [04:39<11:05,  1.33it/s][A
     30%|██▉       | 377/1261 [04:40<10:59,  1.34it/s][A
     30%|██▉       | 378/1261 [04:41<11:02,  1.33it/s][A
     30%|███       | 379/1261 [04:41<11:10,  1.32it/s][A
     30%|███       | 380/1261 [04:42<11:10,  1.31it/s][A
     30%|███       | 381/1261 [04:43<11:03,  1.33it/s][A
     30%|███       | 382/1261 [04:44<10:55,  1.34it/s][A
     30%|███       | 383/1261 [04:44<10:38,  1.37it/s][A
     30%|███       | 384/1261 [04:45<10:28,  1.40it/s][A
     31%|███       | 385/1261 [04:46<10:50,  1.35it/s][A
     31%|███       | 386/1261 [04:47<10:53,  1.34it/s][A
     31%|███       | 387/1261 [04:47<10:39,  1.37it/s][A
     31%|███       | 388/1261 [04:48<10:31,  1.38it/s][A
     31%|███       | 389/1261 [04:49<10:28,  1.39it/s][A
     31%|███       | 390/1261 [04:49<10:25,  1.39it/s][A
     31%|███       | 391/1261 [04:50<10:27,  1.39it/s][A
     31%|███       | 392/1261 [04:51<10:33,  1.37it/s][A
     31%|███       | 393/1261 [04:52<10:30,  1.38it/s][A
     31%|███       | 394/1261 [04:52<10:21,  1.40it/s][A
     31%|███▏      | 395/1261 [04:53<10:34,  1.36it/s][A
     31%|███▏      | 396/1261 [04:54<10:48,  1.33it/s][A
     31%|███▏      | 397/1261 [04:55<11:05,  1.30it/s][A
     32%|███▏      | 398/1261 [04:56<11:41,  1.23it/s][A
     32%|███▏      | 399/1261 [04:56<11:54,  1.21it/s][A
     32%|███▏      | 400/1261 [04:57<11:54,  1.21it/s][A
     32%|███▏      | 401/1261 [04:58<11:33,  1.24it/s][A
     32%|███▏      | 402/1261 [04:59<11:24,  1.25it/s][A
     32%|███▏      | 403/1261 [05:00<11:30,  1.24it/s][A
     32%|███▏      | 404/1261 [05:00<11:42,  1.22it/s][A
     32%|███▏      | 405/1261 [05:01<11:29,  1.24it/s][A
     32%|███▏      | 406/1261 [05:02<11:09,  1.28it/s][A
     32%|███▏      | 407/1261 [05:03<11:00,  1.29it/s][A
     32%|███▏      | 408/1261 [05:04<10:58,  1.30it/s][A
     32%|███▏      | 409/1261 [05:04<10:59,  1.29it/s][A
     33%|███▎      | 410/1261 [05:05<10:57,  1.29it/s][A
     33%|███▎      | 411/1261 [05:06<11:11,  1.27it/s][A
     33%|███▎      | 412/1261 [05:07<11:22,  1.24it/s][A
     33%|███▎      | 413/1261 [05:07<11:11,  1.26it/s][A
     33%|███▎      | 414/1261 [05:08<11:02,  1.28it/s][A
     33%|███▎      | 415/1261 [05:09<11:14,  1.25it/s][A
     33%|███▎      | 416/1261 [05:10<11:13,  1.25it/s][A
     33%|███▎      | 417/1261 [05:11<11:00,  1.28it/s][A
     33%|███▎      | 418/1261 [05:11<10:41,  1.31it/s][A
     33%|███▎      | 419/1261 [05:12<10:26,  1.34it/s][A
     33%|███▎      | 420/1261 [05:13<10:20,  1.36it/s][A
     33%|███▎      | 421/1261 [05:14<10:23,  1.35it/s][A
     33%|███▎      | 422/1261 [05:14<10:55,  1.28it/s][A
     34%|███▎      | 423/1261 [05:15<11:11,  1.25it/s][A
     34%|███▎      | 424/1261 [05:16<11:14,  1.24it/s][A
     34%|███▎      | 425/1261 [05:17<10:58,  1.27it/s][A
     34%|███▍      | 426/1261 [05:18<10:42,  1.30it/s][A
     34%|███▍      | 427/1261 [05:18<10:43,  1.30it/s][A
     34%|███▍      | 428/1261 [05:19<10:34,  1.31it/s][A
     34%|███▍      | 429/1261 [05:20<10:30,  1.32it/s][A
     34%|███▍      | 430/1261 [05:21<10:21,  1.34it/s][A
     34%|███▍      | 431/1261 [05:21<10:29,  1.32it/s][A
     34%|███▍      | 432/1261 [05:22<10:17,  1.34it/s][A
     34%|███▍      | 433/1261 [05:23<10:24,  1.33it/s][A
     34%|███▍      | 434/1261 [05:23<10:07,  1.36it/s][A
     34%|███▍      | 435/1261 [05:24<09:58,  1.38it/s][A
     35%|███▍      | 436/1261 [05:25<09:52,  1.39it/s][A
     35%|███▍      | 437/1261 [05:26<09:52,  1.39it/s][A
     35%|███▍      | 438/1261 [05:26<09:50,  1.39it/s][A
     35%|███▍      | 439/1261 [05:27<09:52,  1.39it/s][A
     35%|███▍      | 440/1261 [05:28<09:49,  1.39it/s][A
     35%|███▍      | 441/1261 [05:28<09:41,  1.41it/s][A
     35%|███▌      | 442/1261 [05:29<09:36,  1.42it/s][A
     35%|███▌      | 443/1261 [05:30<09:42,  1.40it/s][A
     35%|███▌      | 444/1261 [05:31<09:47,  1.39it/s][A
     35%|███▌      | 445/1261 [05:31<09:48,  1.39it/s][A
     35%|███▌      | 446/1261 [05:32<09:47,  1.39it/s][A
     35%|███▌      | 447/1261 [05:33<09:44,  1.39it/s][A
     36%|███▌      | 448/1261 [05:33<09:46,  1.39it/s][A
     36%|███▌      | 449/1261 [05:34<09:54,  1.37it/s][A
     36%|███▌      | 450/1261 [05:35<09:59,  1.35it/s][A
     36%|███▌      | 451/1261 [05:36<09:57,  1.35it/s][A
     36%|███▌      | 452/1261 [05:36<10:00,  1.35it/s][A
     36%|███▌      | 453/1261 [05:37<09:55,  1.36it/s][A
     36%|███▌      | 454/1261 [05:38<09:52,  1.36it/s][A
     36%|███▌      | 455/1261 [05:39<10:07,  1.33it/s][A
     36%|███▌      | 456/1261 [05:39<10:01,  1.34it/s][A
     36%|███▌      | 457/1261 [05:40<09:59,  1.34it/s][A
     36%|███▋      | 458/1261 [05:41<10:02,  1.33it/s][A
     36%|███▋      | 459/1261 [05:42<10:11,  1.31it/s][A
     36%|███▋      | 460/1261 [05:43<10:22,  1.29it/s][A
     37%|███▋      | 461/1261 [05:43<10:24,  1.28it/s][A
     37%|███▋      | 462/1261 [05:44<10:26,  1.28it/s][A
     37%|███▋      | 463/1261 [05:45<10:05,  1.32it/s][A
     37%|███▋      | 464/1261 [05:46<09:48,  1.35it/s][A
     37%|███▋      | 465/1261 [05:46<09:39,  1.37it/s][A
     37%|███▋      | 466/1261 [05:47<09:34,  1.38it/s][A
     37%|███▋      | 467/1261 [05:48<09:28,  1.40it/s][A
     37%|███▋      | 468/1261 [05:48<09:26,  1.40it/s][A
     37%|███▋      | 469/1261 [05:49<09:24,  1.40it/s][A
     37%|███▋      | 470/1261 [05:50<09:19,  1.41it/s][A
     37%|███▋      | 471/1261 [05:50<09:15,  1.42it/s][A
     37%|███▋      | 472/1261 [05:51<09:13,  1.43it/s][A
     38%|███▊      | 473/1261 [05:52<09:11,  1.43it/s][A
     38%|███▊      | 474/1261 [05:53<09:13,  1.42it/s][A
     38%|███▊      | 475/1261 [05:53<09:11,  1.43it/s][A
     38%|███▊      | 476/1261 [05:54<09:08,  1.43it/s][A
     38%|███▊      | 477/1261 [05:55<09:07,  1.43it/s][A
     38%|███▊      | 478/1261 [05:55<09:05,  1.44it/s][A
     38%|███▊      | 479/1261 [05:56<09:04,  1.43it/s][A
     38%|███▊      | 480/1261 [05:57<09:03,  1.44it/s][A
     38%|███▊      | 481/1261 [05:57<09:02,  1.44it/s][A
     38%|███▊      | 482/1261 [05:58<09:02,  1.44it/s][A
     38%|███▊      | 483/1261 [05:59<09:01,  1.44it/s][A
     38%|███▊      | 484/1261 [06:00<08:59,  1.44it/s][A
     38%|███▊      | 485/1261 [06:00<08:58,  1.44it/s][A
     39%|███▊      | 486/1261 [06:01<08:57,  1.44it/s][A
     39%|███▊      | 487/1261 [06:02<08:59,  1.44it/s][A
     39%|███▊      | 488/1261 [06:02<09:00,  1.43it/s][A
     39%|███▉      | 489/1261 [06:03<08:59,  1.43it/s][A
     39%|███▉      | 490/1261 [06:04<08:59,  1.43it/s][A
     39%|███▉      | 491/1261 [06:04<08:55,  1.44it/s][A
     39%|███▉      | 492/1261 [06:05<08:56,  1.43it/s][A
     39%|███▉      | 493/1261 [06:06<08:56,  1.43it/s][A
     39%|███▉      | 494/1261 [06:06<08:53,  1.44it/s][A
     39%|███▉      | 495/1261 [06:07<08:53,  1.43it/s][A
     39%|███▉      | 496/1261 [06:08<08:57,  1.42it/s][A
     39%|███▉      | 497/1261 [06:09<08:59,  1.42it/s][A
     39%|███▉      | 498/1261 [06:09<08:58,  1.42it/s][A
     40%|███▉      | 499/1261 [06:10<09:00,  1.41it/s][A
     40%|███▉      | 500/1261 [06:11<08:57,  1.41it/s][A
     40%|███▉      | 501/1261 [06:11<08:56,  1.42it/s][A
     40%|███▉      | 502/1261 [06:12<08:59,  1.41it/s][A
     40%|███▉      | 503/1261 [06:13<08:55,  1.42it/s][A
     40%|███▉      | 504/1261 [06:14<08:52,  1.42it/s][A
     40%|████      | 505/1261 [06:14<08:50,  1.43it/s][A
     40%|████      | 506/1261 [06:15<08:48,  1.43it/s][A
     40%|████      | 507/1261 [06:16<08:47,  1.43it/s][A
     40%|████      | 508/1261 [06:16<08:45,  1.43it/s][A
     40%|████      | 509/1261 [06:17<08:43,  1.44it/s][A
     40%|████      | 510/1261 [06:18<08:42,  1.44it/s][A
     41%|████      | 511/1261 [06:18<08:41,  1.44it/s][A
     41%|████      | 512/1261 [06:19<08:47,  1.42it/s][A
     41%|████      | 513/1261 [06:20<08:51,  1.41it/s][A
     41%|████      | 514/1261 [06:21<08:49,  1.41it/s][A
     41%|████      | 515/1261 [06:21<08:46,  1.42it/s][A
     41%|████      | 516/1261 [06:22<08:49,  1.41it/s][A
     41%|████      | 517/1261 [06:23<08:49,  1.41it/s][A
     41%|████      | 518/1261 [06:23<08:45,  1.41it/s][A
     41%|████      | 519/1261 [06:24<08:42,  1.42it/s][A
     41%|████      | 520/1261 [06:25<08:40,  1.42it/s][A
     41%|████▏     | 521/1261 [06:26<08:37,  1.43it/s][A
     41%|████▏     | 522/1261 [06:26<08:36,  1.43it/s][A
     41%|████▏     | 523/1261 [06:27<08:35,  1.43it/s][A
     42%|████▏     | 524/1261 [06:28<08:33,  1.44it/s][A
     42%|████▏     | 525/1261 [06:28<08:31,  1.44it/s][A
     42%|████▏     | 526/1261 [06:29<08:31,  1.44it/s][A
     42%|████▏     | 527/1261 [06:30<08:30,  1.44it/s][A
     42%|████▏     | 528/1261 [06:30<08:29,  1.44it/s][A
     42%|████▏     | 529/1261 [06:31<08:28,  1.44it/s][A
     42%|████▏     | 530/1261 [06:32<08:28,  1.44it/s][A
     42%|████▏     | 531/1261 [06:32<08:29,  1.43it/s][A
     42%|████▏     | 532/1261 [06:33<08:28,  1.43it/s][A
     42%|████▏     | 533/1261 [06:34<08:28,  1.43it/s][A
     42%|████▏     | 534/1261 [06:35<08:26,  1.44it/s][A
     42%|████▏     | 535/1261 [06:35<08:25,  1.44it/s][A
     43%|████▎     | 536/1261 [06:36<08:24,  1.44it/s][A
     43%|████▎     | 537/1261 [06:37<08:24,  1.43it/s][A
     43%|████▎     | 538/1261 [06:37<08:24,  1.43it/s][A
     43%|████▎     | 539/1261 [06:38<08:23,  1.43it/s][A
     43%|████▎     | 540/1261 [06:39<08:25,  1.43it/s][A
     43%|████▎     | 541/1261 [06:39<08:24,  1.43it/s][A
     43%|████▎     | 542/1261 [06:40<08:22,  1.43it/s][A
     43%|████▎     | 543/1261 [06:41<08:21,  1.43it/s][A
     43%|████▎     | 544/1261 [06:42<08:20,  1.43it/s][A
     43%|████▎     | 545/1261 [06:42<08:21,  1.43it/s][A
     43%|████▎     | 546/1261 [06:43<08:20,  1.43it/s][A
     43%|████▎     | 547/1261 [06:44<08:19,  1.43it/s][A
     43%|████▎     | 548/1261 [06:44<08:19,  1.43it/s][A
     44%|████▎     | 549/1261 [06:45<08:18,  1.43it/s][A
     44%|████▎     | 550/1261 [06:46<08:17,  1.43it/s][A
     44%|████▎     | 551/1261 [06:46<08:15,  1.43it/s][A
     44%|████▍     | 552/1261 [06:47<08:15,  1.43it/s][A
     44%|████▍     | 553/1261 [06:48<08:14,  1.43it/s][A
     44%|████▍     | 554/1261 [06:49<08:12,  1.44it/s][A
     44%|████▍     | 555/1261 [06:49<08:12,  1.43it/s][A
     44%|████▍     | 556/1261 [06:50<08:11,  1.43it/s][A
     44%|████▍     | 557/1261 [06:51<08:10,  1.44it/s][A
     44%|████▍     | 558/1261 [06:51<08:10,  1.43it/s][A
     44%|████▍     | 559/1261 [06:52<08:10,  1.43it/s][A
     44%|████▍     | 560/1261 [06:53<08:13,  1.42it/s][A
     44%|████▍     | 561/1261 [06:53<08:12,  1.42it/s][A
     45%|████▍     | 562/1261 [06:54<08:12,  1.42it/s][A
     45%|████▍     | 563/1261 [06:55<08:11,  1.42it/s][A
     45%|████▍     | 564/1261 [06:56<08:10,  1.42it/s][A
     45%|████▍     | 565/1261 [06:56<08:10,  1.42it/s][A
     45%|████▍     | 566/1261 [06:57<08:10,  1.42it/s][A
     45%|████▍     | 567/1261 [06:58<08:09,  1.42it/s][A
     45%|████▌     | 568/1261 [06:58<08:09,  1.42it/s][A
     45%|████▌     | 569/1261 [06:59<08:08,  1.42it/s][A
     45%|████▌     | 570/1261 [07:00<08:07,  1.42it/s][A
     45%|████▌     | 571/1261 [07:00<08:04,  1.42it/s][A
     45%|████▌     | 572/1261 [07:01<08:04,  1.42it/s][A
     45%|████▌     | 573/1261 [07:02<08:04,  1.42it/s][A
     46%|████▌     | 574/1261 [07:03<08:06,  1.41it/s][A
     46%|████▌     | 575/1261 [07:03<08:04,  1.41it/s][A
     46%|████▌     | 576/1261 [07:04<08:04,  1.41it/s][A
     46%|████▌     | 577/1261 [07:05<08:01,  1.42it/s][A
     46%|████▌     | 578/1261 [07:05<08:00,  1.42it/s][A
     46%|████▌     | 579/1261 [07:06<08:00,  1.42it/s][A
     46%|████▌     | 580/1261 [07:07<07:59,  1.42it/s][A
     46%|████▌     | 581/1261 [07:08<07:57,  1.42it/s][A
     46%|████▌     | 582/1261 [07:08<07:56,  1.42it/s][A
     46%|████▌     | 583/1261 [07:09<08:05,  1.40it/s][A
     46%|████▋     | 584/1261 [07:10<08:04,  1.40it/s][A
     46%|████▋     | 585/1261 [07:10<07:59,  1.41it/s][A
     46%|████▋     | 586/1261 [07:11<07:58,  1.41it/s][A
     47%|████▋     | 587/1261 [07:12<07:57,  1.41it/s][A
     47%|████▋     | 588/1261 [07:13<07:58,  1.41it/s][A
     47%|████▋     | 589/1261 [07:13<07:56,  1.41it/s][A
     47%|████▋     | 590/1261 [07:14<07:55,  1.41it/s][A
     47%|████▋     | 591/1261 [07:15<07:52,  1.42it/s][A
     47%|████▋     | 592/1261 [07:15<07:50,  1.42it/s][A
     47%|████▋     | 593/1261 [07:16<07:50,  1.42it/s][A
     47%|████▋     | 594/1261 [07:17<07:49,  1.42it/s][A
     47%|████▋     | 595/1261 [07:17<07:48,  1.42it/s][A
     47%|████▋     | 596/1261 [07:18<07:47,  1.42it/s][A
     47%|████▋     | 597/1261 [07:19<07:48,  1.42it/s][A
     47%|████▋     | 598/1261 [07:20<07:46,  1.42it/s][A
     48%|████▊     | 599/1261 [07:20<07:46,  1.42it/s][A
     48%|████▊     | 600/1261 [07:21<07:46,  1.42it/s][A
     48%|████▊     | 601/1261 [07:22<07:45,  1.42it/s][A
     48%|████▊     | 602/1261 [07:22<07:46,  1.41it/s][A
     48%|████▊     | 603/1261 [07:23<07:45,  1.41it/s][A
     48%|████▊     | 604/1261 [07:24<07:46,  1.41it/s][A
     48%|████▊     | 605/1261 [07:25<07:44,  1.41it/s][A
     48%|████▊     | 606/1261 [07:25<07:44,  1.41it/s][A
     48%|████▊     | 607/1261 [07:26<07:42,  1.41it/s][A
     48%|████▊     | 608/1261 [07:27<07:41,  1.42it/s][A
     48%|████▊     | 609/1261 [07:27<07:39,  1.42it/s][A
     48%|████▊     | 610/1261 [07:28<07:39,  1.42it/s][A
     48%|████▊     | 611/1261 [07:29<07:37,  1.42it/s][A
     49%|████▊     | 612/1261 [07:29<07:34,  1.43it/s][A
     49%|████▊     | 613/1261 [07:30<07:35,  1.42it/s][A
     49%|████▊     | 614/1261 [07:31<07:36,  1.42it/s][A
     49%|████▉     | 615/1261 [07:32<07:34,  1.42it/s][A
     49%|████▉     | 616/1261 [07:32<07:35,  1.42it/s][A
     49%|████▉     | 617/1261 [07:33<07:35,  1.41it/s][A
     49%|████▉     | 618/1261 [07:34<07:33,  1.42it/s][A
     49%|████▉     | 619/1261 [07:34<07:30,  1.42it/s][A
     49%|████▉     | 620/1261 [07:35<07:29,  1.43it/s][A
     49%|████▉     | 621/1261 [07:36<07:28,  1.43it/s][A
     49%|████▉     | 622/1261 [07:36<07:27,  1.43it/s][A
     49%|████▉     | 623/1261 [07:37<07:27,  1.43it/s][A
     49%|████▉     | 624/1261 [07:38<07:28,  1.42it/s][A
     50%|████▉     | 625/1261 [07:39<07:25,  1.43it/s][A
     50%|████▉     | 626/1261 [07:39<07:28,  1.42it/s][A
     50%|████▉     | 627/1261 [07:40<07:27,  1.42it/s][A
     50%|████▉     | 628/1261 [07:41<07:25,  1.42it/s][A
     50%|████▉     | 629/1261 [07:41<07:23,  1.42it/s][A
     50%|████▉     | 630/1261 [07:42<07:24,  1.42it/s][A
     50%|█████     | 631/1261 [07:43<07:24,  1.42it/s][A
     50%|█████     | 632/1261 [07:44<07:22,  1.42it/s][A
     50%|█████     | 633/1261 [07:44<07:21,  1.42it/s][A
     50%|█████     | 634/1261 [07:45<07:21,  1.42it/s][A
     50%|█████     | 635/1261 [07:46<07:19,  1.43it/s][A
     50%|█████     | 636/1261 [07:46<07:18,  1.43it/s][A
     51%|█████     | 637/1261 [07:47<07:18,  1.42it/s][A
     51%|█████     | 638/1261 [07:48<07:18,  1.42it/s][A
     51%|█████     | 639/1261 [07:48<07:16,  1.43it/s][A
     51%|█████     | 640/1261 [07:49<07:16,  1.42it/s][A
     51%|█████     | 641/1261 [07:50<07:16,  1.42it/s][A
     51%|█████     | 642/1261 [07:51<07:14,  1.42it/s][A
     51%|█████     | 643/1261 [07:51<07:14,  1.42it/s][A
     51%|█████     | 644/1261 [07:52<07:12,  1.43it/s][A
     51%|█████     | 645/1261 [07:53<07:13,  1.42it/s][A
     51%|█████     | 646/1261 [07:53<07:13,  1.42it/s][A
     51%|█████▏    | 647/1261 [07:54<07:13,  1.42it/s][A
     51%|█████▏    | 648/1261 [07:55<07:11,  1.42it/s][A
     51%|█████▏    | 649/1261 [07:55<07:09,  1.42it/s][A
     52%|█████▏    | 650/1261 [07:56<07:09,  1.42it/s][A
     52%|█████▏    | 651/1261 [07:57<07:10,  1.42it/s][A
     52%|█████▏    | 652/1261 [07:58<07:08,  1.42it/s][A
     52%|█████▏    | 653/1261 [07:58<07:08,  1.42it/s][A
     52%|█████▏    | 654/1261 [07:59<07:08,  1.42it/s][A
     52%|█████▏    | 655/1261 [08:00<07:06,  1.42it/s][A
     52%|█████▏    | 656/1261 [08:00<07:06,  1.42it/s][A
     52%|█████▏    | 657/1261 [08:01<07:04,  1.42it/s][A
     52%|█████▏    | 658/1261 [08:02<07:03,  1.42it/s][A
     52%|█████▏    | 659/1261 [08:03<07:03,  1.42it/s][A
     52%|█████▏    | 660/1261 [08:03<07:03,  1.42it/s][A
     52%|█████▏    | 661/1261 [08:04<07:01,  1.42it/s][A
     52%|█████▏    | 662/1261 [08:05<06:59,  1.43it/s][A
     53%|█████▎    | 663/1261 [08:05<06:59,  1.42it/s][A
     53%|█████▎    | 664/1261 [08:06<06:57,  1.43it/s][A
     53%|█████▎    | 665/1261 [08:07<06:56,  1.43it/s][A
     53%|█████▎    | 666/1261 [08:07<06:56,  1.43it/s][A
     53%|█████▎    | 667/1261 [08:08<06:58,  1.42it/s][A
     53%|█████▎    | 668/1261 [08:09<07:17,  1.35it/s][A
     53%|█████▎    | 669/1261 [08:10<07:11,  1.37it/s][A
     53%|█████▎    | 670/1261 [08:10<07:06,  1.39it/s][A
     53%|█████▎    | 671/1261 [08:11<07:01,  1.40it/s][A
     53%|█████▎    | 672/1261 [08:12<06:58,  1.41it/s][A
     53%|█████▎    | 673/1261 [08:12<06:57,  1.41it/s][A
     53%|█████▎    | 674/1261 [08:13<06:56,  1.41it/s][A
     54%|█████▎    | 675/1261 [08:14<06:56,  1.41it/s][A
     54%|█████▎    | 676/1261 [08:15<06:52,  1.42it/s][A
     54%|█████▎    | 677/1261 [08:15<06:51,  1.42it/s][A
     54%|█████▍    | 678/1261 [08:16<06:51,  1.42it/s][A
     54%|█████▍    | 679/1261 [08:17<06:48,  1.42it/s][A
     54%|█████▍    | 680/1261 [08:17<06:48,  1.42it/s][A
     54%|█████▍    | 681/1261 [08:18<06:47,  1.42it/s][A
     54%|█████▍    | 682/1261 [08:19<06:45,  1.43it/s][A
     54%|█████▍    | 683/1261 [08:19<06:44,  1.43it/s][A
     54%|█████▍    | 684/1261 [08:20<06:43,  1.43it/s][A
     54%|█████▍    | 685/1261 [08:21<06:42,  1.43it/s][A
     54%|█████▍    | 686/1261 [08:22<06:43,  1.43it/s][A
     54%|█████▍    | 687/1261 [08:22<06:46,  1.41it/s][A
     55%|█████▍    | 688/1261 [08:23<06:44,  1.42it/s][A
     55%|█████▍    | 689/1261 [08:24<06:44,  1.42it/s][A
     55%|█████▍    | 690/1261 [08:24<06:44,  1.41it/s][A
     55%|█████▍    | 691/1261 [08:25<06:43,  1.41it/s][A
     55%|█████▍    | 692/1261 [08:26<06:43,  1.41it/s][A
     55%|█████▍    | 693/1261 [08:27<06:42,  1.41it/s][A
     55%|█████▌    | 694/1261 [08:27<06:42,  1.41it/s][A
     55%|█████▌    | 695/1261 [08:28<06:43,  1.40it/s][A
     55%|█████▌    | 696/1261 [08:29<06:41,  1.41it/s][A
     55%|█████▌    | 697/1261 [08:29<06:40,  1.41it/s][A
     55%|█████▌    | 698/1261 [08:30<06:40,  1.41it/s][A
     55%|█████▌    | 699/1261 [08:31<06:39,  1.41it/s][A
     56%|█████▌    | 700/1261 [08:32<06:39,  1.41it/s][A
     56%|█████▌    | 701/1261 [08:32<06:41,  1.39it/s][A
     56%|█████▌    | 702/1261 [08:33<06:40,  1.40it/s][A
     56%|█████▌    | 703/1261 [08:34<06:39,  1.40it/s][A
     56%|█████▌    | 704/1261 [08:34<06:37,  1.40it/s][A
     56%|█████▌    | 705/1261 [08:35<06:38,  1.39it/s][A
     56%|█████▌    | 706/1261 [08:36<06:37,  1.40it/s][A
     56%|█████▌    | 707/1261 [08:37<06:37,  1.39it/s][A
     56%|█████▌    | 708/1261 [08:37<06:37,  1.39it/s][A
     56%|█████▌    | 709/1261 [08:38<06:36,  1.39it/s][A
     56%|█████▋    | 710/1261 [08:39<06:35,  1.39it/s][A
     56%|█████▋    | 711/1261 [08:39<06:35,  1.39it/s][A
     56%|█████▋    | 712/1261 [08:40<06:37,  1.38it/s][A
     57%|█████▋    | 713/1261 [08:41<06:36,  1.38it/s][A
     57%|█████▋    | 714/1261 [08:42<06:33,  1.39it/s][A
     57%|█████▋    | 715/1261 [08:42<06:33,  1.39it/s][A
     57%|█████▋    | 716/1261 [08:43<06:32,  1.39it/s][A
     57%|█████▋    | 717/1261 [08:44<06:29,  1.40it/s][A
     57%|█████▋    | 718/1261 [08:44<06:26,  1.41it/s][A
     57%|█████▋    | 719/1261 [08:45<06:24,  1.41it/s][A
     57%|█████▋    | 720/1261 [08:46<06:22,  1.41it/s][A
     57%|█████▋    | 721/1261 [08:47<06:20,  1.42it/s][A
     57%|█████▋    | 722/1261 [08:47<06:19,  1.42it/s][A
     57%|█████▋    | 723/1261 [08:48<06:19,  1.42it/s][A
     57%|█████▋    | 724/1261 [08:49<06:18,  1.42it/s][A
     57%|█████▋    | 725/1261 [08:49<06:16,  1.42it/s][A
     58%|█████▊    | 726/1261 [08:50<06:16,  1.42it/s][A
     58%|█████▊    | 727/1261 [08:51<06:15,  1.42it/s][A
     58%|█████▊    | 728/1261 [08:51<06:15,  1.42it/s][A
     58%|█████▊    | 729/1261 [08:52<06:14,  1.42it/s][A
     58%|█████▊    | 730/1261 [08:53<06:14,  1.42it/s][A
     58%|█████▊    | 731/1261 [08:54<06:14,  1.42it/s][A
     58%|█████▊    | 732/1261 [08:54<06:12,  1.42it/s][A
     58%|█████▊    | 733/1261 [08:55<06:12,  1.42it/s][A
     58%|█████▊    | 734/1261 [08:56<06:10,  1.42it/s][A
     58%|█████▊    | 735/1261 [08:56<06:10,  1.42it/s][A
     58%|█████▊    | 736/1261 [08:57<06:11,  1.41it/s][A
     58%|█████▊    | 737/1261 [08:58<06:10,  1.41it/s][A
     59%|█████▊    | 738/1261 [08:59<06:09,  1.41it/s][A
     59%|█████▊    | 739/1261 [08:59<06:09,  1.41it/s][A
     59%|█████▊    | 740/1261 [09:00<06:07,  1.42it/s][A
     59%|█████▉    | 741/1261 [09:01<06:05,  1.42it/s][A
     59%|█████▉    | 742/1261 [09:01<06:04,  1.42it/s][A
     59%|█████▉    | 743/1261 [09:02<06:02,  1.43it/s][A
     59%|█████▉    | 744/1261 [09:03<06:04,  1.42it/s][A
     59%|█████▉    | 745/1261 [09:03<06:04,  1.42it/s][A
     59%|█████▉    | 746/1261 [09:04<06:02,  1.42it/s][A
     59%|█████▉    | 747/1261 [09:05<06:01,  1.42it/s][A
     59%|█████▉    | 748/1261 [09:06<05:59,  1.43it/s][A
     59%|█████▉    | 749/1261 [09:06<05:59,  1.42it/s][A
     59%|█████▉    | 750/1261 [09:07<05:58,  1.43it/s][A
     60%|█████▉    | 751/1261 [09:08<05:57,  1.43it/s][A
     60%|█████▉    | 752/1261 [09:08<05:56,  1.43it/s][A
     60%|█████▉    | 753/1261 [09:09<06:04,  1.39it/s][A
     60%|█████▉    | 754/1261 [09:10<06:01,  1.40it/s][A
     60%|█████▉    | 755/1261 [09:11<05:59,  1.41it/s][A
     60%|█████▉    | 756/1261 [09:11<05:57,  1.41it/s][A
     60%|██████    | 757/1261 [09:12<05:59,  1.40it/s][A
     60%|██████    | 758/1261 [09:13<05:57,  1.41it/s][A
     60%|██████    | 759/1261 [09:13<05:55,  1.41it/s][A
     60%|██████    | 760/1261 [09:14<05:54,  1.41it/s][A
     60%|██████    | 761/1261 [09:15<05:52,  1.42it/s][A
     60%|██████    | 762/1261 [09:15<05:50,  1.42it/s][A
     61%|██████    | 763/1261 [09:16<05:51,  1.42it/s][A
     61%|██████    | 764/1261 [09:17<05:50,  1.42it/s][A
     61%|██████    | 765/1261 [09:18<05:48,  1.42it/s][A
     61%|██████    | 766/1261 [09:18<05:47,  1.42it/s][A
     61%|██████    | 767/1261 [09:19<05:48,  1.42it/s][A
     61%|██████    | 768/1261 [09:20<05:47,  1.42it/s][A
     61%|██████    | 769/1261 [09:20<05:45,  1.42it/s][A
     61%|██████    | 770/1261 [09:21<05:44,  1.42it/s][A
     61%|██████    | 771/1261 [09:22<05:44,  1.42it/s][A
     61%|██████    | 772/1261 [09:23<05:44,  1.42it/s][A
     61%|██████▏   | 773/1261 [09:23<05:45,  1.41it/s][A
     61%|██████▏   | 774/1261 [09:24<05:43,  1.42it/s][A
     61%|██████▏   | 775/1261 [09:25<05:42,  1.42it/s][A
     62%|██████▏   | 776/1261 [09:25<05:40,  1.42it/s][A
     62%|██████▏   | 777/1261 [09:26<05:40,  1.42it/s][A
     62%|██████▏   | 778/1261 [09:27<05:38,  1.43it/s][A
     62%|██████▏   | 779/1261 [09:27<05:37,  1.43it/s][A
     62%|██████▏   | 780/1261 [09:28<05:38,  1.42it/s][A
     62%|██████▏   | 781/1261 [09:29<05:37,  1.42it/s][A
     62%|██████▏   | 782/1261 [09:30<05:37,  1.42it/s][A
     62%|██████▏   | 783/1261 [09:30<05:35,  1.42it/s][A
     62%|██████▏   | 784/1261 [09:31<05:35,  1.42it/s][A
     62%|██████▏   | 785/1261 [09:32<05:34,  1.42it/s][A
     62%|██████▏   | 786/1261 [09:32<05:35,  1.42it/s][A
     62%|██████▏   | 787/1261 [09:33<05:34,  1.42it/s][A
     62%|██████▏   | 788/1261 [09:34<05:33,  1.42it/s][A
     63%|██████▎   | 789/1261 [09:34<05:32,  1.42it/s][A
     63%|██████▎   | 790/1261 [09:35<05:31,  1.42it/s][A
     63%|██████▎   | 791/1261 [09:36<05:30,  1.42it/s][A
     63%|██████▎   | 792/1261 [09:37<05:29,  1.42it/s][A
     63%|██████▎   | 793/1261 [09:37<05:29,  1.42it/s][A
     63%|██████▎   | 794/1261 [09:38<05:28,  1.42it/s][A
     63%|██████▎   | 795/1261 [09:39<05:28,  1.42it/s][A
     63%|██████▎   | 796/1261 [09:39<05:29,  1.41it/s][A
     63%|██████▎   | 797/1261 [09:40<05:28,  1.41it/s][A
     63%|██████▎   | 798/1261 [09:41<05:26,  1.42it/s][A
     63%|██████▎   | 799/1261 [09:42<05:25,  1.42it/s][A
     63%|██████▎   | 800/1261 [09:42<05:26,  1.41it/s][A
     64%|██████▎   | 801/1261 [09:43<05:34,  1.37it/s][A
     64%|██████▎   | 802/1261 [09:44<05:54,  1.29it/s][A
     64%|██████▎   | 803/1261 [09:45<05:54,  1.29it/s][A
     64%|██████▍   | 804/1261 [09:45<05:54,  1.29it/s][A
     64%|██████▍   | 805/1261 [09:46<05:57,  1.28it/s][A
     64%|██████▍   | 806/1261 [09:47<05:49,  1.30it/s][A
     64%|██████▍   | 807/1261 [09:48<05:46,  1.31it/s][A
     64%|██████▍   | 808/1261 [09:49<05:47,  1.31it/s][A
     64%|██████▍   | 809/1261 [09:49<05:37,  1.34it/s][A
     64%|██████▍   | 810/1261 [09:50<05:38,  1.33it/s][A
     64%|██████▍   | 811/1261 [09:51<05:54,  1.27it/s][A
     64%|██████▍   | 812/1261 [09:52<06:00,  1.25it/s][A
     64%|██████▍   | 813/1261 [09:53<06:24,  1.16it/s][A
     65%|██████▍   | 814/1261 [09:54<06:19,  1.18it/s][A
     65%|██████▍   | 815/1261 [09:54<06:11,  1.20it/s][A
     65%|██████▍   | 816/1261 [09:55<06:00,  1.23it/s][A
     65%|██████▍   | 817/1261 [09:56<05:53,  1.26it/s][A
     65%|██████▍   | 818/1261 [09:57<06:12,  1.19it/s][A
     65%|██████▍   | 819/1261 [09:58<06:11,  1.19it/s][A
     65%|██████▌   | 820/1261 [09:58<06:05,  1.21it/s][A
     65%|██████▌   | 821/1261 [09:59<05:53,  1.24it/s][A
     65%|██████▌   | 822/1261 [10:00<05:39,  1.29it/s][A
     65%|██████▌   | 823/1261 [10:01<05:37,  1.30it/s][A
     65%|██████▌   | 824/1261 [10:02<05:59,  1.21it/s][A
     65%|██████▌   | 825/1261 [10:02<06:11,  1.17it/s][A
     66%|██████▌   | 826/1261 [10:03<06:02,  1.20it/s][A
     66%|██████▌   | 827/1261 [10:04<05:53,  1.23it/s][A
     66%|██████▌   | 828/1261 [10:05<05:38,  1.28it/s][A
     66%|██████▌   | 829/1261 [10:05<05:27,  1.32it/s][A
     66%|██████▌   | 830/1261 [10:06<05:18,  1.35it/s][A
     66%|██████▌   | 831/1261 [10:07<05:16,  1.36it/s][A
     66%|██████▌   | 832/1261 [10:08<05:14,  1.37it/s][A
     66%|██████▌   | 833/1261 [10:08<05:13,  1.37it/s][A
     66%|██████▌   | 834/1261 [10:09<05:14,  1.36it/s][A
     66%|██████▌   | 835/1261 [10:10<05:09,  1.37it/s][A
     66%|██████▋   | 836/1261 [10:11<05:17,  1.34it/s][A
     66%|██████▋   | 837/1261 [10:11<05:16,  1.34it/s][A
     66%|██████▋   | 838/1261 [10:12<05:15,  1.34it/s][A
     67%|██████▋   | 839/1261 [10:13<05:17,  1.33it/s][A
     67%|██████▋   | 840/1261 [10:14<05:23,  1.30it/s][A
     67%|██████▋   | 841/1261 [10:14<05:29,  1.27it/s][A
     67%|██████▋   | 842/1261 [10:15<05:27,  1.28it/s][A
     67%|██████▋   | 843/1261 [10:16<05:22,  1.30it/s][A
     67%|██████▋   | 844/1261 [10:17<05:22,  1.29it/s][A
     67%|██████▋   | 845/1261 [10:18<05:37,  1.23it/s][A
     67%|██████▋   | 846/1261 [10:18<05:36,  1.23it/s][A
     67%|██████▋   | 847/1261 [10:19<05:41,  1.21it/s][A
     67%|██████▋   | 848/1261 [10:20<05:36,  1.23it/s][A
     67%|██████▋   | 849/1261 [10:21<05:37,  1.22it/s][A
     67%|██████▋   | 850/1261 [10:22<05:46,  1.19it/s][A
     67%|██████▋   | 851/1261 [10:23<05:35,  1.22it/s][A
     68%|██████▊   | 852/1261 [10:24<05:49,  1.17it/s][A
     68%|██████▊   | 853/1261 [10:24<05:45,  1.18it/s][A
     68%|██████▊   | 854/1261 [10:25<05:42,  1.19it/s][A
     68%|██████▊   | 855/1261 [10:26<05:33,  1.22it/s][A
     68%|██████▊   | 856/1261 [10:27<05:21,  1.26it/s][A
     68%|██████▊   | 857/1261 [10:27<05:13,  1.29it/s][A
     68%|██████▊   | 858/1261 [10:28<05:07,  1.31it/s][A
     68%|██████▊   | 859/1261 [10:29<05:02,  1.33it/s][A
     68%|██████▊   | 860/1261 [10:30<04:57,  1.35it/s][A
     68%|██████▊   | 861/1261 [10:30<05:02,  1.32it/s][A
     68%|██████▊   | 862/1261 [10:31<05:03,  1.31it/s][A
     68%|██████▊   | 863/1261 [10:32<05:10,  1.28it/s][A
     69%|██████▊   | 864/1261 [10:33<05:01,  1.32it/s][A
     69%|██████▊   | 865/1261 [10:33<04:59,  1.32it/s][A
     69%|██████▊   | 866/1261 [10:34<04:57,  1.33it/s][A
     69%|██████▉   | 867/1261 [10:35<04:56,  1.33it/s][A
     69%|██████▉   | 868/1261 [10:36<05:11,  1.26it/s][A
     69%|██████▉   | 869/1261 [10:37<05:17,  1.24it/s][A
     69%|██████▉   | 870/1261 [10:37<05:10,  1.26it/s][A
     69%|██████▉   | 871/1261 [10:38<05:07,  1.27it/s][A
     69%|██████▉   | 872/1261 [10:39<05:07,  1.27it/s][A
     69%|██████▉   | 873/1261 [10:40<05:09,  1.25it/s][A
     69%|██████▉   | 874/1261 [10:41<04:57,  1.30it/s][A
     69%|██████▉   | 875/1261 [10:41<04:49,  1.33it/s][A
     69%|██████▉   | 876/1261 [10:42<04:46,  1.34it/s][A
     70%|██████▉   | 877/1261 [10:43<04:44,  1.35it/s][A
     70%|██████▉   | 878/1261 [10:43<04:42,  1.36it/s][A
     70%|██████▉   | 879/1261 [10:44<04:37,  1.38it/s][A
     70%|██████▉   | 880/1261 [10:45<04:34,  1.39it/s][A
     70%|██████▉   | 881/1261 [10:46<04:31,  1.40it/s][A
     70%|██████▉   | 882/1261 [10:46<04:32,  1.39it/s][A
     70%|███████   | 883/1261 [10:47<04:31,  1.39it/s][A
     70%|███████   | 884/1261 [10:48<04:31,  1.39it/s][A
     70%|███████   | 885/1261 [10:48<04:29,  1.39it/s][A
     70%|███████   | 886/1261 [10:49<04:27,  1.40it/s][A
     70%|███████   | 887/1261 [10:50<04:25,  1.41it/s][A
     70%|███████   | 888/1261 [10:51<04:25,  1.40it/s][A
     70%|███████   | 889/1261 [10:51<04:24,  1.40it/s][A
     71%|███████   | 890/1261 [10:52<04:24,  1.40it/s][A
     71%|███████   | 891/1261 [10:53<04:24,  1.40it/s][A
     71%|███████   | 892/1261 [10:53<04:23,  1.40it/s][A
     71%|███████   | 893/1261 [10:54<04:22,  1.40it/s][A
     71%|███████   | 894/1261 [10:55<04:20,  1.41it/s][A
     71%|███████   | 895/1261 [10:56<04:21,  1.40it/s][A
     71%|███████   | 896/1261 [10:56<04:20,  1.40it/s][A
     71%|███████   | 897/1261 [10:57<04:19,  1.40it/s][A
     71%|███████   | 898/1261 [10:58<04:26,  1.36it/s][A
     71%|███████▏  | 899/1261 [10:59<04:31,  1.33it/s][A
     71%|███████▏  | 900/1261 [10:59<04:37,  1.30it/s][A
     71%|███████▏  | 901/1261 [11:00<04:42,  1.28it/s][A
     72%|███████▏  | 902/1261 [11:01<04:33,  1.31it/s][A
     72%|███████▏  | 903/1261 [11:02<04:29,  1.33it/s][A
     72%|███████▏  | 904/1261 [11:02<04:34,  1.30it/s][A
     72%|███████▏  | 905/1261 [11:03<04:32,  1.31it/s][A
     72%|███████▏  | 906/1261 [11:04<04:23,  1.35it/s][A
     72%|███████▏  | 907/1261 [11:05<04:22,  1.35it/s][A
     72%|███████▏  | 908/1261 [11:05<04:19,  1.36it/s][A
     72%|███████▏  | 909/1261 [11:06<04:20,  1.35it/s][A
     72%|███████▏  | 910/1261 [11:07<04:23,  1.33it/s][A
     72%|███████▏  | 911/1261 [11:08<04:30,  1.29it/s][A
     72%|███████▏  | 912/1261 [11:09<04:57,  1.17it/s][A
     72%|███████▏  | 913/1261 [11:10<05:04,  1.14it/s][A
     72%|███████▏  | 914/1261 [11:11<05:14,  1.10it/s][A
     73%|███████▎  | 915/1261 [11:11<05:07,  1.12it/s][A
     73%|███████▎  | 916/1261 [11:12<04:55,  1.17it/s][A
     73%|███████▎  | 917/1261 [11:13<04:45,  1.21it/s][A
     73%|███████▎  | 918/1261 [11:14<04:54,  1.17it/s][A
     73%|███████▎  | 919/1261 [11:15<05:03,  1.13it/s][A
     73%|███████▎  | 920/1261 [11:16<04:54,  1.16it/s][A
     73%|███████▎  | 921/1261 [11:16<04:44,  1.20it/s][A
     73%|███████▎  | 922/1261 [11:17<04:35,  1.23it/s][A
     73%|███████▎  | 923/1261 [11:18<04:26,  1.27it/s][A
     73%|███████▎  | 924/1261 [11:19<04:20,  1.29it/s][A
     73%|███████▎  | 925/1261 [11:20<04:27,  1.25it/s][A
     73%|███████▎  | 926/1261 [11:20<04:34,  1.22it/s][A
     74%|███████▎  | 927/1261 [11:21<04:24,  1.26it/s][A
     74%|███████▎  | 928/1261 [11:22<04:20,  1.28it/s][A
     74%|███████▎  | 929/1261 [11:23<04:16,  1.29it/s][A
     74%|███████▍  | 930/1261 [11:23<04:18,  1.28it/s][A
     74%|███████▍  | 931/1261 [11:24<04:21,  1.26it/s][A
     74%|███████▍  | 932/1261 [11:25<04:19,  1.27it/s][A
     74%|███████▍  | 933/1261 [11:26<04:15,  1.28it/s][A
     74%|███████▍  | 934/1261 [11:27<04:12,  1.29it/s][A
     74%|███████▍  | 935/1261 [11:27<04:11,  1.30it/s][A
     74%|███████▍  | 936/1261 [11:28<04:04,  1.33it/s][A
     74%|███████▍  | 937/1261 [11:29<03:56,  1.37it/s][A
     74%|███████▍  | 938/1261 [11:29<03:56,  1.37it/s][A
     74%|███████▍  | 939/1261 [11:30<03:52,  1.38it/s][A
     75%|███████▍  | 940/1261 [11:31<03:50,  1.39it/s][A
     75%|███████▍  | 941/1261 [11:32<03:48,  1.40it/s][A
     75%|███████▍  | 942/1261 [11:32<03:48,  1.39it/s][A
     75%|███████▍  | 943/1261 [11:33<03:47,  1.40it/s][A
     75%|███████▍  | 944/1261 [11:34<03:45,  1.40it/s][A
     75%|███████▍  | 945/1261 [11:34<03:43,  1.41it/s][A
     75%|███████▌  | 946/1261 [11:35<03:41,  1.42it/s][A
     75%|███████▌  | 947/1261 [11:36<03:44,  1.40it/s][A
     75%|███████▌  | 948/1261 [11:37<03:44,  1.39it/s][A
     75%|███████▌  | 949/1261 [11:37<03:42,  1.40it/s][A
     75%|███████▌  | 950/1261 [11:38<03:42,  1.40it/s][A
     75%|███████▌  | 951/1261 [11:39<03:44,  1.38it/s][A
     75%|███████▌  | 952/1261 [11:40<03:45,  1.37it/s][A
     76%|███████▌  | 953/1261 [11:40<04:01,  1.28it/s][A
     76%|███████▌  | 954/1261 [11:42<04:37,  1.10it/s][A
     76%|███████▌  | 955/1261 [11:42<04:29,  1.13it/s][A
     76%|███████▌  | 956/1261 [11:43<04:18,  1.18it/s][A
     76%|███████▌  | 957/1261 [11:44<04:09,  1.22it/s][A
     76%|███████▌  | 958/1261 [11:45<04:02,  1.25it/s][A
     76%|███████▌  | 959/1261 [11:45<03:56,  1.28it/s][A
     76%|███████▌  | 960/1261 [11:46<03:52,  1.29it/s][A
     76%|███████▌  | 961/1261 [11:47<03:52,  1.29it/s][A
     76%|███████▋  | 962/1261 [11:48<03:45,  1.33it/s][A
     76%|███████▋  | 963/1261 [11:48<03:41,  1.34it/s][A
     76%|███████▋  | 964/1261 [11:49<03:39,  1.35it/s][A
     77%|███████▋  | 965/1261 [11:50<03:46,  1.31it/s][A
     77%|███████▋  | 966/1261 [11:51<04:03,  1.21it/s][A
     77%|███████▋  | 967/1261 [11:52<04:02,  1.21it/s][A
     77%|███████▋  | 968/1261 [11:53<04:13,  1.15it/s][A
     77%|███████▋  | 969/1261 [11:54<04:13,  1.15it/s][A
     77%|███████▋  | 970/1261 [11:54<04:01,  1.20it/s][A
     77%|███████▋  | 971/1261 [11:55<04:02,  1.19it/s][A
     77%|███████▋  | 972/1261 [11:56<04:01,  1.20it/s][A
     77%|███████▋  | 973/1261 [11:57<03:54,  1.23it/s][A
     77%|███████▋  | 974/1261 [11:57<03:43,  1.28it/s][A
     77%|███████▋  | 975/1261 [11:58<03:40,  1.30it/s][A
     77%|███████▋  | 976/1261 [11:59<03:34,  1.33it/s][A
     77%|███████▋  | 977/1261 [12:00<03:31,  1.34it/s][A
     78%|███████▊  | 978/1261 [12:00<03:30,  1.35it/s][A
     78%|███████▊  | 979/1261 [12:01<03:34,  1.31it/s][A
     78%|███████▊  | 980/1261 [12:02<03:32,  1.32it/s][A
     78%|███████▊  | 981/1261 [12:03<03:34,  1.30it/s][A
     78%|███████▊  | 982/1261 [12:04<03:35,  1.29it/s][A
     78%|███████▊  | 983/1261 [12:04<03:42,  1.25it/s][A
     78%|███████▊  | 984/1261 [12:05<03:49,  1.21it/s][A
     78%|███████▊  | 985/1261 [12:06<03:40,  1.25it/s][A
     78%|███████▊  | 986/1261 [12:07<03:33,  1.29it/s][A
     78%|███████▊  | 987/1261 [12:07<03:29,  1.31it/s][A
     78%|███████▊  | 988/1261 [12:08<03:25,  1.33it/s][A
     78%|███████▊  | 989/1261 [12:09<03:23,  1.34it/s][A
     79%|███████▊  | 990/1261 [12:10<03:21,  1.35it/s][A
     79%|███████▊  | 991/1261 [12:10<03:17,  1.37it/s][A
     79%|███████▊  | 992/1261 [12:11<03:14,  1.38it/s][A
     79%|███████▊  | 993/1261 [12:12<03:13,  1.39it/s][A
     79%|███████▉  | 994/1261 [12:12<03:11,  1.40it/s][A
     79%|███████▉  | 995/1261 [12:13<03:09,  1.40it/s][A
     79%|███████▉  | 996/1261 [12:14<03:07,  1.41it/s][A
     79%|███████▉  | 997/1261 [12:15<03:05,  1.42it/s][A
     79%|███████▉  | 998/1261 [12:15<03:04,  1.42it/s][A
     79%|███████▉  | 999/1261 [12:16<03:03,  1.43it/s][A
     79%|███████▉  | 1000/1261 [12:17<03:02,  1.43it/s][A
     79%|███████▉  | 1001/1261 [12:17<03:01,  1.43it/s][A
     79%|███████▉  | 1002/1261 [12:18<03:00,  1.43it/s][A
     80%|███████▉  | 1003/1261 [12:19<03:00,  1.43it/s][A
     80%|███████▉  | 1004/1261 [12:19<03:00,  1.42it/s][A
     80%|███████▉  | 1005/1261 [12:20<03:00,  1.42it/s][A
     80%|███████▉  | 1006/1261 [12:21<03:02,  1.40it/s][A
     80%|███████▉  | 1007/1261 [12:22<03:00,  1.40it/s][A
     80%|███████▉  | 1008/1261 [12:22<02:59,  1.41it/s][A
     80%|████████  | 1009/1261 [12:23<02:58,  1.41it/s][A
     80%|████████  | 1010/1261 [12:24<02:56,  1.42it/s][A
     80%|████████  | 1011/1261 [12:24<02:55,  1.43it/s][A
     80%|████████  | 1012/1261 [12:25<02:54,  1.43it/s][A
     80%|████████  | 1013/1261 [12:26<02:53,  1.43it/s][A
     80%|████████  | 1014/1261 [12:27<02:52,  1.43it/s][A
     80%|████████  | 1015/1261 [12:27<02:51,  1.43it/s][A
     81%|████████  | 1016/1261 [12:28<02:51,  1.43it/s][A
     81%|████████  | 1017/1261 [12:29<02:50,  1.43it/s][A
     81%|████████  | 1018/1261 [12:29<02:49,  1.43it/s][A
     81%|████████  | 1019/1261 [12:30<02:48,  1.44it/s][A
     81%|████████  | 1020/1261 [12:31<02:48,  1.43it/s][A
     81%|████████  | 1021/1261 [12:31<02:47,  1.43it/s][A
     81%|████████  | 1022/1261 [12:32<02:46,  1.43it/s][A
     81%|████████  | 1023/1261 [12:33<02:46,  1.43it/s][A
     81%|████████  | 1024/1261 [12:33<02:45,  1.44it/s][A
     81%|████████▏ | 1025/1261 [12:34<02:44,  1.44it/s][A
     81%|████████▏ | 1026/1261 [12:35<02:43,  1.44it/s][A
     81%|████████▏ | 1027/1261 [12:36<02:42,  1.44it/s][A
     82%|████████▏ | 1028/1261 [12:36<02:41,  1.44it/s][A
     82%|████████▏ | 1029/1261 [12:37<02:41,  1.44it/s][A
     82%|████████▏ | 1030/1261 [12:38<02:42,  1.42it/s][A
     82%|████████▏ | 1031/1261 [12:38<02:43,  1.41it/s][A
     82%|████████▏ | 1032/1261 [12:39<02:49,  1.35it/s][A
     82%|████████▏ | 1033/1261 [12:40<02:48,  1.35it/s][A
     82%|████████▏ | 1034/1261 [12:41<02:45,  1.37it/s][A
     82%|████████▏ | 1035/1261 [12:41<02:43,  1.38it/s][A
     82%|████████▏ | 1036/1261 [12:42<02:40,  1.40it/s][A
     82%|████████▏ | 1037/1261 [12:43<02:38,  1.41it/s][A
     82%|████████▏ | 1038/1261 [12:43<02:38,  1.41it/s][A
     82%|████████▏ | 1039/1261 [12:44<02:38,  1.40it/s][A
     82%|████████▏ | 1040/1261 [12:45<02:39,  1.39it/s][A
     83%|████████▎ | 1041/1261 [12:46<02:39,  1.38it/s][A
     83%|████████▎ | 1042/1261 [12:46<02:36,  1.40it/s][A
     83%|████████▎ | 1043/1261 [12:47<02:34,  1.41it/s][A
     83%|████████▎ | 1044/1261 [12:48<02:32,  1.43it/s][A
     83%|████████▎ | 1045/1261 [12:48<02:30,  1.43it/s][A
     83%|████████▎ | 1046/1261 [12:49<02:29,  1.43it/s][A
     83%|████████▎ | 1047/1261 [12:50<02:28,  1.44it/s][A
     83%|████████▎ | 1048/1261 [12:50<02:27,  1.44it/s][A
     83%|████████▎ | 1049/1261 [12:51<02:26,  1.44it/s][A
     83%|████████▎ | 1050/1261 [12:52<02:26,  1.44it/s][A
     83%|████████▎ | 1051/1261 [12:53<02:26,  1.44it/s][A
     83%|████████▎ | 1052/1261 [12:53<02:25,  1.44it/s][A
     84%|████████▎ | 1053/1261 [12:54<02:24,  1.44it/s][A
     84%|████████▎ | 1054/1261 [12:55<02:23,  1.44it/s][A
     84%|████████▎ | 1055/1261 [12:55<02:24,  1.43it/s][A
     84%|████████▎ | 1056/1261 [12:56<02:24,  1.42it/s][A
     84%|████████▍ | 1057/1261 [12:57<02:27,  1.38it/s][A
     84%|████████▍ | 1058/1261 [12:58<02:25,  1.39it/s][A
     84%|████████▍ | 1059/1261 [12:58<02:24,  1.40it/s][A
     84%|████████▍ | 1060/1261 [12:59<02:23,  1.40it/s][A
     84%|████████▍ | 1061/1261 [13:00<02:22,  1.40it/s][A
     84%|████████▍ | 1062/1261 [13:00<02:21,  1.40it/s][A
     84%|████████▍ | 1063/1261 [13:01<02:21,  1.40it/s][A
     84%|████████▍ | 1064/1261 [13:02<02:20,  1.40it/s][A
     84%|████████▍ | 1065/1261 [13:03<02:19,  1.40it/s][A
     85%|████████▍ | 1066/1261 [13:03<02:18,  1.40it/s][A
     85%|████████▍ | 1067/1261 [13:04<02:18,  1.40it/s][A
     85%|████████▍ | 1068/1261 [13:05<02:17,  1.40it/s][A
     85%|████████▍ | 1069/1261 [13:05<02:17,  1.40it/s][A
     85%|████████▍ | 1070/1261 [13:06<02:15,  1.41it/s][A
     85%|████████▍ | 1071/1261 [13:07<02:21,  1.34it/s][A
     85%|████████▌ | 1072/1261 [13:08<02:21,  1.33it/s][A
     85%|████████▌ | 1073/1261 [13:08<02:21,  1.33it/s][A
     85%|████████▌ | 1074/1261 [13:09<02:28,  1.26it/s][A
     85%|████████▌ | 1075/1261 [13:10<02:31,  1.23it/s][A
     85%|████████▌ | 1076/1261 [13:11<02:34,  1.20it/s][A
     85%|████████▌ | 1077/1261 [13:12<02:34,  1.19it/s][A
     85%|████████▌ | 1078/1261 [13:13<02:33,  1.19it/s][A
     86%|████████▌ | 1079/1261 [13:14<02:27,  1.24it/s][A
     86%|████████▌ | 1080/1261 [13:14<02:20,  1.29it/s][A
     86%|████████▌ | 1081/1261 [13:15<02:15,  1.33it/s][A
     86%|████████▌ | 1082/1261 [13:16<02:11,  1.36it/s][A
     86%|████████▌ | 1083/1261 [13:16<02:11,  1.35it/s][A
     86%|████████▌ | 1084/1261 [13:17<02:16,  1.30it/s][A
     86%|████████▌ | 1085/1261 [13:18<02:12,  1.33it/s][A
     86%|████████▌ | 1086/1261 [13:19<02:13,  1.31it/s][A
     86%|████████▌ | 1087/1261 [13:19<02:09,  1.34it/s][A
     86%|████████▋ | 1088/1261 [13:20<02:07,  1.36it/s][A
     86%|████████▋ | 1089/1261 [13:21<02:04,  1.38it/s][A
     86%|████████▋ | 1090/1261 [13:22<02:03,  1.38it/s][A
     87%|████████▋ | 1091/1261 [13:22<02:02,  1.39it/s][A
     87%|████████▋ | 1092/1261 [13:23<02:00,  1.40it/s][A
     87%|████████▋ | 1093/1261 [13:24<02:00,  1.40it/s][A
     87%|████████▋ | 1094/1261 [13:24<02:00,  1.39it/s][A
     87%|████████▋ | 1095/1261 [13:25<02:01,  1.36it/s][A
     87%|████████▋ | 1096/1261 [13:26<01:59,  1.38it/s][A
     87%|████████▋ | 1097/1261 [13:27<01:57,  1.40it/s][A
     87%|████████▋ | 1098/1261 [13:27<01:56,  1.40it/s][A
     87%|████████▋ | 1099/1261 [13:28<01:54,  1.41it/s][A
     87%|████████▋ | 1100/1261 [13:29<01:53,  1.42it/s][A
     87%|████████▋ | 1101/1261 [13:29<01:52,  1.42it/s][A
     87%|████████▋ | 1102/1261 [13:30<01:51,  1.42it/s][A
     87%|████████▋ | 1103/1261 [13:31<01:51,  1.42it/s][A
     88%|████████▊ | 1104/1261 [13:31<01:50,  1.43it/s][A
     88%|████████▊ | 1105/1261 [13:32<01:49,  1.43it/s][A
     88%|████████▊ | 1106/1261 [13:33<01:48,  1.43it/s][A
     88%|████████▊ | 1107/1261 [13:34<01:47,  1.43it/s][A
     88%|████████▊ | 1108/1261 [13:34<01:46,  1.43it/s][A
     88%|████████▊ | 1109/1261 [13:35<01:46,  1.43it/s][A
     88%|████████▊ | 1110/1261 [13:36<01:45,  1.43it/s][A
     88%|████████▊ | 1111/1261 [13:36<01:44,  1.43it/s][A
     88%|████████▊ | 1112/1261 [13:37<01:44,  1.43it/s][A
     88%|████████▊ | 1113/1261 [13:38<01:43,  1.43it/s][A
     88%|████████▊ | 1114/1261 [13:38<01:43,  1.43it/s][A
     88%|████████▊ | 1115/1261 [13:39<01:43,  1.41it/s][A
     89%|████████▊ | 1116/1261 [13:40<01:42,  1.42it/s][A
     89%|████████▊ | 1117/1261 [13:41<01:41,  1.42it/s][A
     89%|████████▊ | 1118/1261 [13:41<01:40,  1.42it/s][A
     89%|████████▊ | 1119/1261 [13:42<01:39,  1.43it/s][A
     89%|████████▉ | 1120/1261 [13:43<01:38,  1.43it/s][A
     89%|████████▉ | 1121/1261 [13:43<01:37,  1.43it/s][A
     89%|████████▉ | 1122/1261 [13:44<01:37,  1.43it/s][A
     89%|████████▉ | 1123/1261 [13:45<01:36,  1.44it/s][A
     89%|████████▉ | 1124/1261 [13:45<01:35,  1.43it/s][A
     89%|████████▉ | 1125/1261 [13:46<01:35,  1.43it/s][A
     89%|████████▉ | 1126/1261 [13:47<01:34,  1.43it/s][A
     89%|████████▉ | 1127/1261 [13:48<01:34,  1.42it/s][A
     89%|████████▉ | 1128/1261 [13:48<01:33,  1.42it/s][A
     90%|████████▉ | 1129/1261 [13:49<01:32,  1.43it/s][A
     90%|████████▉ | 1130/1261 [13:50<01:31,  1.43it/s][A
     90%|████████▉ | 1131/1261 [13:50<01:30,  1.43it/s][A
     90%|████████▉ | 1132/1261 [13:51<01:30,  1.42it/s][A
     90%|████████▉ | 1133/1261 [13:52<01:29,  1.43it/s][A
     90%|████████▉ | 1134/1261 [13:53<01:29,  1.42it/s][A
     90%|█████████ | 1135/1261 [13:53<01:28,  1.43it/s][A
     90%|█████████ | 1136/1261 [13:54<01:27,  1.43it/s][A
     90%|█████████ | 1137/1261 [13:55<01:26,  1.43it/s][A
     90%|█████████ | 1138/1261 [13:55<01:25,  1.43it/s][A
     90%|█████████ | 1139/1261 [13:56<01:25,  1.43it/s][A
     90%|█████████ | 1140/1261 [13:57<01:24,  1.44it/s][A
     90%|█████████ | 1141/1261 [13:57<01:23,  1.44it/s][A
     91%|█████████ | 1142/1261 [13:58<01:22,  1.44it/s][A
     91%|█████████ | 1143/1261 [13:59<01:21,  1.44it/s][A
     91%|█████████ | 1144/1261 [13:59<01:21,  1.44it/s][A
     91%|█████████ | 1145/1261 [14:00<01:20,  1.43it/s][A
     91%|█████████ | 1146/1261 [14:01<01:20,  1.44it/s][A
     91%|█████████ | 1147/1261 [14:02<01:19,  1.44it/s][A
     91%|█████████ | 1148/1261 [14:02<01:19,  1.43it/s][A
     91%|█████████ | 1149/1261 [14:03<01:18,  1.43it/s][A
     91%|█████████ | 1150/1261 [14:04<01:17,  1.43it/s][A
     91%|█████████▏| 1151/1261 [14:04<01:16,  1.43it/s][A
     91%|█████████▏| 1152/1261 [14:05<01:16,  1.43it/s][A
     91%|█████████▏| 1153/1261 [14:06<01:15,  1.44it/s][A
     92%|█████████▏| 1154/1261 [14:06<01:14,  1.44it/s][A
     92%|█████████▏| 1155/1261 [14:07<01:13,  1.43it/s][A
     92%|█████████▏| 1156/1261 [14:08<01:13,  1.42it/s][A
     92%|█████████▏| 1157/1261 [14:09<01:13,  1.42it/s][A
     92%|█████████▏| 1158/1261 [14:09<01:12,  1.41it/s][A
     92%|█████████▏| 1159/1261 [14:10<01:11,  1.42it/s][A
     92%|█████████▏| 1160/1261 [14:11<01:10,  1.42it/s][A
     92%|█████████▏| 1161/1261 [14:11<01:10,  1.43it/s][A
     92%|█████████▏| 1162/1261 [14:12<01:09,  1.43it/s][A
     92%|█████████▏| 1163/1261 [14:13<01:08,  1.42it/s][A
     92%|█████████▏| 1164/1261 [14:13<01:08,  1.42it/s][A
     92%|█████████▏| 1165/1261 [14:14<01:07,  1.43it/s][A
     92%|█████████▏| 1166/1261 [14:15<01:06,  1.43it/s][A
     93%|█████████▎| 1167/1261 [14:16<01:05,  1.43it/s][A
     93%|█████████▎| 1168/1261 [14:16<01:05,  1.42it/s][A
     93%|█████████▎| 1169/1261 [14:17<01:04,  1.42it/s][A
     93%|█████████▎| 1170/1261 [14:18<01:03,  1.43it/s][A
     93%|█████████▎| 1171/1261 [14:18<01:02,  1.43it/s][A
     93%|█████████▎| 1172/1261 [14:19<01:02,  1.43it/s][A
     93%|█████████▎| 1173/1261 [14:20<01:02,  1.40it/s][A
     93%|█████████▎| 1174/1261 [14:21<01:04,  1.35it/s][A
     93%|█████████▎| 1175/1261 [14:21<01:04,  1.33it/s][A
     93%|█████████▎| 1176/1261 [14:22<01:04,  1.32it/s][A
     93%|█████████▎| 1177/1261 [14:23<01:04,  1.30it/s][A
     93%|█████████▎| 1178/1261 [14:24<01:02,  1.33it/s][A
     93%|█████████▎| 1179/1261 [14:24<01:00,  1.36it/s][A
     94%|█████████▎| 1180/1261 [14:25<00:58,  1.38it/s][A
     94%|█████████▎| 1181/1261 [14:26<00:57,  1.39it/s][A
     94%|█████████▎| 1182/1261 [14:26<00:56,  1.41it/s][A
     94%|█████████▍| 1183/1261 [14:27<00:55,  1.41it/s][A
     94%|█████████▍| 1184/1261 [14:28<00:54,  1.42it/s][A
     94%|█████████▍| 1185/1261 [14:29<00:53,  1.42it/s][A
     94%|█████████▍| 1186/1261 [14:29<00:52,  1.42it/s][A
     94%|█████████▍| 1187/1261 [14:30<00:51,  1.42it/s][A
     94%|█████████▍| 1188/1261 [14:31<00:51,  1.43it/s][A
     94%|█████████▍| 1189/1261 [14:31<00:50,  1.43it/s][A
     94%|█████████▍| 1190/1261 [14:32<00:49,  1.43it/s][A
     94%|█████████▍| 1191/1261 [14:33<00:49,  1.43it/s][A
     95%|█████████▍| 1192/1261 [14:34<00:48,  1.41it/s][A
     95%|█████████▍| 1193/1261 [14:34<00:48,  1.39it/s][A
     95%|█████████▍| 1194/1261 [14:35<00:48,  1.37it/s][A
     95%|█████████▍| 1195/1261 [14:36<00:47,  1.38it/s][A
     95%|█████████▍| 1196/1261 [14:36<00:46,  1.39it/s][A
     95%|█████████▍| 1197/1261 [14:37<00:46,  1.39it/s][A
     95%|█████████▌| 1198/1261 [14:38<00:45,  1.39it/s][A
     95%|█████████▌| 1199/1261 [14:39<00:44,  1.38it/s][A
     95%|█████████▌| 1200/1261 [14:39<00:46,  1.32it/s][A
     95%|█████████▌| 1201/1261 [14:40<00:45,  1.32it/s][A
     95%|█████████▌| 1202/1261 [14:41<00:44,  1.33it/s][A
     95%|█████████▌| 1203/1261 [14:42<00:43,  1.34it/s][A
     95%|█████████▌| 1204/1261 [14:42<00:42,  1.33it/s][A
     96%|█████████▌| 1205/1261 [14:43<00:43,  1.30it/s][A
     96%|█████████▌| 1206/1261 [14:44<00:42,  1.30it/s][A
     96%|█████████▌| 1207/1261 [14:45<00:41,  1.31it/s][A
     96%|█████████▌| 1208/1261 [14:46<00:41,  1.29it/s][A
     96%|█████████▌| 1209/1261 [14:46<00:40,  1.30it/s][A
     96%|█████████▌| 1210/1261 [14:47<00:41,  1.24it/s][A
     96%|█████████▌| 1211/1261 [14:48<00:39,  1.27it/s][A
     96%|█████████▌| 1212/1261 [14:49<00:38,  1.28it/s][A
     96%|█████████▌| 1213/1261 [14:50<00:37,  1.28it/s][A
     96%|█████████▋| 1214/1261 [14:50<00:36,  1.29it/s][A
     96%|█████████▋| 1215/1261 [14:51<00:35,  1.30it/s][A
     96%|█████████▋| 1216/1261 [14:52<00:34,  1.32it/s][A
     97%|█████████▋| 1217/1261 [14:52<00:32,  1.34it/s][A
     97%|█████████▋| 1218/1261 [14:53<00:31,  1.36it/s][A
     97%|█████████▋| 1219/1261 [14:54<00:30,  1.38it/s][A
     97%|█████████▋| 1220/1261 [14:55<00:29,  1.39it/s][A
     97%|█████████▋| 1221/1261 [14:55<00:28,  1.40it/s][A
     97%|█████████▋| 1222/1261 [14:56<00:27,  1.41it/s][A
     97%|█████████▋| 1223/1261 [14:57<00:26,  1.41it/s][A
     97%|█████████▋| 1224/1261 [14:57<00:26,  1.41it/s][A
     97%|█████████▋| 1225/1261 [14:58<00:25,  1.41it/s][A
     97%|█████████▋| 1226/1261 [14:59<00:24,  1.42it/s][A
     97%|█████████▋| 1227/1261 [15:00<00:23,  1.42it/s][A
     97%|█████████▋| 1228/1261 [15:00<00:23,  1.42it/s][A
     97%|█████████▋| 1229/1261 [15:01<00:22,  1.42it/s][A
     98%|█████████▊| 1230/1261 [15:02<00:21,  1.42it/s][A
     98%|█████████▊| 1231/1261 [15:02<00:21,  1.41it/s][A
     98%|█████████▊| 1232/1261 [15:03<00:20,  1.42it/s][A
     98%|█████████▊| 1233/1261 [15:04<00:19,  1.42it/s][A
     98%|█████████▊| 1234/1261 [15:04<00:19,  1.41it/s][A
     98%|█████████▊| 1235/1261 [15:05<00:18,  1.42it/s][A
     98%|█████████▊| 1236/1261 [15:06<00:17,  1.40it/s][A
     98%|█████████▊| 1237/1261 [15:07<00:17,  1.38it/s][A
     98%|█████████▊| 1238/1261 [15:07<00:16,  1.39it/s][A
     98%|█████████▊| 1239/1261 [15:08<00:15,  1.40it/s][A
     98%|█████████▊| 1240/1261 [15:09<00:15,  1.39it/s][A
     98%|█████████▊| 1241/1261 [15:09<00:14,  1.40it/s][A
     98%|█████████▊| 1242/1261 [15:10<00:13,  1.41it/s][A
     99%|█████████▊| 1243/1261 [15:11<00:12,  1.42it/s][A
     99%|█████████▊| 1244/1261 [15:12<00:11,  1.42it/s][A
     99%|█████████▊| 1245/1261 [15:12<00:11,  1.41it/s][A
     99%|█████████▉| 1246/1261 [15:13<00:10,  1.41it/s][A
     99%|█████████▉| 1247/1261 [15:14<00:10,  1.39it/s][A
     99%|█████████▉| 1248/1261 [15:15<00:09,  1.37it/s][A
     99%|█████████▉| 1249/1261 [15:15<00:08,  1.36it/s][A
     99%|█████████▉| 1250/1261 [15:16<00:08,  1.34it/s][A
     99%|█████████▉| 1251/1261 [15:17<00:07,  1.36it/s][A
     99%|█████████▉| 1252/1261 [15:17<00:06,  1.37it/s][A
     99%|█████████▉| 1253/1261 [15:18<00:05,  1.36it/s][A
     99%|█████████▉| 1254/1261 [15:19<00:05,  1.36it/s][A
    100%|█████████▉| 1255/1261 [15:20<00:04,  1.36it/s][A
    100%|█████████▉| 1256/1261 [15:20<00:03,  1.37it/s][A
    100%|█████████▉| 1257/1261 [15:21<00:02,  1.38it/s][A
    100%|█████████▉| 1258/1261 [15:22<00:02,  1.39it/s][A
    100%|█████████▉| 1259/1261 [15:23<00:01,  1.39it/s][A
    100%|█████████▉| 1260/1261 [15:23<00:00,  1.39it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_output.mp4 
    



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))
```





<video width="960" height="540" controls>
  <source src="project_video_output.mp4">
</video>




### Discussion and Closing Thoughts

This video detection pipeline works well, but it's far from perfect. Some general thoughts from my many trial-and-error attempts:

- Faraway vehicles and extremely close-up vehicles aren't always labeled. If I had more time, I'd upsample these images in the training data.

- Ensembling seems to have some potential here. While YCrCb worked the best, some of the other color spaces I tried had their moments. An ensemble prediction model would involved training multiple classifiers using different HOG feature vectors, predict using all of them, and aggregate using the same method as above, probably with a higher threshold. I think this would actually do quite well; the weaker performance on totally out-of-sample data suggests some degree of overfitting, which ensembling is meant to help rectify. The final step of the current pipelin, where we aggregate consecutive labels and filter out weaker signals, has a similar effect to ensembling, but it would still be interesting to see where this gets us.  

The biggest challenge of this project was the disconnect between classifier performance and actual video detection quality. A higher accuracy/precision/recall did not always translate into better video labels. The multiple sets of hyperparameters also added another layer of complexity to determining which direction to proceed.  

Being equipped with a mere CPU for this process made re-training and video processing a very time-consuming process. Grid search over all the parameters was out of the question, and I had to rely on intuition and heuristics to narrow down the set of optimal parameters. 


```python

```
