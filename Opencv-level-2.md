# OpenCV - level - 2
OpenCV repository


## Comprehensive list of openCV functions - level - 2




### Drawing polyline on facial landmark

```
def drawPolyline(im, landmarks, start, end, isClosed=False):
  points = []
  for i in range(start, end+1):
    point = [landmarks.part(i).x, landmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(im, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)
```


### RenderFace for 70 points facial landmark

```
# Use this function for 70-points facial landmark detector model
def renderFace(im, landmarks):
    assert(landmarks.num_parts == 68)
    drawPolyline(im, landmarks, 0, 16)           # Jaw line
    drawPolyline(im, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(im, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im, landmarks, 27, 30)          # Nose bridge
    drawPolyline(im, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(im, landmarks, 36, 41, True)    # Left eye
    drawPolyline(im, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(im, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(im, landmarks, 60, 67, True)    # Inner lip
```



### RenderFace for 70 points facial landmark using polylines

```
# Use this function for 70-points facial landmark detector model
def renderFace(im, landmarks):
    assert(landmarks.num_parts == 68)
    drawPolyline(im, landmarks, 0, 16)           # Jaw line
    drawPolyline(im, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(im, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im, landmarks, 27, 30)          # Nose bridge
    drawPolyline(im, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(im, landmarks, 36, 41, True)    # Left eye
    drawPolyline(im, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(im, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(im, landmarks, 60, 67, True)    # Inner lip
```


### RenderFace for 70 points facial landmark using circles
```

# Use this function for any model other than
# 70 points facial_landmark detector model

def renderFace2(im, landmarks, color=(0, 255, 0), radius=3):
  for p in landmarks.parts():
    cv2.circle(im, (p.x, p.y), radius, color, -1)

```

### Landmark detection training
```
Requires the XML files with landmarks annotation
training_with_face_landmarks.xml
testing_with_face_landmarks.xml

dlib models are available in the below location

http://dlib.net/files/
https://github.com/davisking/dlib-models/tree/master/gender-classifier

Data files 
https://www.dropbox.com/s/e2wa5wf4vpe2kni/facial_landmark_data.zip?dl=1


import os
import sys
import dlib

print("USAGE : python trainFLD.py <path to facial_landmark_data folder> <number of points>")

# Default values
fldDatadir = "./data/facial_landmark_data"
numPoints = 70

if len(sys.argv) == 2:
  fldDatadir = sys.argv[1]
if len(sys.argv) == 3:
  fldDatadir = sys.argv[1]
  numPoints = sys.argv[2]

modelName = 'shape_predictor_{}_face_landmarks.dat'.format(numPoints)

options = dlib.shape_predictor_training_options()
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.tree_depth = 4
options.nu = 0.1
options.oversampling_amount = 20
options.feature_pool_size = 400
options.feature_pool_region_padding = 0
options.lambda_param = 0.1
options.num_test_splits = 20

# Tell the trainer to print status messages to the console so we can
# see training options and how long the training will take.
options.be_verbose = True


trainingXmlPath = os.path.join(fldDatadir, "training_with_face_landmarks.xml")
testingXmlPath = os.path.join(fldDatadir, "testing_with_face_landmarks.xml")
outputModelPath = os.path.join(fldDatadir, modelName)

# check whether path to XML files is correct
if os.path.exists(trainingXmlPath) and os.path.exists(testingXmlPath):

  # dlib.train_shape_predictor() does the actual training.  It will save the
  # final predictor to predictor.dat.  The input is an XML file that lists the
  # images in the training dataset and also contains the positions of the face
  # parts.
  dlib.train_shape_predictor(trainingXmlPath, outputModelPath, options)

  # Now that we have a model we can test it.  dlib.test_shape_predictor()
  # measures the average distance between a face landmark output by the
  # shape_predictor and ground truth data.

  print("\nTraining accuracy: {}".format(
    dlib.test_shape_predictor(trainingXmlPath, outputModelPath)))

  # The real test is to see how well it does on data it wasn't trained on.
  print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testingXmlPath, outputModelPath)))
else:
  print('training and test XML files not found.')
  print('Please check paths:')
  print('train: {}'.format(trainingXmlPath))
  print('test: {}'.format(testingXmlPath))

```


### Lucas Kanade - Opticalflow
```
one way to implement stabilization requires optical flow calculation. Fortunately, OpenCV has a good implementation of Lukas Kanade optical flow described in this section. It can be invoked using calcOpticalFlowPyrLK.

nextPts, status, err    =   cv.calcOpticalFlowPyrLK(    prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]  )
Where,

prevImg - first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
nextImg - second input image or pyramid of the same size and the same type as prevImg.
prevPts - vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.
nextPts - output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
status - output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
err - output vector of errors; each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
winSize - size of the search window at each pyramid level.
maxlevel - 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel.
criteria - parameter, specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon.
flags - operation flags:
OPTFLOW_USE_INITIAL_FLOW - uses initial estimations, stored in nextPts; if the flag is not set, then prevPts is copied to nextPts and is considered the initial estimate.
OPTFLOW_LK_GET_MIN_EIGENVALS - use minimum eigen values as an error measure (see minEigThreshold description); if the flag is not set, then L1 distance between patches around the original and a moved point, divided by number of pixels in a window, is used as a error measure.
minEigThreshold - the algorithm calculates the minimum eigen value of a 2x2 normal matrix of optical flow equations (this matrix is called a spatial gradient matrix), divided by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding feature is filtered out and its flow is not processed, so it allows to remove bad points and get a performance boost.
As mentioned earlier, optical flow computation requires building Image pyramids for the current frame and the previous frame. calcOpticalFlowPyrLKdoes this calculation internally when you pass the previous frame and the current frame. When using optical flow in a video, the image pyramid for the same frame is built twice -- once while doing optical flow calculation for the current frame and the other for the next frame. This double calculation can be avoided by building and storing the image pyramid for every frame and passing it to calcOpticalFlowPyrLK.

The most common usage of buildOpticalFlowPyramid is shown below.

retval, pyramid =   cv.buildOpticalFlowPyramid( img, winSize, maxLevel[, pyramid[, withDerivatives[, pyrBorder[, derivBorder[, tryReuseInputImage]]]]]  )
Where,

img - 8-bit input image.
pyramid - output pyramid.
winSize - window size of optical flow algorithm. Must be not less than winSize argument of calcOpticalFlowPyrLK. It is needed to calculate required padding for pyramid levels.
maxLevel - 0-based maximal pyramid level number.
withDerivatives - set to precompute gradients for the every pyramid level. If pyramid is constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.
pyrBorder - the border mode for pyramid layers.
derivBorder - the border mode for gradients.
tryReuseInputImage - put ROI of input image into the pyramid if possible. You can pass false to force data copying.
```

