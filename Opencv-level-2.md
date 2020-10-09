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

### 33 Points
```
points33Indices = [
                   1, 3, 5, 8, 11, 13, 15,     # Jaw line
                   17, 19, 21,                 # Left eyebrow
                   22, 24, 26,                 # Right eyebrow
                   30, 31,                     # Nose bridge
                   33, 35,                     # Lower nose
                   36, 37, 38, 39, 40, 41,     # Left eye
                   42, 43, 44, 45, 46, 47,     # Right Eye
                   48, 51, 54, 57              # Outer lip
                  ]

```

### Create train / test Xmls
```
import sys
import os
import random
try:
  from lxml import etree as ET
except ImportError:
  print('install lxml using pip')
  print('pip install lxml')

# create XML from annotations
def createXml(imageNames, xmlName, numPoints):
  # create a root node names dataset
  dataset = ET.Element('dataset')
  # create a child node "name" within root node "dataset"
  ET.SubElement(dataset, "name").text = "Training Faces"
  # create another child node "images" within root node "dataset"
  images = ET.SubElement(dataset, "images")

  # print information about xml filename and total files
  numFiles = len(imageNames)
  print('{0} : {1} files'.format(xmlName, numFiles))

  # iterate over all files
  for k, imageName in enumerate(imageNames):
    # print progress about files being read
    print('{}:{} - {}'.format(k+1, numFiles, imageName))

    # read rectangle file corresponding to image
    rect_name = os.path.splitext(imageName)[0] + '_rect.txt'
    with open(os.path.join(fldDatadir, rect_name), 'r') as file:
      rect = file.readline()
    rect = rect.split()
    left, top, width, height = rect[0:4]

    # create a child node "image" within node "images"
    # this node will have annotation data for an image
    image = ET.SubElement(images, "image", file=imageName)
    # create a child node "box" within node "image"
    # this node has values for bounding box or rectangle of face
    box = ET.SubElement(image, 'box', top=top, left=left, width=width, height=height)

    # read points file corresponding to image
    points_name = os.path.splitext(imageName)[0] + '_bv' + numPoints + '.txt'
    with open(os.path.join(fldDatadir, points_name), 'r') as file:
      for i, point in enumerate(file):
        x, y = point.split()
        # points annotation file has coordinates in float
        # but we want them to be in int format
        x = str(int(float(x)))
        y = str(int(float(y)))
        # name is the facial landmark or point number, starting from 0
        name = str(i).zfill(2)
        # create a child node "parts" within node "box"
        # this node has values for facial landmarks
        ET.SubElement(box, 'part', name=name, x=x, y=y)

  # finally create an XML tree
  tree = ET.ElementTree(dataset)

  print('writing on disk: {}'.format(xmlName))
  # write XML file to disk. pretty_print=True indents the XML to enhance readability
  tree.write(xmlName, pretty_print=True, xml_declaration=True, encoding="UTF-8")


if __name__ == '__main__':

  # Default values
  fldDatadir = "../data/facial_landmark_data"
  numPoints = 70

  if len(sys.argv) == 2:
    # facial landmark data directory
    fldDatadir = sys.argv[1]
  if len(sys.argv) == 3:
    # and number of facial landmarks
    numPoints = sys.argv[2]

  # Read names of all images
  with open(os.path.join(fldDatadir, 'image_names.txt')) as d:
    imageNames = [x.strip() for x in d.readlines()]

  ################# trick to use less data #################
  # If you are unable to train all images on your machine,
  # you can reduce training data by randomly sampling n
  # images from the total list.
  # Keep decreasing the value of n from len(imageNames) to
  # a value which works on your machine.
  # Uncomment the next two lines to decrease training data
  # n = 1000
  # imageNames = random.sample(imageNames, n)
  ##########################################################

  totalNumFiles = len(imageNames)
  # We will split data into 95:5 for train and test
  numTestFiles = int(0.05 * totalNumFiles)

  # randomly sample 5% items from list of image names
  testFiles = random.sample(imageNames, numTestFiles)
  # assign rest of image names as train
  trainFiles = list(set(imageNames) - set(testFiles))

  # generate XML files for train and test data
  createXml(trainFiles, os.path.join(fldDatadir, 'training_with_face_landmarks.xml'), numPoints)
  createXml(testFiles, os.path.join(fldDatadir, 'testing_with_face_landmarks.xml'), numPoints)
```


### Returns 8 points on the boundary of a rectangle
```
def getEightBoundaryPoints(h, w):
  boundaryPts = []
  boundaryPts.append((0,0))
  boundaryPts.append((w/2, 0))
  boundaryPts.append((w-1,0))
  boundaryPts.append((w-1, h/2))
  boundaryPts.append((w-1, h-1))
  boundaryPts.append((w/2, h-1))
  boundaryPts.append((0, h-1))
  boundaryPts.append((0, h/2))
  return np.array(boundaryPts, dtype=np.float)

```

### Constrains points to be inside boundary
```
def constrainPoint(p, w, h):
  p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
  return p
```

### Convert Dlib shape detector object to list of tuples
```
def dlibLandmarksToPoints(shape):
  points = []
  for p in shape.parts():
    pt = (p.x, p.y)
    points.append(pt)
  return points
```

### Compute similarity transform given two sets of two points.
```
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()

  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

  inPts.append([np.int(xin), np.int(yin)])

  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

  outPts.append([np.int(xout), np.int(yout)])

  # Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
  return tform

```

### Normalizes a facial image to a standard size given by outSize.
```
# Normalization is done based on Dlib's landmark points passed as pointsIn
# After normalization, left corner of the left eye is at (0.3 * w, h/3 )
# and right corner of the right eye is at ( 0.7 * w, h / 3) where w and h
# are the width and height of outSize.
def normalizeImagesAndLandmarks(outSize, imIn, pointsIn):
  h, w = outSize

  # Corners of the eye in input image
  eyecornerSrc = [pointsIn[36], pointsIn[45]]

  # Corners of the eye in normalized image
  eyecornerDst = [(np.int(0.3 * w), np.int(h/3)),
                  (np.int(0.7 * w), np.int(h/3))]

  # Calculate similarity transform
  tform = similarityTransform(eyecornerSrc, eyecornerDst)
  imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

  # Apply similarity transform to input image
  imOut = cv2.warpAffine(imIn, tform, (w, h))

  # reshape pointsIn from numLandmarks x 2 to numLandmarks x 1 x 2
  points2 = np.reshape(pointsIn, (pointsIn.shape[0], 1, pointsIn.shape[1]))

  # Apply similarity transform to landmarks
  pointsOut = cv2.transform(points2, tform)

  # reshape pointsOut to numLandmarks x 2
  pointsOut = np.reshape(pointsOut, (pointsIn.shape[0], pointsIn.shape[1]))

  return imOut, pointsOut
  
```
