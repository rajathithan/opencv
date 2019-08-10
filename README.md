# OpenCV
OpenCV repository


## Comprehensive list of openCV functions




### Image as a matrix

```
retval  =   cv2.imread( filename[, flags]   )
It has 2 arguments:

retval is the image if it is successfully loaded. Otherwise it is None. This may happen if the filename is wrong or the file is corrupt.
Path of the image file: This can be an absolute or relative path. This is a mandatory argument.
Flags: These flags are used to read an image in a particular format (for example, grayscale/color/with alpha channel). This is an optional argument with a default value of cv2.IMREAD_COLOR or 1 which loads the image as a color image.

Flags

cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_UNCHANGED or -1: Loads image as such including alpha channel.
```


### Image Properties
```
testImage = cv2.imread(imagePath,0)

print("Data type = {}\n".format(testImage.dtype))
print("Object type = {}\n".format(type(testImage)))
print("Image Dimensions = {}\n".format(testImage.shape))

Data type = uint8

Object type = <class 'numpy.ndarray'>

Image Dimensions = (13, 11)


```


### Matplotlib's & opencv's imshow

```
plt.imshow( mat )

mat - Image to be displayed.

cv2.imshow( winname, mat )

winname - Name of the window.
mat - Image to be displayed.

```


### Display utilities

```
cv2.namedWindow(    winname[, flags]    )
winname - Name of the window in the window caption that may be used as a window identifier.
flags - Flags of the window. The supported flags are: (cv::WindowFlags)


cv2.waitKey(    [, delay]   )
delay - Delay in milliseconds. 0 is the special value that means "forever".


cv2.destroyWindow(  winname )
winname - Name of the window to be destroyed


cv2.destroyAllWindows()

```


### Write to disk

```
cv2.imwrite(    filename, img [, params]    )

filename - String providing the relative or absolute path where the image should be saved.
img - Image matrix to be saved.
params - Additional information, like specifying the JPEG compression quality etc

```
(https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac)


### Read & convert image

```
img = cv2.imread("filename.jpg")

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB,cmap='gray')

or

plt.imshow(img[:,:,::-1],cmap='gray')

```


### Split and Merge

```
b,g,r = cv2.split(img)


imgMerged = cv2.merge((b,g,r))

```


### Empty numpy array

```
image = cv2.imread("filename.jpg")

emptyMatrix = np.zeros((100,200,3),dtype='uint8')
plt.imshow(emptyMatrix)

emptyMatrix = 255*np.ones((100,200,3),dtype='uint8')
plt.imshow(emptyMatrix)

emptyOriginal = 100*np.ones_like(image)
plt.imshow(emptyOriginal)


```


### Resizing the image

```
dst =   cv2.resize( src, dsize[, dst[, fx[, fy[, interpolation]]]]  )

src - input image
dst - output resized image
dsize - output image size
fx - scale factor along the horizontal axis;
fy - scale factor along the vertical axis; Either dsize or both fx and fy must be non-zero.
interpolation - interpolation method ( Bilinear / Bicubic etc ).

Methods:
1.Specify width and height of output image explicitly
cv2.resize(image, (resizeWidth, resizeHeight), interpolation= cv2.INTER_LINEAR)

2.Specify the scaling factors for resizing ( for both width and height )
cv2.resize(image, None, fx= scalex, fy= scaley, interpolation= cv2.INTER_LINEAR)

```


### Mask using co-ordinates

```
mask1 = np.zeros_like(image)
plt.imshow(mask1)

mask1[50:200,170:320] = 255

```


### Create a mask using pixel intensity or color

```
The color with focus to be given  high intensity with a value of 100 to 255
The color with low focus to be given low intensity with a value of 0 to 100

inRange - provides a binary output with white pixels which falls within range and black pixels for out of range. 

mask2 = cv2.inRange(    src, lowerb, upperb[, dst]  )

mask2 = cv2.inRange(image, (0,0,150), (100,100,255))
```


### Datatype conversion
```
Convert to float
================
image = cv2.imread("filename.jpg")
scalingfactor = 1/255.0

unsigned int to float
=====================
image = np.float32(image)
image = image * scalingfactor

Convert back to unsigned int
============================
image = image * (1.0/scalingFactor)
image = np.uint8(image)

```


### Contrast Enhancement
```
Io = αI
α - scalingfactor
contrastPercentage = 30

If the image is in float datatype, then the range should be [0,1]. Anything above 255 is considered as white.
If you want to keep the image in float format, then Normalize the instensity values so that it lies in [0,1]

contrastImage = image * (1+contrastPercentage/100)
clippedContrastImage = np.clip(contrastImage, 0, 255)
contrastHighClippedUint8 = np.uint8(clippedContrastImage)

If the image is in int datatype, then the range should be [0,255]
Clip the intensity values to 0 ~ 255 and change the data type to uint8.

contrastHighNormalized = (image * (1+contrastPercentage/100))/255
contrastHighNormalized01Clipped = np.clip(contrastHighNormalized,0,1)

```


### Brightness Enhancement
```
Brightness is a measure of light falling on the scene. In RGB color space, it can be thought of as the arithmetic mean of the R, G and B color values. To make an image brighter, the intensity values should be increased by some offset (  β  ) and vice-versa.

If  I  is the input image, and  Io  is the output image, brightness enhanced image is given by the equation

Io=I+β

brightnessoffset = 50

# Add the offset for increasing brightness
brightHigh = image + brightnessOffset

plt.imshow(brightHigh[...,::-1])

use cv2.add or np.clip to prevent overflow

int
brightHighOpenCV = cv2.add(image, np.ones(image.shape,dtype='uint8')*brightnessOffset)
brightHighInt32 = np.int32(image) + brightnessOffset
brightHighInt32Clipped = np.clip(brightHighInt32,0,255)

float
brightHighFloat32 = np.float32(image) + brightnessOffset
brightHighFloat32NormalizedClipped = np.clip(brightHighFloat32/255,0,1)
brightHighFloat32ClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)

```

### Bitwise Operations
```
AND operation: cv2.bitwise_and

OR operation: cv2.bitwise_or

NOT operation: cv2.bitwise_not

XOR operation: cv2.bitwise_xor

dst    =    cv2.bitwise_XXX(    src1, src2[, dst[, mask]]    )

Parameters

src1 - first input.
src2 - second input.
dst - output array that has the same size and type as the input array.
mask - optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed. The operation is applied only on those pixels of the input images where the mask is non-zero.


Here's a cheat sheet on the input and output table for these bitwise operations.

OperationInput-1 Input-2	Output
AND	0	0	0
AND	0	1	0
AND	1	0	0
AND	1	1	1
OR	0	0	0
OR	0	1	1
OR	1	0	1
OR	1	1	1
NOT	0	NA	1
NOT	1	NA	0
XOR	0	0	0
XOR	0	1	1
XOR	1	0	1
XOR	1	1	0

They will be highly useful while extracting any part of the image , defining and working with non-rectangular ROI etc.


```


## Draw a Line

```
line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
The arguments that we will focus on are:

img: Image on which we will draw a line
pt1: First point(x,y location) of the line segment
pt2: Second point of the line segment
color: Color of the line which will be drawn
The above arguments are compulsory. Other arguments that are important for us to know and are optional are:

thickness: Integer specifying the line thickness. Default value is 1.
lineType: Type of the line. Default value is 8 which stands for an 8-connected line. Usually, cv2.LINE_AA (antialiased or smooth line) is used for the lineType.

cv2.line(imageLine, (200, 80), (280, 80), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA);

```


## Draw a circle

```
circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
First, the mandatory arguments:

img: Image where the circle is drawn.
center: Center of the circle.
radius: Radius of the circle.
color: Circle color
Next, let's check out the (optional) arguments which we are going to use quite extensively.

thickness: Thickness of the circle outline (if positive). If a negative value is supplied for this argument, it will result in a filled circle.
lineType: Type of the circle boundary. This is exact same as lineType argument in cv2.line

cv2.circle(imageCircle, (250, 125), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA);

cv2.circle(imageFilledCircle, (250, 125), 100, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA);

```


## Draw an Ellipse

```
ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> img
The mandatory arguments are as follows.

img: Image on which the ellipse is to be drawn.
center: Center of the ellipse.
axes: radius of the ellipse major and minor axes.
angle: Ellipse rotation angle in degrees.
startAngle: Starting angle of the elliptic arc in degrees.
endAngle: Ending angle of the elliptic arc in degrees.
color: Ellipse line color
The optional arguments that we are going to use are the same as before and carry the same meaning.

starting angle is 0 and ending angle is 360

cv2.ellipse(imageEllipse, (250, 125), (100, 50), 0, 0, 360, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA);
cv2.ellipse(imageEllipse, (250, 125), (100, 50), 90, 0, 360, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA);

starting angle = 180 and ending angle = 360 - Incomplete open eclipse

cv2.ellipse(imageEllipse, (250, 125), (100, 50), 0, 180, 360, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA);

starting angle = 0 and ending angle = 180 - Incomplete filled eclipse

cv2.ellipse(imageEllipse, (250, 125), (100, 50), 0, 0, 180, (0, 0, 255), thickness=-2, lineType=cv2.LINE_AA);

```

## Draw a rectangle

```
rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
The mandatory arguments are as follows.

img: Image on which the rectangle is to be drawn.
pt1: Vertex of the rectangle. Usually we use the top-left vertex here.
pt2: Vertex of the rectangle opposite to pt1. Usually we use the bottom-right vertex here.
color: Rectangle color
The optional arguments that we are going to use are same as before.

We need two points to draw a rectangle. These are the opposite vertices of the rectangle. From the sample image, we can approximately find the vertices as

top-left - (170,50)
bottom-right - (300,200)

cv2.rectangle(imageRectangle, (170, 50), (300, 200), (255, 0, 255), thickness=5, lineType=cv2.LINE_8);

```

## Add Text

```
putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
The mandatory arguments that we need to focus on are:

img: Image on which the text has to be written.
text: Text string to be written.
org: Bottom-left corner of the text string in the image.
fontFace: Font type
fontScale: Font scale factor that is multiplied by the font-specific base size.
color: Font color
The optional arguments that we are going to use are same as before.

text = "I am studying"
fontScale = 1.5
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontColor = (250, 10, 10)
fontThickness = 2
cv2.putText(imageText, text, (20, 350), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

Get font size from pixel height of text 

fontScale   =   cv2.getFontScaleFromHeight( fontFace, pixelHeight[, thickness]  )
Parameters

fontFace - Font to use
pixelHeight - Pixel height to compute the fontScale for
thickness - Thickness of lines used to render the text. See putText for details
fontScale (Output) - The fontsize to use in cv2.putText() function.

pixelHeight = 20

# Calculate the fontScale
fontScale = cv2.getFontScaleFromHeight(fontFace, pixelHeight, fontThickness)
print("fontScale = {}".format(fontScale))

imageTextFontScale = image.copy()
cv2.putText(imageTextFontScale, text, (20, 350), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

Get height and width of text

textSize, baseLine  =   cv2.getTextSize(    text, fontFace, fontScale, thickness    )
Parameters

text - Input text string.
fontFace - Font to use, see HersheyFonts.
fontScale - Font scale factor that is multiplied by the font-specific base size.
thickness - Thickness of lines used to render the text. See putText for details.
baseLine (Output) - y-coordinate of the baseline relative to the bottom-most text point. In our example, this value will be the difference in height of the bottom-most tip of y and i
textSize (Output) - The text size (width, height)


imageGetTextSize = image.copy()
imageHeight, imageWidth=imageGetTextSize.shape[:2]

# Get the text box height and width and also the baseLine
textSize, baseLine = cv2.getTextSize(text,fontFace,fontScale,fontThickness)
textWidth,textHeight = textSize
print("TextWidth = {}, TextHeight = {}, baseLine = {}".format(textWidth, textHeight, baseLine))

# Get the coordinates of text box bottom left corner
# The xccordinate will be such that the text is centered
xcoordinate = (imageWidth - textWidth)//2
# The y coordinate will be such that the entire box is just 10 pixels above the bottom of image
ycoordinate = (imageHeight - baseLine - 10)
print("TextBox Bottom Left = ({},{})".format(xcoordinate,ycoordinate))

# Draw the Canvas using a filled rectangle
canvasColor = (255, 255, 255)
canvasBottomLeft = (xcoordinate,ycoordinate+baseLine)
canvasTopRight = (xcoordinate+textWidth, ycoordinate-textHeight)
cv2.rectangle(imageGetTextSize, canvasBottomLeft, canvasTopRight, canvasColor, thickness=-1);
print("Canvas Bottom Left = {}, Top Right = {}".format(canvasBottomLeft,canvasTopRight))

# Now draw the baseline ( just for reference )
lineThickness = 2
lineLeft = (xcoordinate, ycoordinate)
lineRight = (xcoordinate+textWidth, ycoordinate)
lineColor = (0,255,0)
cv2.line(imageGetTextSize, lineLeft, lineRight, lineColor, thickness = lineThickness, lineType=cv2.LINE_AA);

# Finally Draw the text
cv2.putText(imageGetTextSize, text, (xcoordinate,ycoordinate), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

```


## Video Capture
```
cv2.VideoCapture to create a VideoCapture object and read from input file (video).

Function Syntax 
<VideoCapture object>   =   cv2.VideoCapture(        )
<VideoCapture object>   =   cv2.VideoCapture(    filename[, apiPreference]   )
<VideoCapture object>   =   cv2.VideoCapture(    index[, apiPreference]  )
Parameters

filename it can be:
name of video file (eg. video.avi)
or image sequence (eg. img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...) -or URL of video stream (eg. protocol://host:port/script_name?script_params|auth). Note that each video stream or IP camera feed has its own URL scheme. Please refer to the documentation of source stream to know the right URL.
apiPreference: preferred Capture API backends to use. Can be used to enforce a specific reader implementation if multiple are available: e.g. cv::CAP_FFMPEG or cv::CAP_IMAGES or cv::CAP_DSHOW.
```


## Video set and get properties
```
cap.get(propId)
cap.set(propId,value)

cap is the VideoCapture object from where we want to extract (or set) the properties, propId stands for the Property ID and value is the value we want to set for the property with id propId.

Here are some of the common properties and their ID.

Enumerator	Numerical Value	Property
cv2.CAP_PROP_POS_MSEC	0	Current position of the video file in milliseconds
cv2.CAP_PROP_FRAME_WIDTH	3	Width of the frames in the video stream
cv2.CAP_PROP_FRAME_HEIGHT	4	Height of the frames in the video stream
cv2.CAP_PROP_FPS	5	Frame rate
cv2.CAP_PROP_FOURCC	6	4-character code of codec
```

## Video Writer
```
Create a VideoWriter object

Function Syntax 
<VideoWriter object>    =   cv2.VideoWriter(     )
<VideoWriter object>    =   cv2.VideoWriter( filename, fourcc, fps, frameSize[, isColor] )
<VideoWriter object>    =   cv2.VideoWriter( filename, apiPreference, fourcc, fps, frameSize[, isColor]  )
Parameters

filename: Name of the output video file.
fourcc: 4-character code of codec used to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc. List of codes can be obtained at Video Codecs by FOURCC page. FFMPEG backend with MP4 container natively uses other values as fourcc code: see ObjectType, so you may receive a warning message from OpenCV about fourcc code conversion.
fps: Framerate of the created video stream.
frameSize: Size of the video frames.
isColor: If it is not zero, the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only).
2. Write frames to the object in a loop.

3. Close and release the object.

out = cv2.VideoWriter('outputChaplin.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
    
  if ret == True:
    
    # Write the frame into the file 'outputChaplin.avi'
    out.write(frame)
    
    # Wait for 25 ms before moving on to the next frame
    cv2.imshow("Frame",frame)
    cv2.waitKey(25)
    
  # Break the loop
  else: 
    break
    
# When everything done, release the VideoCapture and VideoWriter objects
cap.release()
out.release()

```


## Mouse for Annotation
```
cv2.setMouseCallback

cv.setMouseCallback(winname, onMouse, userdata  )
Parameters

winname - Name of the window.
onMouse - Callback function for mouse events.
userdata - The optional parameter passed to the callback.

cv2.setMouseCallback("Window", functionName)
functionName(action, x, y, flags, userdata)
action==cv2.EVENT_LBUTTONDOWN
action==cv2.EVENT_LBUTTONUP
```

## Trackbar 
```
For creating trackbars, we have to specify a named window and use the cv2.createTrackbar() function in which we need to specify the window name. A callback function needs to be specified for detecting events on the trackbar. Let’s see an example code.

Let's first focus on the callback functions.

The trackbars are created using the createTrackbar function. The different parameters of the function are given below.

cv2.createTrackbar(trackbarName, windowName, value, count, onChange)

trackbarname is the name that will be displayed alongside the trackbar
windowName is the namedWindow associated with the callback function
value is a pointer to an integer variable whose value indicates the position of the trackbar
Count is the maximum position of the trackbar, minimum being 0 always
onChange is the callback function which is associated with the winname window and gets triggered when the trackbar is accessed by the user
# Create Trackbar to choose scale percentage
cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)

# Create Trackbar to choose tyoe of scaling ( Up or Down )
cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleImage)

# Get Tracjbar position
cv2.getTrackbarPos(trackbarType, windowName)
```


## Thresholding using Loop
```
def thresholdUsingLoop(src, thresh, maxValue):
    # Create a output image
    dst = src.copy()
    height,width = src.shape[:2]

    # Loop over rows
    for i in range(height):
        # Loop over columns
        for j in range(width):
            if src[i,j] > thresh:
                dst[i,j] = maxValue
            else:
                dst[i,j] = 0
                
    return dst

t = time.time()
dst = thresholdUsingLoop(src, thresh, maxValue)
```


## Thresholding using Vectorized Operations
```
def thresholdUsingVectors(src, thresh, maxValue):
    # Create a black output image ( all zeros )
    dst = np.zeros_like(src)
    
    # Find pixels which have values>threshold value
    thresholdedPixels = src>thresh
    
    # Assign those pixels maxValue
    dst[thresholdedPixels] = maxValue
    
    return dst

t = time.time()
dst = thresholdUsingVectors(src, thresh, maxValue)
```


## Thresholding using OpenCV function
```
cv2.threshold has the following syntax :

retval, dst = cv.threshold(src, thresh, maxval, type[, dst])

Where,

Input:

src is the input array ot image (multiple-channel, 8-bit or 32-bit floating point).
thresh is the threshold value.
maxval is the maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
type is thethresholding type ( THRESH_BINARY, THRESH_BINARY_INV, etc )
Output:

dst is the output array or image of the same size and type and the same number of channels as src.
retval is the threshold value if you use other thresholding types such as Otsu or Triangle

thresh = 100
maxValue = 150 

Threshold Binary
th, dst_bin = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
Convert any pixel above the threshold of 100 to 150 and below the threshold to zero.

Threshold Binary Inverse
th, dst_bin_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV)
Convert any pixel above the threshold of 100 to 0 and below or equal to the threshold is set to 150 max_value.

Threshold truncate
th, dst_trunc = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TRUNC)
Convert any pixel above or equal to the maxvalue is made equal to the threshold.

Threshold to zero
th, dst_to_zero = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO)
Convert any pixel above the threshold of 100 will maintain its pixel value and below the threshold to zero.

Threshold to zero Inverse
th, dst_to_zero_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO_INV)
Convert any pixel above the threshold to zero and below the threshold to maintain its pixel value.
```


## Morphological Operations - Dilasion - Increase white pixels in a Binary image // Erosion - Erode white pixels in a Binary image
```
Dilation
Function Syntax
dst    =    cv.dilate(    src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]    )

Erosion
Function Syntax
dst    =    cv.erode(    src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]    )

Parameters

Both functions take the same set of arguments

src input image; the number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst output image of the same size and type as src.
kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular structuring element is used.

# Get structuring element/kernel which will be used for dilation
kSize = (7,7)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
plt.imshow(kernel1)

cv2.dilate(image, kernel1, iterations=1)

cv2.erode(image, kernel1)



anchor position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
iterations number of times dilation is applied.
borderType pixel extrapolation method.
borderValue border value in case of a constant border
Note: In the functions above, the parameter ‘iterations’ is optional and if not mentioned default is taken as 1. In case, we need to run the dilate/erode function n number of times we specify "iterations = n" in the function parameter list.
```


## Opening ( First Erosion , Then Dilation ) / Closing ( First Dilation, Then Erosion )
```
Opening refers Erosion followed by Dilation and these operations is used for clearing white blobs and Closing refers Dilation followed by Erosion and are used for clearing black holes

Opening
python:

imageMorphOpened = cv2.morphologyEx( src, cv2.MORPH_OPEN, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )

Closing
python:
imageMorphOpened = cv2.morphologyEx( src, cv2.MORPH_CLOSE, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )

Parameters

src - Source image. The number of channels can be arbitrary. The depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst - Destination image of the same size and type as source image.
op - Type of a morphological operation
kernel - Structuring element. It can be created using getStructuringElement.
anchor - Anchor position with the kernel. Negative values mean that the anchor is at the kernel center.
iterations - Number of times erosion and dilation are applied.
borderType - Pixel extrapolation method.
borderValue - Border value in case of a constant border. The default value has a special meaning.

# Get structuring element/kernel which will be used 
# for opening operation
openingSize = 3

# Selecting a elliptical kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
            (2 * openingSize + 1, 2 * openingSize + 1),
            (openingSize,openingSize))
            
imageMorphOpened = cv2.morphologyEx(image, cv2.MORPH_OPEN, 
                        element,iterations=3)
------------------------                        
# Get structuring element/kernel 
# which will be used for closing operation
closingSize = 10

# Selecting an elliptical kernel 
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2 * closingSize + 1, 2 * closingSize + 1),
            (closingSize,closingSize))

imageMorphClosed = cv2.morphologyEx(image,
                                    cv2.MORPH_CLOSE, element)
```


## Connected Component Analysis
```
It is a fancy name for labeling blobs in a binary image. So, it can also be used to count the number of blobs ( also called connected components ) in a binary image. (Two-pass algorithm - To label blobs based on white pixels on first pass, rename the labels for adjacent pixels (Connected pixels) in second pass

# Threshold Image
th, imThresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

# Find connected components
_, imLabels = cv2.connectedComponents(imThresh)
plt.imshow(imLabels)

Display Individual Label
# Display the labels
nComponents = imLabels.max()

displayRows = np.ceil(nComponents/3.0)
plt.figure(figsize=[20,12])
for i in range(nComponents+1):
    plt.subplot(displayRows,3,i+1)
    plt.imshow(imLabels==i)
    if i == 0:
        plt.title("Background, Component ID : {}".format(i))
    else:
        plt.title("Component ID : {}".format(i))

```

## ColorMap
```
It is a bit difficult to visualize the difference in intensity value in grayscale images, we apply a colormap so that grayscale values are converted to color for the purpose of display.

First, we normalize the image pixel values to 0 to 255. To achieve this we first find the min and max values in the image, and then normalize the image by subtracting the min and dividing by (max - min). This normalizes the image to be between 0 and 1. Finally 255 is multiplied to we get an image with values between 0 and 255. Finally, we apply a colormap on the labelled image. 

Lets see how to apply colormaps to the labels obtained from connected component analysis

# The following line finds the min and max pixel values
# and their locations on an image.
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)

# Normalize the image so that the min value is 0 and max value is 255.
imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)

# Convert image to 8-bits unsigned type
imLabels = np.uint8(imLabels)

# Apply a color map
imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
plt.imshow(imColorMap[:,:,::-1])

```


## Contour Analysis
```
Find border of image using white pixels from a binary image, 

Input to find_contour:
Binary Image
Mode
Method

Output:
Contour
Hierarchy

MODE: 
RETR_External - Outer Contour
RETR_List - List of Contours
RETR_CCOMP - Retrieves Contours with 2 level Hierarchy 
RETR_TREE - Full Hierarchy of Nested Contours

Methods: Points to Display the Contour
CHAIN_APPROX_NONE - All the boundary points are returned as part of the contour
CHAIN_APPROX_SIMPLE - Captures only end points of the contour
CHAIN_APPROX_TC89_L1 - Approximation method used for finding contours of un-even shapes
CHAIN_APPROX_TC89_KCOS - Approximation method used for finding contours of un-even shapes

Contours - Array of array of points
Hierarchy - ContourNo, Next, Previous, First Child, Parent

approxPolyDP - approximate poly dynamic programming gives the points in the contour.

contours, hierarchy =   cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
Where,

image - input image (8-bit single-channel). Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary . You can use compare, inRange, threshold , adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one.
contours - Detected contours. Each contour is stored as a vector of points.
hierarchy - Optional output vector containing information about the image topology. It has been described in detail in the video above.
mode - Contour retrieval mode, ( RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE )
method - Contour approximation method. ( CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1 etc )
offset - Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.

```
