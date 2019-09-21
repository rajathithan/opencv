# OpenCV - level - 1
OpenCV repository


## Comprehensive list of openCV functions - level - 1




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
Hierarchy - Next, Previous, First Child, Parent

approxPolyDP - approximate poly dynamic programming gives the points in the contour.

contours, hierarchy =   cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
Where,

image - input image (8-bit single-channel). Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary . You can use compare, inRange, threshold , adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one.
contours - Detected contours. Each contour is stored as a vector of points.
hierarchy - Optional output vector containing information about the image topology. It has been described in detail in the video above.
mode - Contour retrieval mode, ( RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE )
method - Contour approximation method. ( CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1 etc )
offset - Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.

contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found = {}".format(len(contours)))
print("\nHierarchy : \n{}".format(hierarchy))

```

## Draw Contours
```
To draw the contours, cv2.drawContours function is used. It can also be used to draw any shape provided you have its boundary points. Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.

cv2.drawContours(image, contours, -1, (0,255,0), 3);

# Draw only the 3rd contour
# Note that right now we do not know
# the numbering of contour in terms of the shapes
# present in the figure
image = imageCopy.copy()
cv2.drawContours(image, contours[2], -1, (0,0,255), 3);

```


## Contour Moments
```
Contour moments are used to find the center of an arbitary shape

Image Moment is a particular weighted average of image pixel intensities, with the help of which we can find some specific properties of an image, like radius, area, centroid etc. To find the centroid of the image, we generally convert it to binary format and then find its center.

for cnt in contours:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    
    # Mark the center
    cv2.circle(image, (x,y), 10, (255,0,0), -1);

C_X = m10/m00
C_Y = m01/m00

C_x is the x coordinate and C_y is the y coordinate of the centroid and M denotes the Moment.

```

## Area and Perimeter of a contour
```
To find the area and perimeter of a contour

for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))
    
```

## Bounding boxes
```
There are 2 type of bounding boxes we can create around a contour:

A vertical rectangle
A rotated rectangle - This is the bounding box with the minimum area

for cnt in contours:
    # Vertical rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 2)

for cnt in contours:
    # Rotated bounding box
    box = cv2.minAreaRect(cnt)
    boxPts = np.int0(cv2.boxPoints(box))
    # Use drawContours function to draw 
    # rotated bounding box
    cv2.drawContours(image, [boxPts], -1, (0,255,255), 2)

```


## Fitting Circle and Ellipse
```
Fitting a circle and/or an ellipse over the contour

for cnt in contours:
    # Fit a circle
    ((x,y),radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(image, (int(x),int(y)), int(round(radius)), (125,125,125), 2)
    
for cnt in contours:
    # Fit an ellipse
    # We can fit an ellipse only
    # when our contour has minimum
    # 5 points
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(image, ellipse, (255,0,125), 2)

```


## Blob Detection
```
A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ). In the image above, the dark connected regions are blobs, and the goal of blob detection is to identify and mark these regions.

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(im)

im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
 
# Mark blobs using image annotation concepts we have studied so far
for k in keypoints:
    x,y = k.pt
    x=int(round(x))
    y=int(round(y))
    # Mark center in BLACK
    cv2.circle(im,(x,y),5,(0,0,0),-1)
    # Get radius of blob
    diameter = k.size
    radius = int(round(diameter/2))
    # Mark blob in RED
    cv2.circle(im,(x,y),radius,(0,0,255),2)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

```

## Blob Detection - Parameters
```
Thresholding : Convert the source images to several binary images by thresholding the source image with thresholds starting at minThreshold. These thresholds are incremented by thresholdStep until maxThreshold. So the first threshold is minThreshold, the second is minThreshold + thresholdStep, the third is minThreshold + 2 x thresholdStep, and so on.
Grouping : In each binary image, connected white pixels are grouped together. Let’s call these binary blobs.
Merging : The centers of the binary blobs in the binary images are computed, and blobs located closer than minDistBetweenBlobs are merged.
Center & Radius Calculation : The centers and radii of the new merged blobs are computed and returned.

####The parameters for SimpleBlobDetector can be set to filter the type of blobs we want.

By Color : First you need to set filterByColor = 1. Set blobColor = 0 to select darker blobs, and blobColor = 255 for lighter blobs.
By Size : You can filter the blobs based on size by setting the parameters filterByArea = 1, and appropriate values for minArea and maxArea. E.g. setting minArea = 100 will filter out all the blobs that have less then 100 pixels.
By Shape : Now shape has three different parameters.
Circularity : This just measures how close to a circle the blob is. E.g. a regular hexagon has higher circularity than say a square. To filter by circularity, set filterByCircularity = 1. Then set appropriate values for minCircularity and maxCircularity.

Circularity=4π×Area(perimeter)2
 
This means that a circle has a circularity of 1, circularity of a square is 0.785, and so on.

Convexity : A picture is worth a thousand words. Convexity is defined as the (Area of the Blob / Area of it’s convex hull). Now, Convex Hull of a shape is the tightest convex shape that completely encloses the shape. To filter by convexity, set filterByConvexity = 1, followed by setting 0 ≤ minConvexity ≤ 1 and maxConvexity ( ≤ 1)

Inertia Ratio : Don’t let this scare you. Mathematicians often use confusing words to describe something very simple. All you have to know is that this measures how elongated a shape is. E.g. for a circle, this value is 1, for an ellipse it is between 0 and 1, and for a line it is 0. To filter by inertia ratio, set filterByInertia = 1, and set 0 ≤ minInertiaRatio ≤ 1 and maxInertiaRatio (≤ 1 ) appropriately.

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
 
# Filter by Area.
params.filterByArea = True
params.minArea = 1500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

```


## ColorSpaces
```
A color space is a representation of color using different attributes. The attributes can be the color tint (Hue), saturation and brightness or Red, Green and Blue or something else. The different attributes are represented along the axes of a 3-dimensional space, so that it is easier to describe them mathematically and find relations among the different color spaces

It should be noted that the choice of color space depends largely on the problem you are trying to solve. Given a problem, you should always try and experiment with different color spaces for arriving at the desired solution

plt.imshow(bgr[:,:,::-1])

```


## The RGB color space
```
The RGB color space is an additive color space in which Red, Green and Blue light rays are added in various proportions to produce different colors. It is the most commonly used color space in image processing.

plt.figure(figsize=(20,15))
plt.subplot(131)
plt.imshow(bgr[:,:,0],cmap='gray')
plt.title("Blue Channel");
plt.subplot(132)
plt.imshow(bgr[:,:,1],cmap='gray')
plt.title("Green Channel");
plt.subplot(133)
plt.imshow(bgr[:,:,2],cmap='gray')
plt.title("Red Channel");

In the RGB color space, all three channels contain information about the color as well as brightness. It is better for some applications if we can separate the color component, also known as Chrominance , from the lightness or brightness component also known as Luminance
```


## HSV ColorSpace
```
This is one of the most popular color spaces used in image processing after the RGB color space. Its three components are :

Hue - indicates the color / tint of the pixel

Saturation - indicates the purity (or richness) of the color

Value - indicates the amount of brightness of the pixel

The HSV color space converts the RGB color space from cartesian coordinates (x, y, z) to cylindrical coordinates (ρ, φ, z). It is more intuitive than the RGB color space because it separates the color and brightness into different axes. This makes it easier for us to describe any color directly.

hsvImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

plt.figure(figsize=(20,15))
plt.subplot(131)
plt.imshow(hsvImage[:,:,0],cmap='gray')
plt.title("Hue");
plt.subplot(132)
plt.imshow(hsvImage[:,:,1],cmap='gray')
plt.title("Saturation");
plt.subplot(133)
plt.imshow(hsvImage[:,:,2],cmap='gray')
plt.title("Value");

Hue 
The Hue channel refers to the color and its values, ranging from 0 to 180 in OpenCV. Since the HSV color space is represented in a cylindrical coordinate system, the values for Hue wrap around 180. For example, the Hue for red color is near 180. So, some tints of red can wrap around 180 and have values around 0. This is evident from the middle (red) pepper in the figure above, which shows both very high (180) and low (0) Hue values for the red pepper.

Saturation 
Saturation refers to how pure the color is. Pure red has high saturation. Different shades of a color correspond to different saturation levels. Saturation of 0 corresponds to white color which indicates that the color shade is at the lowest or the color is simply absent.

So, in the figure above, we can see that the green and red peppers are highly saturated, i.e. these colors are in their purest form. On the other hand, the yellow pepper has relatively lower saturation. With the Hue and Saturation channels known, we have a better idea about the color and tints or shades of color in our image.

Value 
Value refers to lightness or brightness. It indicates how dark or bright the color is. It also signifies the amount of light that might have fallen on the object. It is pretty clear from the original image and the Value channel that the red and yellow peppers are much brighter as compared to the green pepper.

Hue controls the perception of color. It is represented as an angle where a hue of 0 is red, green is 120 degrees ( 60 in OpenCV ), and blue is at 240 degrees( 120 in OpenCV ). In OpenCV, Hue is goes from 0 to 180 intensities values where one grayscale intensity change represents 2 degrees.

Because hue is angle, you get red color for both H = 0 and H = 360 ( or 180 in OpenCV's representation )

When value is 0 it is black , when saturation is the image looks grayscale.

```


## YCrCb ColorSpace
```
The YCrCb color space is derived from the RGB color space. Its three components are :
Y (Luma), derived from the RGB values of the image
Cr = R - Y (how far is the red component from the Luma, also known as Red Difference)
Cb = B - Y (how far is the blue component from the Luma, also known as Blue Difference)

Let us convert the image from BGR to YCrCb and take a look at the three channels.

ycbImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

Observations 
Y Channel looks similar to a grayscale image.
Cr Channel indicates the amount of Red in the image, which is evident from the high values of the middle (red) pepper. Similarly,
Cb indicates the amount of Blue in the image, which is why the blue background displays high values.

plt.figure(figsize=(20,15))
plt.subplot(1,3,1)
plt.title("Y Channel")
plt.imshow(ycbImage[:,:,0],cmap="gray")
plt.subplot(1,3,2)
plt.title("Cr Channel")
plt.imshow(ycbImage[:,:,1],cmap="gray")
plt.subplot(1,3,3)
plt.title("Cb Channel")
plt.imshow(ycbImage[:,:,2],cmap="gray")
plt.show()

```

## The Lab color space
```
The Lab color space consists of :
Lightness
A (a color component ranging from Green to Magenta)
B (a color component ranging from Blue to Yellow).
The L channel encodes brightness, while the A and B channels encode color.
The following code shows how to convert from BGR to Lab color space in OpenCV.

labImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

plt.figure(figsize=(20,15))
plt.subplot(1,3,1)
plt.title("L Channel")
plt.imshow(labImage[:,:,0],cmap="gray")
plt.subplot(1,3,2)
plt.title("A Channel")
plt.imshow(labImage[:,:,1],cmap="gray")
plt.subplot(1,3,3)
plt.title("B Channel")
plt.imshow(labImage[:,:,2],cmap="gray")
plt.show()
```

## Histograms
```
A histogram is a very important tool in Image processing. It is a graphical representation of the distribution of data. An image histogram gives a graphical representation of the tonal distribution in a digital image.

the x-axis represents the different intensity values or range of intensity values ( also called bins ), which lie between 0 and 255, and
the y-axis represents the number of times a particular intensity value occurs in the image.

plt.hist() available in the matplotlib library for drawing the histogram.

Function Syntax 
hist, bins, patches =   plt.hist( x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None )
Parameters

There are many parameters in the function. Let us go over the most important and frequently used ones.

Input

x - source image as an array
bins - number of bins
color - color for plotting the histogram
Output

hist - histogram array
bins - edges of bins

================
img = cv2.imread(filename)
plt.figure(figsize=[20,10])
plt.imshow(img[...,::-1])
plt.axis('off')
hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsvImage)
print(H)
print(H.shape)
# Remove unsaturated (white/gray) pixels 
H_array = H[S > 10].flatten()

print(H_array)
print(H_array.shape)
plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Image");plt.axis('off')
plt.subplot(122);plt.hist(H_array, bins=180, color='r');plt.title("Histogram")

This is helpful in doing color based segmentation

```


## Desaturation - Image Enhancement Technique
```
Photo editing apps like photoshop or instagram use many different kinds of image enhancement techniques to make the images look special. One such image enhancement technique is desaturation. We desaturate the image by decreasing the values in the Saturation channel. This result in an image which looks faded or washed out, with no colors. This effect is used in many instagram filters

we convert the image to HSV color space using the cvtColor function, convert the hsvImage to float32 and split the image into its channels H, S, V.

# Specify scaling factor
saturationScale = 0.01

# Convert to HSV color space
hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Convert to float32
hsvImage = np.float32(hsvImage)

# Split the channels
H, S, V = cv2.split(hsvImage)

Next, we scale the S channel with a scale factor and merge the channels back to get the final output.

We need to convert to the uint8 datatype since we had done the multiplication in float32.

# Multiply S channel by scaling factor and clip the values to stay in 0 to 255 
S = np.clip(S * saturationScale , 0, 255)

# Merge the channels and show the output
hsvImage = np.uint8( cv2.merge([H, S, V]) )
imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title('Original Image')
plt.subplot(122);plt.imshow(imSat[...,::-1]);plt.title('Desaturated Image');
```


## Historgram Equalization - Contrast Enhancement
```
Histogram Equalization is a non-linear method for enhancing contrast in an image. We have already seen the theory in the video. Now, let's see how to perform histogram equalization using OpenCV using equalizeHist().

equalizeHist() performs histogram equalization on a grayscale image. The syntax is given below.

Function Syntax
dst =   cv2.equalizeHist(   src[, dst]  )
Parameters

src - Source 8-bit single channel image.
dst - Destination image of the same size and type as src .

==================

import matplotlib
matplotlib.rcParams['figure.figsize'] = (50.0, 50.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['axes.titlesize'] = 40
matplotlib.rcParams['image.interpolation'] = 'bilinear'

im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Equalize histogram
imEq = cv2.equalizeHist(im)

#Display images
plt.figure()

ax = plt.subplot(1,2,1)
plt.imshow(im, vmin=0, vmax=255)
ax.set_title("Original Image")
ax.axis('off')


ax = plt.subplot(1,2,2)
plt.imshow(imEq, vmin=0, vmax=255)
ax.set_title("Histogram Equalized")
ax.axis('off')

Display Historgram
==================

we had used calcHist() to calculate histogram. Matplotlib provides an alternative way which has a slightly better syntax

plt.figure(figsize=(30,10))
plt.subplot(1,2,1)
plt.hist(im.ravel(),256,[0,256]); 

plt.subplot(1,2,2)
plt.hist(imEq.ravel(),256,[0,256]); 
plt.show()
```


## Histogram Equalization on Color Images
```
The right way to perform histogram equalization on color images is to transform the images to a space like the HSV colorspace where colors/hue/tint is separated from the intensity.

These are the steps involved

Tranform the image to HSV colorspace.
Perform histogram equalization only on the V channel.
Transform the image back to RGB colorspace.

im = cv2.imread(filename)

# Convert to HSV 
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Perform histogram equalization only on the V channel
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

# Convert back to BGR format
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)

#Display images
plt.figure()

ax = plt.subplot(1,2,1)
plt.imshow(im[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Original Image")
ax.axis('off')


ax = plt.subplot(1,2,2)
plt.imshow(imEq[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Histogram Equalized")
ax.axis('off')
```


## Contrast Limited Adaptive Histogram Equalization (CLAHE) 
```
Histogram equalization uses the pixels of the entire image to improve contrast. While this may look good in many cases, sometimes we may want to enhance the contrast locally so the image does not looks more natural and less dramatic.

========================

im = cv2.imread(filename)

# Convert to HSV 
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
imhsvCLAHE = imhsv.copy()

# Perform histogram equalization only on the V channel
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imhsvCLAHE[:,:,2] = clahe.apply(imhsvCLAHE[:,:,2])

# Convert back to BGR format
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)
imEqCLAHE = cv2.cvtColor(imhsvCLAHE, cv2.COLOR_HSV2BGR)


#Display images
plt.figure(figsize=(40,40))

ax = plt.subplot(1,3,1)
plt.imshow(im[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Original Image")
ax.axis('off')

ax = plt.subplot(1,3,2)
plt.imshow(imEq[:,:,::-1], vmin=0, vmax=255)
ax.set_title("Histogram Equalized")
ax.axis('off')

ax = plt.subplot(1,3,3)
plt.imshow(imEqCLAHE[:,:,::-1], vmin=0, vmax=255)
ax.set_title("CLAHE")
ax.axis('off')

CLAHE performs stretching of values locally . The final image will be more close to the original image with required contrast enhancement

```


## Look Up table
```
Look-up tables (LUTs) are very common in custom filters in which two pixels with the same value in the input involves the same value in the output too. An LUT transformation assigns a new pixel value to each pixel in the input image according to the values given by a table. In this table, the index represents the input intensity value and the content of the cell given by the index represents the corresponding output value. As the transformation is actually computed for each possible intensity value, this results in a reduction in the time needed to apply the transformation over an image (images typically have more pixels than the number of intensity values).

The LUT(InputArray src, InputArray lut, OutputArray dst, int interpolation = 0) OpenCV function applies a look-up table transformation over an 8-bit signed or an src unsigned image. Thus, the table given in the lut parameter contains 256 elements. 

```


## Color tone Adjustment using Curves
```
Curves are nothing increasing / Decreasing certain pixel points in the R channel and B channel , to get photoshop effects, We increase / decrease certain pixel points and use interpolation to fill up the respective values from 0 to 256.

( Most ridiculous explanation - I can understand this, i hope you can too)
f(x1) = input function Has 10 values, f(y1) = output values based on f(x1), now f(x1') = Uses the same input function has 100 values, now to identify f(y1') use np.interp as you dont know the output function to derive f(y1')
==================================
Determine the pixel points for rchannel and bchannel

# pivot points for X-Coordinates
originalValue = np.array([0, 50, 100, 150, 200, 255])

# Changed points on Y-axis for each channel
rCurve = np.array([0, 80, 150, 190, 220, 255])
bCurve = np.array([0, 20,  40,  75, 150, 255])

# Create a LookUp Table
fullRange = np.arange(0,256)
rLUT = np.interp(fullRange, originalValue, rCurve )
bLUT = np.interp(fullRange, originalValue, bCurve )

Use the lookup table and modify the rchannel and the bchannel in the original image. 

# Get the blue channel and apply the mapping
bChannel = img[:,:,0]
bChannel = cv2.LUT(bChannel, bLUT)
img[:,:,0] = bChannel

# Get the red channel and apply the mapping
rChannel = img[:,:,2]
rChannel = cv2.LUT(rChannel, rLUT)
img[:,:,2] = rChannel

# show and save the ouput
combined = np.hstack([original,img])

plt.imshow(combined[:,:,::-1])
plt.title("Warming filter output")
plt.show()

```


## Signal Processing Jargons
```
Image Patch: An image patch is simply a small (3x3, 5x5 … ) region of the image centered around a pixel.

Low Frequency Information : An image patch is said to have low frequency information if it is smooth and does not have a lot of texture.

High Frequency Information : An image patch is said to have high frequency information if it has a lot of texture (edges, corners etc.).

Low Pass Filtering : This is essentially image blurring / smoothing. It you blur an image, you smooth out the texture. As the name suggests, low pass filtering lets lower frequency information pass and blocks higher frequency information.

High Pass Filtering : This is essentially a sharpening and edge enhancement type of operation. As the name suggests, low frequency information is suppressed and high frequency information is preserved in high pass filtering.
```


## Convolution
```
Convolution is the basis of all linear filters, to sharp , blur or to detect the edges of an image, we can use convolution. 

Boundary Conditions:

Ignore the boundary pixels : If we discount the boundary pixels, the output image will be slightly smaller than the input image.

Zero padding : We can pad the input image with zeros at the boundary pixels to make it larger and then perform convolution.

Replicate border : The other option is to replicate the boundary pixels of the input image and then perform the convolution operation on this larger image.

Reflect border : The preferred option is to reflect the border about the boundary. Reflecting ensures a smooth intensity transition of pixels at the boundary.

By default OpenCV uses border type BORDER_REFLECT_101 which is the same as option 4

filter2D. The basic usage is given below.

Function Syntax
dst =   cv.filter2D(    src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]
Parameters

src input image.
dst output image of the same size and the same number of channels as src.
ddepth desired depth of the destination image.
kernel convolution kernel (or rather a correlation kernel), a single-channel floating point matrix; if you want to apply different kernels to different channels, split the image into separate color planes using split and process them individually.
anchor anchor of the kernel that indicates the relative position of a filtered point within the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor is at the kernel center.
delta optional value added to the filtered pixels before storing them in dst.
borderType pixel extrapolation method.

filter2D is used to perform the convolution. Notice, you do not have to explicitly allocate space for the filtered image. It is done automatically for you inside filter2D.

The second parameter (depth) is set to -1, which means the bit-depth of the output image is the same as the input image. So if the input image is of type uint8, the output image will also be of the same type.

kernel_size = 5
# Create a 5*5 kernel with all elements equal to 1
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2

# Print Kernel
print (kernel)

result = cv2.filter2D(image, -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(result[...,::-1]);plt.title("Convolution Result")

```


## Box Blur
```
Box blur - replaces the center pixel by the average of all pixels in the neighborhood

Box Blur in OpenCV
The simplest usage of the blur function is given below

Function Syntax
dst =   cv2.blur(   src, ksize[, dst[, anchor[, borderType]]]   )
Parameters

src input image; it can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst output image of the same size and type as src.
ksize blurring kernel size.
anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
borderType border mode used to extrapolate pixels outside of the image.
======

img = cv2.imread(filename)

dst1=cv2.blur(img,(3,3),(-1,-1))

```


## Gausian Blur
```
A Gaussian Blur kernel,  weights the contribution of a neighboring pixel based on the distance of the pixel from the center pixel

Unlike the box kernel, the Gaussian kernel is not uniform. The middle pixel gets the maximum weight while the pixels farther away are given less weight.

An image blurred using the Gaussian kernel looks less blurry compared to a box kernel of the same size. Small amount of Gaussian blurring is frequently used to remove noise from an image. It is also applied to the image prior to a noise sensitive image filtering operations. For example, the Sobel kernel used for calculating the derivative of an image is a combination of a Gaussian kernel and a finite difference kernel.


Function Syntax
dst =   cv2.GaussianBlur(   src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]   )
Parameters

src input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst output image of the same size and type as src.
ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
sigmaX Gaussian kernel standard deviation in X direction.
sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively; to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
borderType pixel extrapolation method.

Note:
In most cases, sigmaX and sigmaY are the same. But it is fun to experiment with different values of sigmaX and sigmaY and see the results.

It is much easier to control the Gaussian blurring using just one parameter. Usually, we simply provide the kernel size, and let OpenCV automatically calculate the optimum sigma for based on the following equation.

𝜎=0.3⋅((size−1)⋅0.5−1)+0.8
 
If you set sigmaY equal to zero, it is set to sigmaX internally by OpenCV. If both sigmaX and sigmaY are zero, the above formula is used to calculate  𝜎 .

==================

In the following code, Gaussian blur is applied using two different kernels.

The first is a 5x5 kernel with sigmaX and sigmaY set to 0. OpenCV automatically calculates sigma when it is set to 0.

The second is a 25x25 kernel with sigmaX and sigmaY set to 50.

# Apply gaussian blur
dst1=cv2.GaussianBlur(img,(5,5),0,0)
dst2=cv2.GaussianBlur(img,(25,25),50,50)
```


## Median Blur
```
Replace the center pixel with the median value of the neighborhood. This is mostly used for images with salt and pepper grain. 

Function Syntax
dst =   cv2.medianBlur( src, ksize[, dst]   )
Parameters

src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
dst destination array of the same size and type as src.
ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5,7, ...

=========================

img = cv2.imread("saltandpepper.png")
# Defining the kernel size
kernelSize = 7

# Performing Median Blurring and store it in numpy array "medianBlurred"
medianBlurred = cv2.medianBlur(img,kernelSize)

# Display the original and median blurred image
plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(medianBlurred[...,::-1]);plt.title("Median Blur Result : KernelSize = 7")

```


## Bilateral Filtering
```
Bilateral filtering is an edge preserving filter, where the center pixel is modified not by the intensity or spatial closeness of the whole neighborhood, rather the decision is taken based on the pixels in center and right columns. The spatial difference and the intensity difference are determined to change the value of the center pixel

If the neighborhood pixels are edges, the difference in intensity  (𝐼𝑝−𝐼𝑞)  will be higher. Since the Gaussian is a decreasing function,  𝐺𝜎𝑟(𝐼𝑝−𝐼𝑞)  will have lower weights for higher values. Hence, the smoothing effect will be lower for such pixels, preserving the edges.

intensity & spaitial differences are controlled using the parameter  𝜎𝑟 & 𝜎s.

 Bilateral filter in OpenCV.

Function Syntax
dst =   cv2.bilateralFilter(    src, d, sigmaColor, sigmaSpace[, dst[, borderType]] )
Parameters

src Source 8-bit or floating-point, 1-channel or 3-channel image.
dst Destination image of the same size and type as src .
d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
sigmaColor Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
borderType border mode used to extrapolate pixels outside of the image,

============

# diameter of the pixel neighbourhood used during filtering
dia=15;

# Larger the value the distant colours will be mixed together 
# to produce areas of semi equal colors
sigmaColor=80

# Larger the value more the influence of the farther placed pixels 
# as long as their colors are close enough
sigmaSpace=80

#Apply bilateralFilter
result = cv2.bilateralFilter(img, dia, sigmaColor, sigmaSpace)

plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(result[...,::-1]);plt.title("Bilateral Blur Result")
```


## Prewitt filter
```
To make gradient calculations even more robust and noisefree, the image can be Gaussian-blurred slightly before applying a gradient filter. As you know, blurring is also a convolution operation. So applying a Gaussian blur filter before applying the gradient filter would require two convolution operations.

calculate the gradient over a 3x3 patch instead of over a line. The filters below provide a slightly less noisy version of the gradients in the x- and y-directions.

img = cv2.imread('messi5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

```


## Sobel Filter
```
Sobel filter combines the two convolution operations into one.

Sobel function for calculating the X and Y Gradients. Below, you can see the most common usage.

Function Syntax
dst =   cv2.Sobel(  src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]] )
Parameters

src input image.
dst output image of the same size and the same number of channels as src .
ddepth output image depth,in the case of 8-bit input images it will result in truncated derivatives.
dx order of the derivative x.
dy order of the derivative y.
ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
scale optional scale factor for the computed derivative values; by default, no scaling is applied.
delta optional delta value that is added to the results prior to storing them in dst.
borderType pixel extrapolation method.

The X and Y Gradients are calculated using the Sobel function. Note that the depth of the output images is set to CV_32F because gradients can take negative values.

# Apply sobel filter along x direction
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
# Apply sobel filter along y direction
sobely = cv2.Sobel(image,cv2.CV_32F,0,1)

# Normalize image for display
cv2.normalize(sobelx, 
                dst = sobelx, 
                alpha = 0, 
                beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)
cv2.normalize(sobely, 
                dst = sobely, 
                alpha = 0, 
                beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)
```


## Laplacian Filter 
```
Laplacian filter is a second order derivative filter, it is very senstive noise, so it needs to be blurred (smoothed) out before usage.

always note that summing and averaging operations are less affected by noise, and differencing operations are greatly affected by noise

===========

kernelSize = 3

# Applying laplacian
img1 = cv2.GaussianBlur(img,(3,3),0,0)
laplacian = cv2.Laplacian(img1, cv2.CV_32F, ksize = kernelSize, 
                            scale = 1, delta = 0)
                            
# Normalize results
cv2.normalize(laplacian, 
                dst = laplacian, 
                alpha = 0, 
                beta = 1, 
                norm_type = cv2.NORM_MINMAX, 
                dtype = cv2.CV_32F)

```


## Sharpening Filter
```

Step 1: Blur the image to smooth out texture. The blurred image contains low frequency information of the original image. Let  𝐼 be the original image and  𝐼𝑏  be the blurred image.

Step 2: Obtain the high frequency information of the original image by subtracting the blurred image from the original image.

Step 3: Now, put back the high frequency information back in the image and control the amount using a parameter. The final sharpened image is therefore,

𝐼𝑠=𝐼+𝛼(𝐼âˆ′𝐼𝑏)

# Sharpen kernel
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
Next, filter2D is used to perform the convolution.

The third parameter (depth) is set to -1 which means the bit-depth of the output image is the same as the input image. So if the input image is of type CV_8UC3

# Using 2D filter by applying the sharpening kernel
sharpenOutput = cv2.filter2D(image, -1, sharpen)

plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(sharpenOutput[...,::-1]);plt.title("Sharpening Result")

```


## Canny Edge filter
```
Canny edge detection is the most widely-used edge detector. For many of the applications that require edge detection, Canny edge detection is sufficient.

Canny edge detection has the following three steps:

Gradient calculations: Edges are pixels where intensity changes abruptly. From previous modules, we know that the magnitude of gradient is very high at edge pixels. Therefore, gradient calculation is the first step in Canny edge detection.

Non-maxima suppression: In the real world, the edges in an image are not sharp. The magnitude of gradient is very high not only at the real edge location, but also in a small neighborhood around it. Ideally, we want an edge to be represented by a single, pixel-thin contour. Simply thresholding the gradient leads to a fat contour that is several pixels thick. Fortunately, this problem can be eliminated by selecting the pixel with maximum gradient magnitude in a small neighborhood (say 3x3 neighborhood) of every pixel in the gradient image. The name non-maxima suppression comes from the fact that we eliminate (i.e. set to zero) all gradients except the maximum one in small 3x3 neighborhoods over the entire image.

Hysteresis thresholding: After non-maxima suppression, we could threshold the gradient image to obtain a new binary image which is black in all places except for pixels where the gradient is very high. This kind of thresholding would naively exclude a lot of edges because, in real world images, edges tend to fade in and out along their length. For example, an edge may be strong in the middle but fade out at the two ends. To fix this problem, Canny edge detection uses two thresholds. First, a higher threshold is used to select pixels with very high gradients. We say these pixels have a strong edge. Second, a lower threshold is used to obtain new pixels that are potential edge pixels. We say these pixels have a weak edge. A weak edge pixel can be re-classified as a strong edge if one of its neighbor is a strong edge. The weak edges that are not reclassified as strong are dropped from the final edge map.

Canny edge detector in OpenCV is shown below.

Function Syntax
edges   =   cv.Canny(   dx, dy, threshold1, threshold2[, edges[, L2gradient]]   )
Parameters

dx 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
dy 16-bit y derivative of input image (same type as dx).
edges output edge map; single channels 8-bit image, which has the same size as image .
threshold1 first threshold for the hysteresis procedure.
threshold2 second threshold for the hysteresis procedure.
L2gradient a flag, indicating whether a more accurate L2 norm =âˆš(dI/dx)2+(dI/dy)2 should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
If you want better accuracy at the expense of speed, you can set the L2gradient flag to true.

=============

lowThreshold = 50
highThreshold = 100

maxThreshold = 1000

apertureSizes = [3, 5, 7]
maxapertureIndex = 2
apertureIndex = 0

blurAmount = 0
maxBlurAmount = 20

The function applyCanny is called whenever any trackbar value is changed. The image is first blurred. The amount of blur depends on blurAmount. A Sobel apertureSize (3, 5 or 7) is chosen based on the trackbar value. Finally, the Canny function is called and results are displayed.

def applyCanny():
    # Blur the image before edge detection
    if(blurAmount > 0):
        blurredSrc = cv2.GaussianBlur(src, 
                        (2 * blurAmount + 1, 2 * blurAmount + 1), 0);
    else:
        blurredSrc = src.copy()

    # Canny requires aperture size to be odd
    apertureSize = apertureSizes[apertureIndex];

    # Apply canny to detect the images
    edges = cv2.Canny( blurredSrc, 
                        lowThreshold, 
                        highThreshold, 
                        apertureSize = apertureSize )
    plt.imshow(edges[...,::-1])
 
lowThreshold : Keeping all other parameters constant, when you lower the lowThreshold, broken edges tend to get connected. If you increase it, continuous edges may break.

highThreshold : Keeping all other parameters constant, when you increase highThreshold, fewer edges are detected. On the other hand, decreasing highThreshold leads to more edges.

apertureSize : Increasing the aperture size leads to many more edges. This is simply because larger Sobel kernels return larger gradient values. Low and high thresholds should be changed when aperture size is changed.

blurAmount : When the blur amount is increased, noise in the image is reduced, and spurious edges are removed. As a result, fewer edges are detected.
```


## Hough Transform
```
Hough transform is a feature extraction method for detecting simple shapes such as circles, lines etc. in an image.The main advantage of using the Hough transform is that it is insensitive to occlusion. (Occlusion often occurs when two or more objects come too close and seemingly merge or combine with each other. Image processing system with object tracking often wrongly track the occluded objects)

olar form of a line is represented as:

ρ=xcos(θ)+ysin(θ)(1)
ρ  represents the perpendicular distance of the line from the origin in pixels
θ  is the angle measured in radians, which the line makes with the origin as shown in the figure above

When we say that a line in 2D space is parameterized by  ρ  and  θ , it means that if we any pick a  (ρ,θ) , it corresponds to a line.

Imagine a 2D array where the x-axis has all possible  θ  values and the y-axis has all possible  ρ  values. Any bin in this 2D array corresponds to one line.

This 2D array is called an accumulator because we will use the bins of this array to collect evidence about which lines exist in the image. The top left cell corresponds to a  (−R,0)  and the bottom right corresponds to  (R,π) .The value inside the bin  (ρ,θ)  will increase as more evidence is gathered about the presence of a line with parameters  ρ  and  θ .

Step 1 : Initialize Accumulator
==============================
First, we need to create an accumulator array. The number of cells you choose to have is a design decision. Let’s say you chose a 10×10 accumulator. It means that  ρ  can take only 10 distinct values and the  θ  can take 10 distinct values, and therefore you will be able to detect 100 different kinds of lines. The size of the accumulator will also depend on the resolution of the image. But if you are just starting, don’t worry about getting it perfectly right. Pick a number like 20×20 and see what results you get.

Step 2: Detect Edges
====================
Now that we have set up the accumulator, we want to collect evidence for every cell of the accumulator because every cell of the accumulator corresponds to one line.

How do we collect evidence?

The idea is that if there is a visible line in the image, an edge detector should fire at the boundaries of the line. These edge pixels provide evidence for the presence of a line.

The output of edge detection is an array of edge pixels  [(x1,y1),(x2,y2)...(xn,yn)] 

Step 3: Voting by Edge Pixels
=============================
For every edge pixel  (x,y)  in the above array, we vary the values of  θ  from  0  to  π  and plug it in equation 1 to obtain a value for  ρ .

we vary the  θ  pixels and obtain the values for  ρ  using equation 1.

These curves intersect at a point indicating that a line with parameters  θ=1  and  ρ=9.5  is passing through them.

Typically, we have hundreds of edge pixels and the accumulator is used to find the intersection of all the curves generated by the edge pixels.

Let’s see how this is done.

Let’s say our accumulator is 20×20 in size. So, there are 20 distinct values of  θ  and so for every edge pixel  (x,y) , we can calculate 20  (ρ,θ)  pairs by using equation 1. The bin of the accumulator corresponding to these 20 values of  (ρ,θ)  is incremented.

We do this for every edge pixel and now we have an accumulator that has all the evidence about all possible lines in the image.

We can simply select the bins in the accumulator above a certain threshold to find the lines in the image. If the threshold is higher, you will find fewer strong lines, and if it is lower, you will find a large number of lines including some weak ones.

```

## Hough Lines
```
line detection using Hough Transform is implemented in the function HoughLines and HoughLinesP [Probabilistic Hough Transform].

Function Syntax
lines = cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
lines = cv.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]])
Parameters

image - 8-bit, single-channel binary source image. The image may be modified by the function.
lines - Output vector of lines. Each line is represented by a 4-element vector  (x1,y1,x2,y2)  , where  (x1,y1)  and  (x2,y2)  are the ending points of each detected line segment.
rho - Distance resolution of the accumulator in pixels.
theta - Angle resolution of the accumulator in radians.
threshold - Accumulator threshold parameter. Only those lines are returned that get enough votes  (>threshold) .
srn - For the multi-scale Hough transform, it is a divisor for the distance resolution rho . The coarse accumulator distance resolution is rho and the accurate accumulator resolution is rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these parameters should be positive.
stn - For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
min_theta - For standard and multi-scale Hough transform, minimum angle to check for lines. Must fall between 0 and max_theta.
max_theta - For standard and multi-scale Hough transform, maximum angle to check for lines. Must fall between min_theta and CV_PI.

====================

# Read image 
img = cv2.imread(DATA_PATH + 'images/lanes.jpg', cv2.IMREAD_COLOR)
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
# Detect points that form a line
rho : The resolution of the parameter r in pixels. We use 1 pixel.
theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
plt.imshow(img[:,:,::-1])
```

## 	Hough Circles
```
we require three parameters:

(x,y)  coordinates of the center of the circle.
radius.
As you can imagine, a circle detector will require a 3D accumulator — one for each parameter.

The equation of a circle is given by

(x−x0)^2+(y−y0)^2=r^2
The following steps are followed to detect circles in an image: –

Find the edges in the given image with the help of edge detectors (Canny).
For detecting circles in an image, we set a threshold for the maximum and minimum value of the radius.
Evidence is collected in a 3D accumulator array for the presence of circles with different centers and radii.
The function HoughCircles is used in OpenCV to detect the circles in an image.

Function Syntax
circles = cv.HoughCircles( image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
Parameters

image - 8-bit, single-channel binary source image. The image may be modified by the function.
circles - Output vector of found circles. Each vector is encoded as 3 or 4 element floating-point vector (x,y,radius) or (x,y,radius,votes) .
method - Detection method. Currently, the only implemented method is HOUGH_GRADIENT
dp - Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
minDist - Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
param1 - First method-specific parameter. In case of HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
param2 - Second method-specific parameter. In case of HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
minRadius - Minimum circle radius.
maxRadius - Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, returns centers without finding the radius.

===============

# Read image as gray-scale
img = cv2.imread(DATA_PATH + 'images/circles.jpg', cv2.IMREAD_COLOR)
# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Apply hough transform on the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 50, param1=450, param2=10, minRadius=30, maxRadius=40)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
plt.imshow(img[:,:,::-1])

==============

circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=10, minRadius=50, maxRadius=115)

```


## HDR Imaging
```
High Dynamic Range (HDR) image using multiple images taken with different exposure settings which is better than each one of these images
It actually takes 3 images at three different exposures. The images are taken in quick succession so there is almost no movement between the three shots. The three images are then combined to produce the HDR image

Step 1: Capture multiple images with different exposures
========================================================
We load the 4 images and create 2 lists:

times : contains the exposure times for the images
images : contains the images

def readImagesAndTimes():
  # List of exposure times
  times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
   
  # List of image filenames
  filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
  images = []
  for filename in filenames:
    im = cv2.imread(DATA_PATH + "images/" + filename)
    images.append(im)
   
  return images, times

images, times = readImagesAndTimes()

Step 2: Align Images
====================

OpenCV provides an easy way to align these images using AlignMTB. This algorithm converts all the images to median threshold bitmaps (MTB). An MTB for an image is calculated by assigning the value 1 to pixels brighter than median luminance and 0 otherwise. An MTB is invariant to the exposure time. Therefore, the MTBs can be aligned without requiring us to specify the exposure time.

MTB based alignment is performed using the following lines of code in OpenCV.Specifically, we create an object of AlignMTB class and apply the alignment function (process) on the set of images we loaded earlier.

cv.AlignMTB.process(    src, dst, times, response   )
Parameters

src - vector of input images ( Input )
dst - vector of aligned images ( Output )

# Align input images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

Step 3: Recover the Camera Response Function 
============================================

CRF is done using just two lines of code in OpenCV using CalibrateDebevec or CalibrateRobertson. In this module we will use CalibrateDebevec.

Consider just ONE pixel at some location (x,y) of the images. If the CRF was linear, the pixel value would be directly proportional to the exposure time unless the pixel is too dark ( i.e. nearly 0 ) or too bright ( i.e. nearly 255) in a particular image. We can filter out these bad pixels ( too dark or too bright ), and estimate the brightness at a pixel by dividing the pixel value by the exposure time and then averaging this brightness value across all images where the pixel is not bad ( too dark or too bright ). We can do this for all pixels and obtain a single image where all pixels are obtained by averaging “good” pixels.

Function Syntax 
CRFObject = cv2.createCalibrateDebevec()
dst =   CRFObject.process(  src, times[, dst]   )
Parameters

src - vector of input images
times - vector of exposure time values for each image
dst ( optional ) - 256x1 matrix with inverse camera response function

# Obtain Camera Response Function (CRF)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)


# Plot CRF
x = np.arange(256, dtype=np.uint8)
y = np.squeeze(responseDebevec)

plt.figure(figsize=(15,8))
plt.plot(x, y[:,0],'r'  , x, y[:,1],'g', x, y[:,2],'b');
plt.title("CRF")
plt.xlabel("Measured Intensity")
plt.ylabel("Calibrated Intensity")

Step 4: Merge Images 
====================

Once the CRF has been estimated, we can merge the exposure images into one HDR image using MergeDebevec.

# Merge images into an HDR linear image
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# Save HDR image.
cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

Step 5: Tone mapping
====================

The process of converting a High Dynamic Range (HDR) image to an 8-bit per channel image while preserving as much detail as possible is called Tone mapping.

Some of the common parameters of the different tone mapping algorithms are listed below.

gamma : This parameter compresses the dynamic range by applying a gamma correction. When gamma is equal to 1, no correction is applied. A gamma of less than 1 darkens the image, while a gamma greater than 1 brightens the image.

saturation : This parameter is used to increase or decrease the amount of saturation. When saturation is high, the colors are richer and more intense. Saturation value closer to zero, makes the colors fade away to grayscale.

contrast : Controls the contrast ( i.e. log (maxPixelValue/minPixelValue) ) of the output image. Let us explore the four tone mapping algorithms available in OpenCV.

Drago Tonemap 
The parameters for Drago Tonemap are shown below

retval  =   cv.createTonemapDrago(  [, gamma[, saturation[, bias]]] )
Here, bias is the value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give the best results. The default value is 0.85.

The Python code is shown below. The parameters were obtained by trial and error. The final output is multiplied by 3 just because it gave the most pleasing results to me. You should play around with the parameters and see how the output changes.

# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago
plt.imshow(ldrDrago[:,:,::-1])

Reinhard Tonemap 
The parameters for Reinhard Tonemap are shown below.

retval  =   cv.createTonemapReinhard(   [, gamma[, intensity[, light_adapt[, color_adapt]]]]    )
parameters

intensity should be in the [-8, 8] range. Greater intensity value produces brighter results.

light_adapt controls the light adaptation and is in the [0, 1] range. A value of 1 indicates adaptation based only on pixel value and a value of 0 indicates global adaptation. An in-between value can be used for a weighted combination of the two.

color_adapt controls chromatic adaptation and is in the [0, 1] range. The channels are treated independently if the value is set to 1 and the adaptation level is the same for every channel if the value is set to 0. An in-between value can be used for a weighted combination of the two.

# Tonemap using Reinhard's method to obtain 24-bit color image
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)
plt.imshow(ldrReinhard[:,:,::-1])

Mantiuk Tonemap 
The parameters for Mantinuk Tonemap are shown below.

retval  =   cv.createTonemapMantiuk(    [, gamma[, scale[, saturation]]]    )
The parameter scale is the contrast scale factor. Values from 0.6 to 0.9 produce best results.

# Tonemap using Mantiuk's method to obtain 24-bit color image
tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = np.clip(3 * ldrMantiuk,0,1)
plt.imshow(ldrMantiuk[:,:,::-1])

plt.figure(figsize=[20,10])
plt.subplot(131);plt.imshow(ldrDrago[:,:,::-1]);plt.title("HDR using Drago Tone Mapping");plt.axis('off')
plt.subplot(132);plt.imshow(ldrMantiuk[:,:,::-1]);plt.title("HDR using Mantiuk Tone Mapping");plt.axis('off')
plt.subplot(133);plt.imshow(ldrReinhard[:,:,::-1]);plt.title("HDR using Reinhard Tone Mapping");plt.axis('off')
```


## Seamless Cloning
```
Seamless cloning between images is done by copying the gradient of the images, to blend one image into another image, we should not copy the pixels , instead use the gradient to create a seamless blending of images.

Find the x and y gradients of the source and destination images
Copy the gradients from source images to the destination image
Integration in the gradients domain with Dirichlet boundary conditions

output = cv2.seamlessClone(src, dst, mask, center, flags)
Where,

src - Source image that will be cloned into the destination image. In our example it is the airplane.
dst - Destination image into which the source image will be cloned. In our example it is the sky image.
mask - A rough mask around the object you want to clone. This should be the size of the source image. Set it to an all white image if you are lazy!
center - Location of the center of the source image in the destination image.
flags - The two flags that currently work are NORMAL_CLONE and MIXED_CLONE. I have included an example to show the difference.
output - Output / result image.

===================================================

# Read images
src = cv2.imread(DATA_PATH + "images/airplane.jpg")
dst = cv2.imread(DATA_PATH + "images/sky.jpg")

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (800,100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

plt.imshow(output[:,:,::-1])
plt.show()

In Normal Cloning the texture ( gradient ) of the source image is preserved in the cloned region.
In Mixed Cloning, the texture ( gradient ) of the cloned region is determined by a combination of the source and the destination images.
Mixed Cloning does not produce smooth regions because it picks the dominant texture ( gradient ) between the source and destination images. (more preferred)

```


## Face Blending
```
Simple Alpha Blending with Mask

The lighting in the images is very different
The skin tones are very different
The blend will look ridiculous

===================
alpha = cv2.cvtColor(src_mask.copy(), cv2.COLOR_GRAY2RGB)
alpha = alpha.astype(np.float32) / 255.0
output_blend = src * alpha + dst * (1 - alpha)
output_blend = output_blend.astype(np.uint8)
plt.figure(figsize=(7,7)); plt.imshow(output_blend); plt.axis('off');

Semeless clone face blending
============================

Find Center of the mask 
# Find blob centroid
ret, src_mask_bin = cv2.threshold(src_mask, 128,255, cv2.THRESH_BINARY)
m = cv2.moments(src_mask_bin)
center = (int(m['m01']/m['m00']), int(m['m10']/m['m00']) )

# Clone seamlessly.
output_clone = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

plt.figure(figsize=(15,6)); 
plt.subplot(121); plt.imshow(src); plt.title("Barack Obama");plt.axis('off'); 
plt.subplot(122); plt.imshow(dst);  plt.title("Donald Trump");plt.axis('off'); 
plt.figure(figsize=(15,6)); 
plt.subplot(121); plt.imshow(output_blend); plt.title("Using Normal Blending");plt.axis('off'); 
plt.subplot(122); plt.imshow(output_clone);  plt.title("Using Seamless Cloning");plt.axis('off');
```
