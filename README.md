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

1.Increase Aperture Size

>> More Edges

2.Increase Blur

>> Less Noise
>> Fewer Edges


3.Increase High Threshold

>> Fewer Edges


4.Decrease Lower Threshold

>> Connects broken edges
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


## In-Painting
```
In-painting is an image processing technique used for smoothing an image if there any discontinuties in between,
if there is crack in an image, we can draw over that crack to create a mask and then use the sketcher opencv function to clear the crack by identifying the image pixels , near to it and then establishing the boundary using the original image. 

======================================================
# OpenCV Utility Class for Mouse Handling
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])
        cv.imshow(self.windowname + ": mask", self.dests[1])

    # onMouse function for Mouse Handling
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()

====================================================================================

# Read image in color mode
filename = DATA_PATH + "images/Lincoln.jpg"
img = cv.imread(filename, cv.IMREAD_COLOR)

# If image is not read properly, return error
if img is None:
    print('Failed to load image file: {}'.format(filename))
    
# Create a copy of original image
img_mask = img.copy()
# Create a black copy of original image
# Acts as a mask
inpaintMask = np.zeros(img.shape[:2], np.uint8)
# Create sketch using OpenCV Utility Class: Sketcher
sketch = Sketcher('image', [img_mask, inpaintMask], lambda : ((0, 255, 0), 255))

while True:
    ch = cv.waitKey()
    if ch == 27:
        break
    if ch == ord('t'):
        # Use Algorithm proposed by Alexendra Telea: Fast Marching Method (2004)
        # Reference: https://pdfs.semanticscholar.org/622d/5f432e515da69f8f220fb92b17c8426d0427.pdf
        res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
        cv.imshow('Inpaint Output using FMM', res)
    if ch == ord('n'):
        # Use Algorithm proposed by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro: Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting (2001)
        res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
        cv.imshow('Inpaint Output using NS Technique', res)
    if ch == ord('r'):
        img_mask[:] = img
        inpaintMask[:] = 0
        sketch.show()
        
cv.destroyAllWindows()
```


## Affine Transform
```
To apply an affine transform to the entire image you can use the function warpAffine.

dst =   cv2.warpAffine( src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]  )
Parameters

src input image.
dst output image that has the size dsize and the same type as src .
M 2×3 transformation matrix.
dsize size of the output image.
flags combination of interpolation methods (see InterpolationFlags) and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( dst→src ).
borderMode pixel extrapolation method (see BorderTypes); when borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.
borderValue value used in case of a constant border; by default, it is 0.

=====================================

# Scale along x direction
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   1.0, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])

======================================

# Scale along x direction
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   1.0, 0]
    ])

result = cv2.warpAffine(im, warpMat, (2 * outDim[0], outDim[1]))

# Display image
plt.imshow(result[...,::-1])

=====================================

# Scale along x and y directions
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   2.0, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, (2 * outDim[0], 2 * outDim[1]))

# Display image
plt.imshow(result[...,::-1])

=====================================

# Rotate image 
angleInDegrees = 30
angleInRadians = 30 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

# Rotation matrix 
# https://en.wikipedia.org/wiki/Rotation_matrix
    
warpMat = np.float32(
    [
        [ cosTheta, sinTheta, 0],
        [ -sinTheta, cosTheta, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])

=====================================

# Rotate image at a certain point
angleInDegrees = 30
angleInRadians = 30 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

centerX = im.shape[0] / 2
centerY = im.shape[1] / 2

tx = (1-cosTheta) * centerX - sinTheta * centerY
ty =  sinTheta * centerX  + (1-cosTheta) * centerY

# Rotation matrix 
# https://en.wikipedia.org/wiki/Rotation_matrix
    
warpMat = np.float32(
    [
        [ cosTheta, sinTheta, tx],
        [ -sinTheta,  cosTheta, ty]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])

================================
Rotate image the easy way
We can also use a built in function getRotationMatrix2D to rotate the image about any center. The syntax is given below.

retval  =   cv2.getRotationMatrix2D(    center, angle, scale    )
Parameters

center - Center of the rotation in the source image.
angle - Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the -top-left corner).
scale - Isotropic scale factor.

# Get rotation matrix
rotationMatrix = cv2.getRotationMatrix2D((centerX, centerY), angleInDegrees, 1)

# Warp Image
result = cv2.warpAffine(im, rotationMatrix, outDim)

# Display image
plt.imshow(result[...,::-1])

===================================

Shear Transformation

shearAmount = 0.1

warpMat = np.float32(
    [
        [ 1, shearAmount, 0],
        [ 0, 1.0        , 0]
    ])


# Warp image
result = cv2.warpAffine(im, warpMat, outDim, None, flags=cv2.INTER_LINEAR)

# Display image
plt.imshow(result[...,::-1])

```

### Complex Affine transformation
```
let's say we want to perform multiple operations -- rotation, scale, shear, and translate. We can obviously the transforms one after the other, but a more efficient way is to do this in one shot. This can be done by multiplying the non-translation part of the the matrices, and adding the translation parts.

Let's do a experiment where we first scale the image by 1.1, shear it by -0.1, rotate it by 10 degrees, and move in in the x direction by 10 pixels.

# Scale 
scaleAmount = 1.1
scaleMat = np.float32(
    [
        [ scaleAmount, 0.0,       ],
        [ 0,           scaleAmount]
    ])

# Shear 
shearAmount = -0.1 
shearMat = np.float32(
    [
        [ 1, shearAmount],
        [ 0, 1.0        ]
    ])

# Rotate by 10 degrees about (0,0)

angleInRadians = 10.0 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

rotMat = np.float32(
    [
        [ cosTheta, sinTheta],
        [ -sinTheta, cosTheta]
    ])

translateVector = np.float32(
    [
        [10],
        [0]
    ])

# First scale is applied, followed by shear, followed by rotation. 
scaleShearRotate = rotMat @ shearMat @ scaleMat

# Add translation
warpMat = np.append(scaleShearRotate, translateVector, 1)
print(warpMat)
outPts = scaleShearRotate @ np.float32([[50, 50],[50, 149],[149, 50], [149, 149]]).T + translateVector
print(outPts)

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])
```


### Complex Transformations using 3-Point Correspondences
```
We know that an affine transfrom that 6 degrees of freedom

Two for translation (tx, ty)
Two for scale (sx, sy)
One for shear
One for in-plane rotation
This means that if two images are related by an affine transform and we know the location of at least 3 points ( i.e. 6 coordinates ) in the source image and the destination image, we can recover the affine transform between them.

Now, let us consider the coordinates of 3 corners of the original square. They are located at (50,50), (50, 149) and (149, 50).

In the destination image above, the points are at (74, 50), (83,170), (192, 29) respectively. We can use the function estimateAffine2D to calculate the matrix.

srcPoints = np.float32([[50, 50],[50, 149],[149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[176, 27]])
estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]
print("True warp matrix:\n\n", warpMat)
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

If we have more point correspondences, we can use all of them to get better results. Here is an example of using all four points on the square.

srcPoints = np.float32([[50, 50],[50, 149],[149, 149], [149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[183, 135], [176, 27]])

estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]

print("True warp matrix:\n\n", warpMat)
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

# Warp image
result = cv2.warpAffine(im, estimatedMat, outDim)

# Display image
plt.imshow(result[...,::-1])

```

### Homography
```
A Homography is a transformation ( a 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.

To calculate a homography between two images, you need to know at least 4 point correspondences between the two images. If you have more than 4 corresponding points, it is even better. OpenCV will robustly estimate a homography that best fits all corresponding points. Usually, these point correspondences are found automatically by matching features like SIFT or SURF between the images, but in this section we are simply going to click the points by hand.

Homography Example
The code below shows how to take four corresponding points in two images and warp image onto the other. We specify the four corners of the book in source and destination image. Then use the findHmography function to find the matrix that relates the two set of points. Finally we apply this matrix on the source image using the warpPerspective function to get the final output.

Function Syntax</span>
pts_src and pts_dst are numpy arrays of points in source and destination images. We need at least 4 corresponding points.

h, status = cv2.findHomography(pts_src, pts_dst)
The calculated homography can be used to warp the source image to destination. Size is the size (width,height) of im_dst.

im_dst = cv2.warpPerspective(im_src, h, size)
Let us look at a more complete example in Python.

===============================================================================

# Read source image.
im_src = cv2.imread(DATA_PATH+'images/book2.jpg')
# Four corners of the book in source image
pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]], dtype=float)


# Read destination image.
im_dst = cv2.imread(DATA_PATH+'images/book1.jpg')
# Four corners of the book in destination image.
pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]], dtype=float)

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))


```


## Oriented FAST and Rotated BRIEF
```
FAST detects Features and BRIEF descriptors are for matching the feature points

ORB in OpenCV
Function Syntax
Let's have a look at the function syntax for cv2.ORB_create() function which is used to create an ORB detector.

retval  =   cv2.ORB_create()
The above function has many arguments, but the default ones work pretty well. It creates an object with 500 features points.

The ORB belongs to the feature2D class. It has a few important functions : cv2.Feature2D.detect(), cv2.Feature2D.compute() and cv2.Feature2D.detectAndCompute() which can be used as orb.detect(), orb.compute() and orb.detectAndCompute() where, orb = cv2.ORB_create().

Let's see the function syntax:

1. **cv2.Feature2D.detect()**
keypoints   =   cv2.Feature2D.detect(   image[, mask]   )
Where,

image - Image.
keypoints - The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .
mask - Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.

=======================================================================================

2. **cv2.Feature2D.compute()**
keypoints, descriptors  =   cv2.Feature2D.compute(  image, keypoints[, descriptors] )
Where,

image- Image.
keypoints - Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation).
descriptors - Computed descriptors. In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
And finally:

=======================================================================================

3. **cv2.Feature2D.detectAndCompute()**
keypoints, descriptors  =   cv2.Feature2D.detectAndCompute( image, mask[, descriptors[, useProvidedKeypoints]]  )
We can also draw the detected keypoints using cv2.drawKeypoints()

=======================================================================================

4. **cv2.drawKeypoints()**
outImage    =   cv2.drawKeypoints(  image, keypoints, outImage[, color[, flags]]    )
Where,

image - Source image.
keypoints - Keypoints from the source image.
outImage - Output image. Its content depends on the flags value defining what is drawn in the output image. See possible flags bit values below.
color - Color of keypoints.
flags - Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags. See details above in drawMatches
Below is a simple code which shows the use of ORB.

=======================================================================================

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(imgGray,None)

# compute the descriptors with ORB
kp, des = orb.compute(imgGray, kp)

# draw keypoints location, size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(12,12))
plt.imshow(img2[:,:,::-1])

=======================================================================================

orb = cv2.ORB_create(10)
kp, des = orb.detectAndCompute(imgGray, None)
img2 = cv2.drawKeypoints(img, kp, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(12,12))
plt.imshow(img2[:,:,::-1])

=======================================================================================


```

### Brute Force Matcher 
```
Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned

=======================================================================================

Create Matcher object 
retval  =   cv2.BFMatcher_create(   [, normType[, crossCheck]]  )

or

retval  =   cv2.BFMatcher()
It takes two optional params.

normType. It specifies the distance measurement to be used. By default, it is cv2.NORM_L2 which is good for SIFT, SURF etc. For binary string based descriptors like ORB, BRIEF, BRISK etc, cv2.NORM_HAMMING should be used, which uses Hamming distance as measurement.

crossCheck which is False by default. If it is True, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistent result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.


=======================================================================================
2. Match Features
Once the matcher is created, two important methods that can be used for mathing are

BFMatcher.match() - returns the best match, or
BFMatcher.knnMatch(). - returns k best matches where k is specified by the user. It may be useful when we need to do additional work on that.


=======================================================================================
3. Drawing Matches 
Like we used cv2.drawKeypoints() to draw keypoints, cv2.drawMatches() helps us to draw the matches. It stacks two images horizontally and lines are drawn from the first image to second showing best matches.

There is also cv2.drawMatchesKnn which draws all the k best matches. If k=2, it will draw two match-lines for each keypoint. So we have to pass a mask if we want to selectively draw it.


=======================================================================================

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()
```


### FLANN based Matcher 
```
FLANN stands for Fast Library for Approximate Nearest Neighbors. It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. It works faster than BFMatcher for large datasets.

Specify algorithm parameters
For FLANN based matcher, we need to specify the algorithm to be used. The following algorithms are implemented.Unfortunately, the names of the algorithms are not exposed in the Python API. Therefore, we need to use their ids. To make the code readable we first create a mapping between the algorithm name and index as shown below.

Algorithms	ID
FLANN_INDEX_LINEAR	0
FLANN_INDEX_KDTREE	1
FLANN_INDEX_KMEANS	2
FLANN_INDEX_COMPOSITE	3
FLANN_INDEX_KDTREE_SINGLE	4
FLANN_INDEX_HIERARCHICAL	5
FLANN_INDEX_LSH	6
FLANN_INDEX_SAVED	254
FLANN_INDEX_AUTOTUNED	255

IndexParams. - Specifies the algorithm to be used

e.g. For algorithms like SIFT, SURF etc. you can pass following:

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
For ORB, you can pass the following :

index_params= dict(algorithm = FLANN_INDEX_LSH,
                 table_number = 6,
                 key_size = 12,
                 multi_probe_level = 1)
SearchParams. It specifies the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time. If you want to change the value, pass search_params = dict(checks=100).

=======================================================================================


# Define FLANN algorithm indices
FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_KDTREE_SINGLE = 4
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_SAVED = 254
FLANN_INDEX_AUTOTUNED = 255


# FLANN parameters
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1, trainDescriptors = des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches[:10],None)

plt.imshow(img3)
plt.show()

```

### Image Alignment
```
The technique we will use is often called “feature based” image alignment because in this technique a sparse set of features are detected in one image and matched with the features in the other image. A transformation is then calculated based on these matched features that warps one image on to the other.

retval, mask    =   cv2.findHomography( srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]]   )
Where

srcPoints - Coordinates of the points in the original plane.
dstPoints - Coordinates of the points in the target plane.
method - Method used to compute a homography matrix. The following methods are possible:
0 - a regular method using all the points, i.e., the least squares method
RANSAC - RANSAC-based robust method
LMEDS - Least-Median robust method
RHO - PROSAC-based robust method
ransacReprojThreshold - Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods only). If srcPoints and dstPoints are measured in pixels, it usually makes sense to set this parameter somewhere in the range of 1 to 10.
mask - Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input mask values are ignored.
maxIters- The maximum number of RANSAC iterations.
confidence - Confidence level, between 0 and 1.

=======================================================================================


Step 1: Read Images

Step 2: Detect Features
We then detect ORB features in the two images. Although we need only 4 features to compute the homography, typically hundreds of features are detected in the two images. We control the number of features using the parameter MAX_FEATURES in the Python code.

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Convert images to grayscale
im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

Step 3: Match Features
We find the matching features in the two images, sort them by goodness of match and keep only a small percentage of original matches. We finally display the good matches on the images and write the file to disk for visual inspection. We use the hamming distance as a measure of similarity between two feature descriptors. The matched features are shown in the figure below by drawing a line connecting them. Notice, we have many incorrect matches and thefore we will need to use a robust method to calculate homography in the next step.

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(im, keypoints1, imReference, keypoints2, matches, None)
cv2.imwrite("matches.jpg", imMatches)

plt.figure(figsize=[20,10])
plt.imshow(imMatches[:,:,::-1])
plt.show()


Step 4: Calculate Homography
A homography can be computed when we have 4 or more corresponding points in two images. Automatic feature matching explained in the previous section does not always produce 100% accurate matches. It is not uncommon for 20-30% of the matches to be incorrect. Fortunately, the findHomography method utilizes a robust estimation technique called Random Sample Consensus (RANSAC) which produces the right result even in the presence of large number of bad matches.

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

Step 5: Warping Image
Once an accurate homography has been calculated, the transformation can be applied to all pixels in one image to map it to the other image. This is done using the warpPerspective function in OpenCV.

# Use homography
height, width, channels = imReference.shape
im1Reg = cv2.warpPerspective(im, h, (width, height))

plt.imshow(im1Reg[:,:,::-1])
plt.show()

# Print estimated homography
print("Estimated homography : \n",  h)

```


### Creating a Panorama
```
The basic principle is to align the concerned images using a homography and 'stitch'ing them intelligently so that you do not see the seams
ORB stands for Oriented FAST and Rotated BRIEF. Let’s see what FAST and BRIEF mean.

A feature point detector has two parts

Locator: This identifies points on the image that are stable under image transformations like translation (shift), scale (increase / decrease in size), and rotation. The locator finds the  (x,y)  coordinates of such points. The locator used by the ORB detector is called FAST.
Descriptor: The locator in the above step only tells us the location of interesting points. The second part of the feature detector is the descriptor which encodes the appearance of the point so that we can distinguish one feature point from the other. The descriptor evaluated at a feature point is simply an array of numbers. Ideally, the same physical point in two images should have the same descriptor. ORB uses a modified version of the feature descriptor called BRISK.

Find Keypoints and Descriptors for both images.
Find Corresponding points by matching their Descriptors.
Align second image with respect to first image using Homography.
Warp the second image using Perspective Transformation.
Combine the first image with the warped image to get the Panorama.

Step 1 : Find Keypoints and Descriptors

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


# Read reference image
imageFile1 = DATA_PATH + "images/scene/scene1.jpg"
print("Reading First image : ", imageFile1)
im1 = cv2.imread(imageFile1, cv2.IMREAD_COLOR)

# Read image to be aligned
imageFile2 = DATA_PATH + "images/scene/scene3.jpg"
print("Reading Second Image : ", imageFile2);
im2 = cv2.imread(imageFile2, cv2.IMREAD_COLOR)

# Convert images to grayscale
im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

im1Keypoints = np.array([])
im1Keypoints = cv2.drawKeypoints(im1, keypoints1, im1Keypoints, color=(0,0,255),flags=0)
print("Saving Image with Keypoints")
cv2.imwrite("keypoints.jpg", im1Keypoints)

plt.imshow(im1Keypoints[:,:,::-1])
plt.title("Keypoints obtained from the ORB detector")
====================================================

Step 2 : Find matching corresponding points

# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(im1, keypoints1,
                            im2, keypoints2, 
                            matches, None)
                            
plt.figure(figsize=[15,10])
plt.imshow(imMatches[:,:,::-1])
plt.title("Matchings obtained from the descriptor matcher")
====================================================

Step 3 : Image Alignment using Homography

A homography can be computed when we have 4 or more corresponding points in two images. Automatic feature matching explained in the previous step does not always produce 100% accurate matches. It is not uncommon for 20-30% of the matches to be incorrect. Fortunately, the findHomography method utilizes a robust estimation technique called Random Sample Consensus (RANSAC) which produces the right result even in the presence of large number of bad matches.

Code
After matching is done, the output ( matches ) has the following attributes :

matches.distance - Distance between descriptors. Should be lower for better match.
matches.trainIdx - Index of the descriptor in train descriptors
matches.queryIdx - Index of the descriptor in query descriptors
matches.imgIdx - Index of the train image.
To simplify things, the queryIdx corresponds to points in image1 and trainIdx corresponds to points in image2. We will create two lists - points1 and points2 from the matching points which will serve as the final set of correspoding points to be used in the findHomography function.


points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
print("Homograhy matrix \n{}".format(h))

====================================================

Step 4 : Warp Image

# Use homography
im1Height, im1Width, channels = im1.shape
im2Height, im2Width, channels = im2.shape

im2Aligned = cv2.warpPerspective(im2, h, 
                            (im2Width + im1Width, im2Height))
                            
plt.figure(figsize=[15,10])
plt.imshow(im2Aligned[:,:,::-1])
plt.title("Second image aligned to first image obtained using homography and warping")
                            
====================================================

Step 5 : Stitch Images

# Stitch Image 1 with aligned image 2
stitchedImage = np.copy(im2Aligned)
stitchedImage[0:im1Height,0:im1Width] = im1

plt.figure(figsize=[15,10])
plt.imshow(stitchedImage[:,:,::-1])
plt.title("Final Stitched Image")

```


### Object Detection using feature matching
```
Find Features in both images
============================

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2Gray, None)


Set up the matcher
==================

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

Find Matches or Corresponding points
====================================

matches = flann.knnMatch(np.float32(descriptors1),np.float32(descriptors2),k=2)

Find good matches 
=================

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)
        
       
Find the location of the book in the cluttered image
====================================================
This is done in 2 steps:

First find the homography matrix using the corresponding points
We know the location of the book in the first image, thus, find the location of the 4 points in the cluttered image using the computed Homography.        

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
#     Points in the original image
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     Find points in the Cluttered image corresponding to the book
    dst = cv2.perspectiveTransform(pts,M)
#     Draw a red box around the detected book
    img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
  
Display the matches and the detected object  
===========================================

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good,None,**draw_params)

plt.figure(figsize=(12,12))
plt.imshow(img3[...,::-1]),plt.show()
```


### Grabcut in OpenCV
```
GrabCut is an interactive segmentation method. It is used to separate an image into a background and a foreground. 

mask, bgdModel, fgdModel    =   cv.grabCut( img, mask, rect, bgdModel, fgdModel, iterCount[, mode]  )
Parameters

img: Input 8-bit 3-channel image.
mask: Input/output 8-bit single-channel mask. The mask is initialized by the function when mode is set to GC_INIT_WITH_RECT. Its elements may have one of the GrabCutClasses.
rect: ROI containing a segmented object. The pixels outside of the ROI are marked as "obvious background". The parameter is only used when mode==GC_INIT_WITH_RECT .
bgdModel: Temporary array for the background model. Do not modify it while you are processing the same image.
fgdModel: Temporary arrays for the foreground model. Do not modify it while you are processing the same image.
iterCount: Number of iterations the algorithm should make before returning the result. Note that the result can be refined with further calls with mode==GC_INIT_WITH_MASK or mode==GC_EVAL .
mode: Operation mode that could be one of the GrabCutModes

REFER TO THE GRABCUT.PY IN THE REPOS

```


### Image Classification using HOG + SVM
```
cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

==========================================================================

winSize = This parameter is set to the size of the window over which the descriptor is calculated. In classification problems, we set it to the size of the image. E.g. we set it to 64x128 for pedestrian detection.

blockSize = The size of the blue box in the image. The notion of blocks exist to tackle illumination variation. A large block size makes local changes less significant while a smaller block size weights local changes more. Typically blockSize is set to 2 x cellSize.

blockStride = The blockStride determines the overlap between neighboring blocks and controls the degree of contrast normalization. Typically a blockStride is set to 50% of blockSize.

cellSize = The cellSize is the size of the green squares. It is chosen based on the scale of the features important to do the classification. A very small cellSize would blow up the size of the feature vector and a very large one may not capture relevant information.

nbins = Sets the number of bins in the histogram of gradients. The authors of the HOG paper had recommended a value of 9 to capture gradients between 0 and 180 degrees in 20 degrees increments.

derivAperture = Size of the Sobel kernel used for derivative calculation.

winSigma = According to the HOG paper, it is useful to “downweight pixels near the edges of the block by applying a Gaussian spatial window to each pixel before accumulating orientation votes into cells”. winSigma is the standard deviation of this Gaussian. In practice, it is best to leave this parameter to default ( -1 ). On doing so, winSigma is automatically calculated as shown below: winSigma = ( blockSize.width + blockSize.height ) / 8

histogramNormType = In the HOG paper, the authors use four different kinds of normalization. OpenCV 3.2 implements only one of those types L2Hys. So, we simply use the default. L2Hys is simply L2 normalization followed by a threshold (L2HysThreshold)where all values above a threshold are clipped to that value.

L2HysThreshold = Threshold used in L2Hys normalization. E.g. If the L2 norm of a vector is [0.87, 0.43, 0.22], the L2Hys normalization with L2HysThreshold = 0.8 is [0.8, 0.43, 0.22].

gammaCorrection = Boolean indicating whether or not Gamma correction should be done as a pre-processing step.

nlevels = Number of pyramid levels used during detection. It has no effect when the HOG descriptor is used for classification.

signedGradient = Typically gradients can have any orientation between 0 and 360 degrees. These gradients are referred to as “signed” gradients as opposed to “unsigned” gradients that drop the sign and take values between 0 and 180 degrees. In the original HOG paper, unsigned gradients were used for pedestrian detection.

=============================================================================================

Please refer to the HOG descriptor file in this repo

```


### HAAR CASCADES
```
Paul Viola and Michael Jones came up with their seminal paper which not only detected faces robustly, but did so in real-time. It is one of the most cited papers in Computer Vision

cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors]])
Where,

image is the input grayscale image.
objects is the rectangular region enclosing the objects detected
scaleFactor is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
minNeighbors is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. Higher number gives lower false positives.

we discuss the effect of the parameters. The effect of scaleFactor is mostly related to speed. Lower the value, slower will be the speed. In this example, we check the effect of the minNeighbors parameter. As the value of this variable is increased, false positives are decreased

=============================================================================================

# Load the cascade classifier from the xml file.
faceCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_frontalface_default.xml')
faceNeighborsMax = 10
neighborStep = 1

# Read the image
frame = cv2.imread(DATA_PATH + "images/hillary_clinton.jpg")
frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Perform multi scale detection of faces
plt.figure(figsize=(18,18))
count = 1
for neigh in range(1, faceNeighborsMax, neighborStep):
    faces = faceCascade.detectMultiScale(frameGray, 1.2, neigh)
    frameClone = np.copy(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frameClone, (x, y), 
                      (x + w, y + h), 
                      (255, 0, 0),2)

    cv2.putText(frameClone, 
    "# Neighbors = {}".format(neigh), (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    
    plt.subplot(3,3,count)
    plt.imshow(frameClone[:,:,::-1])
    count += 1

plt.show()

==============================================================================================

Face and Smile Detection
The effect of minNeighbors was not very pronounced in the face detection example as the face has very unique features. The mouth or smile on the other hand is very difficult to detect without false positives. This is illustrated using the example

# Detect the face using the cascade
faceCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_smile.xml')
smileNeighborsMax = 90
neighborStep = 10

frame = cv2.imread(DATA_PATH + "images/hillary_clinton.jpg")

frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(frameGray, 1.4, 5)

# Get the face area from the detected face rectangle
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), 
                  (x + w, y + h), 
                  (255, 0, 0), 2)
    faceRoiGray = frameGray[y: y + h, x: x + w]
    faceRoiOriginal = frame[y: y + h, x: x + w]

count = 1
plt.figure(figsize=(18,18))
# Detect the smile from the detected face area and display the image
for neigh in range(1, smileNeighborsMax, neighborStep):
    smile = smileCascade.detectMultiScale(faceRoiGray, 
                          1.5, neigh)

    frameClone = np.copy(frame)
    faceRoiClone = frameClone[y: y + h, x: x + w]
    for (xx, yy, ww, hh) in smile:
        cv2.rectangle(faceRoiClone, (xx, yy), 
                      (xx + ww, yy + hh), 
                      (0, 255, 0), 2)

    cv2.putText(frameClone, 
              "# Neighbors = {}".format(neigh), 
              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
              (0, 0, 255), 4)
    plt.subplot(3,3,count)
    plt.imshow(frameClone[:,:,::-1])
    count += 1

plt.show()

```


### Lucas canade optical flow
```
To detect the points in a moving image. 

==================================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'

videoFileName = DATA_PATH + "videos/cycle.mp4"

cap = cv2.VideoCapture(videoFileName)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('sparse-output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, (width,height))

====================================================================================

Detect Corners for tracking them

We will use the Shi Tomasi corner detection algorithm to find some points which we will track over the video. It is implemented in OpenCV using the function goodFeaturesToTrack.

Function Syntax 
cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, mask[, blockSize]])
where,

image - Input image
maxCorners - maximum Number of corners to be detected
qualityLevel - Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure
minDistance - Minimum possible Euclidean distance between the returned corners.
mask - Optional region of interest
blockSize - Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
We are specifying the parameters in a separate dictionary as given below.

====================================================================================
# params for ShiTomasi corner detection
numCorners = 100
feature_params = dict( maxCorners = numCorners,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
                       
                       
Set up the Lucas Kanade Tracker 
After detecting certain points in the first frame, we want to track them in the next frame. This is done using Lucas Kanade algorithm. It is implemented in OpenCV using the following function.

Function Syntax 
nextPts, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, winSize[, maxLevel[, criteria]]])
where,

prevImg - previous image
nextImg - next image
prevPts - points in previous image
nextPts - points in next image
winSize - size of the search window at each pyramid level
maxLevel - 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on
criteria - parameter, specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon


=====================================================================================================
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                  
# Create some random colors
color = np.random.randint(0,255,(numCorners,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing the tracks 
mask = np.zeros_like(old_frame)
count = 0
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count += 1
    # calculate optical flow
    new_points, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

    # Select good points
    good_new = new_points[status==1]
    good_old = old_points[status==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2, cv2.LINE_AA)
        cv2.circle(frame,(a,b),3,color[i].tolist(), -1)
    
    
    # display every 5th frame
    display_frame = cv2.add(frame,mask)
    out.write(display_frame)
    if count % 5 == 0:
        plt.imshow(display_frame[:,:,::-1])
        plt.show()
    if count > 50:
        break
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    old_points = good_new.reshape(-1,1,2)
    
```


### Video Stabilization
```
Video stabilization refers to a family of methods used to reduce the effect of camera motion on the final video. The motion of the camera would be a translation ( i.e. movement in the x, y, z-direction ) or rotation (yaw, pitch, roll).

Digital Video Stabilization: This method does not require special sensors for estimating camera motion. There are three main steps — 1) motion estimation 2) motion smoothing, and 3) image composition. The transformation parameters between two consecutive frames are derived in the first stage. The second stage filters out unwanted motion and in the last stage the stabilized video is reconstructed.

=========================================================================

Step 1 : Set Input and Output Videos

import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read input video
cap = cv2.VideoCapture(DATA_PATH+'videos/video.mp4')

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up output video
out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w*2, h))

# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


=========================================================================

Step 2: Find good features to track


We will iterate over all the frames, and find the motion between the current frame and the previous frame. It is not necessary to know the motion of each and every pixel. The Euclidean motion model requires that we know the motion of only 2 points in the two frames. However, in practice, it is a good idea to find the motion of 50-100 points, and then use them to robustly estimate the motion model.


corners =   cv2.goodFeaturesToTrack(    image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize[, corners[, useHarrisDetector[, k]]]    )
Where,

image - Input 8-bit or floating-point 32-bit, single-channel image.
corners - Output vector of detected corners.
maxCorners - Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
qualityLevel - Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response. The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
minDistance - Minimum possible Euclidean distance between the returned corners.
mask - Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
blockSize - Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
useHarrisDetector - Parameter indicating whether to use a Harris detector or cornerMinEigenVal.
k - Free parameter of the Harris detector.

=========================================================================

Step 3: Lucas-Kanade Optical Flow

Once we have found good features in the previous frame, we can track them in the next frame using an algorithm called Lucas-Kanade Optical Flow named after the inventors of the algorithm.

It is implemented using the function calcOpticalFlowPyrLK in OpenCV. In the name calcOpticalFlowPyrLK, LK stands for Lucas-Kanade, and Pyr stands for the pyramid. An image pyramid in computer vision is used to process an image at different scales (resolutions).

Function Syntax
nextPts, status, err    =   cv2.calcOpticalFlowPyrLK(   prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]  )
Where,

prevImg - first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
nextImg - second input image or pyramid of the same size and the same type as prevImg.
prevPts - vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.
nextPts - output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
status - output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
err - output vector of errors; each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
winSize - size of the search window at each pyramid level.
maxLevel -0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel.
criteria - parameter, specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon.
flags - operation flags:
OPTFLOW_USE_INITIAL_FLOW uses initial estimations, stored in nextPts; if the flag is not set, then prevPts is copied to nextPts and is considered the initial estimate.
OPTFLOW_LK_GET_MIN_EIGENVALS use minimum eigen values as an error measure (see minEigThreshold description); if the flag is not set, then L1 distance between patches around the original and a moved point, divided by number of pixels in a window, is used as a error measure.
minEigThreshold - the algorithm calculates the minimum eigen value of a 2x2 normal matrix of optical flow equations (this matrix is called a spatial gradient matrix), divided by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding feature is filtered out and its flow is not processed, so it allows to remove bad points and get a performance boost.
calcOpticalFlowPyrLK may not be able to calculate the motion of all the points because of a variety of reasons. For example, the feature point in the current frame could get occluded by another object in the next frame. Fortunately, as you will see in the code below, the status flag in calcOpticalFlowPyrLK can be used to filter out these values.


=========================================================================

Step 4: Robustly estimate transform

To recap, in step 3.1, we found good features to track in the previous frame. In step 3.2, we used optical flow to track the features. In other words, we found the location of the features in the current frame, and we already knew the location of the features in the previous frame. So we can use these two sets of points to find the rigid (Euclidean) transformation that maps the previous frame to the current frame. This is done using the function estimateRigidTransform.

Once we have estimated the motion, we can decompose it into x and y translation and rotation (angle). We store these values in an array so we can change them smoothly.

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)

for i in range(n_frames-2):
  # Detect feature points in previous frame
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
    
  # Read next frame
  success, curr = cap.read() 
  if not success: 
    break
 
  # Convert to grayscale
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
 
  # Calculate optical flow (i.e. track feature points)
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
 
  # Sanity check
  assert prev_pts.shape == curr_pts.shape 
 
  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]
 
  #Find transformation matrix
  m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
  # Extract traslation
  dx = m[0][0,2]
  dy = m[0][1,2]
 
  # Extract rotation angle
  da = np.arctan2(m[0][1,0], m[0][0,0])
    
  # Store transformation
  transforms[i] = [dx,dy,da]
    
  # Move to next frame
  prev_gray = curr_gray
 
  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
  

=========================================================================

Step 5: Calculate smooth motion between frames

 the previous step, we estimated the motion between the frames and stored them in an array. We now need to find the trajectory of motion by cumulatively adding the differential motion estimated in the previous step.

Step 5.1 : Calculate trajectory
In this step, we will add up the motion between the frames to calculate the trajectory. Our ultimate goal is to smooth out this trajectory.

In Python, it is easily achieved using cumsum (cumulative sum) in numpy.


# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

=========================================================================
Step 5.2 : Calculate smooth trajectory

The easiest way to smooth any curve is to use a moving average filter.

def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed
  
def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
  return smoothed_trajectory
  
 ========================================================================= 
  
Step 5.3: Calculate smooth transforms
So far we have obtained a smooth trajectory. In this step, we will use the smooth trajectory to obtain smooth transforms that can be applied to frames of the videos to stabilize it.

This is done by finding the difference between the smooth trajectory and the original trajectory and adding this difference back to the original transforms.

 ========================================================================= 
 
 Step 6: Apply smoothed camera motion to frames
 
 When we stabilize a video, we may see some black boundary artifacts. This is expected because to stabilize the video, a frame may have to shrink in size.

We can mitigate the problem by scaling the video about its center by a small amount (e.g. 4%).

The function fixBorder below shows the implementation. We use getRotationMatrix2D because it scales and rotates the image without moving the center of the image. All we need to do is call this function with 0 rotation and scale 1.04 ( i.e. 4% upscale).

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame
  
# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break
 
  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]
 
  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy
 
  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
 
  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 
 
  # Write the frame to the file
  frame_out = cv2.hconcat([frame, frame_stabilized])
 
  # If the image is too big, resize it.
  if(frame_out.shape[1] > 1920): 
    frame_out = cv2.resize(frame_out, (w,h))
  #cv2.imshow("Frame",frame_out)
  #cv2.waitKey(0)

  out.write(frame_out)
  
cv2.destroyAllWindows()
out.release()


```


### Object Tracking
```
There are 8 different trackers available in OpenCV 4.1.0

BOOSTING Tracker
This tracker is based on an online version of AdaBoost — the algorithm that the HAAR cascade based face detector uses internally. This classifier needs to be trained at runtime with positive and negative examples of the object. The initial bounding box supplied by the user ( or by another object detection algorithm ) is taken as the positive example for the object, and many image patches outside the bounding box are treated as the background. Given a new frame, the classifier is run on every pixel in the neighborhood of the previous location and the score of the classifier is recorded. The new location of the object is the one where the score is maximum. So now we have one more positive example for the classifier. As more frames come in, the classifier is updated with this additional data.

Pros : None. This algorithm is a decade old and works ok, but I could not find a good reason to use it especially when other advanced trackers (MIL, KCF) based on similar principles are available.

Cons : Tracking performance is mediocre. It does not reliably know when tracking has failed.
==============================================================================================
MIL Tracker
This tracker is similar in idea to the BOOSTING tracker described above. The big difference is that instead of considering only the current location of the object as a positive example, it looks in a small neighborhood around the current location to generate several potential positive examples. You may be thinking that it is a bad idea because in most of these "positive" examples the object is not centered.

This is where Multiple Instance Learning ( MIL ) comes to rescue. In MIL, you do not specify positive and negative examples, but positive and negative "bags". The bag is labeled as positive if any of the instance in the bag is labeled as positive by the classifier. Otherwise the bag is labeled as negative. We will discuss this method in detail in the next chapter.

Pros : The performance is pretty good. It does not drift as much as the BOOSTING tracker and it does a reasonable job under partial occlusion.

Cons : Tracking failure is not reported reliably. Does not recover from full occlusion.
==============================================================================================
KCF Tracker
KCF stands for Kernelized Correlation Filters. This tracker builds on the ideas presented in the previous two trackers. This tracker utilizes the fact that the multiple positive samples used in the MIL tracker have large overlapping regions. This overlapping data leads to some nice mathematical properties that is exploited by this tracker to make tracking faster and more accurate at the same time.

Pros: Accuracy and speed are both better than MIL and it reports tracking failure better than BOOSTING and MIL.

Cons : Does not recover from full occlusion.

TLD Tracker
TLD stands for Tracking, learning and detection. As the name suggests, this tracker decomposes the long term tracking task into three components — (short term) tracking, learning, and detection. From the author’s paper, "The tracker follows the object from frame to frame. The detector localizes all appearances that have been observed so far and corrects the tracker if necessary. The learning estimates detector’s errors and updates it to avoid these errors in the future." This output of this tracker tends to jump around a bit. For example, if you are tracking a pedestrian and there are other pedestrians in the scene, this tracker can sometimes temporarily track a different pedestrian than the one you intended to track. On the positive side, this track appears to track an object over a larger scale, motion, and occlusion. If you have a video sequence where the object is hidden behind another object, this tracker may be a good choice.

Pros : Works the best under occlusion over multiple frames. Also, tracks best over scale changes.

Cons : Lots of false positives making it almost unusable.
==============================================================================================
MEDIANFLOW Tracker
Internally, this tracker tracks the object in both forward and backward directions in time and measures the discrepancies between these two trajectories. Minimizing this "ForwardBackward" error enables them to reliably detect tracking failures and select reliable trajectories in video sequences.

In ours tests, we found this tracker works best when the motion is predictable and small. Unlike, other trackers that keep going even when the tracking has clearly failed, this tracker knows when the tracking has failed.

Pros : Excellent tracking failure reporting. Works very well when the motion is predictable and there is no occlusion.

Cons : Fails under large motion.

==============================================================================================

GOTURN tracker
Out of all the tracking algorithms in the tracker class, this is the only one based on Convolutional Neural Network (CNN). It is also the only one that uses an offline trained model, because of which it is faster that other trackers. From OpenCV documentation, we know it is "robust to viewpoint changes, lighting changes, and deformations". But it does not handle occlusion very well.

**NOTE :** GOTURN being a CNN based tracker, uses a caffe model for tracking. The Caffe model and the prototxt file must be present in the directory in which the code is present. You can download the files from this link.
==============================================================================================
MOSSE tracker
The idea of using correlation filters for tracking is very old. However, if we simply use an image patch around the detected object and try to find its location in the next frame using correlation the results are not very good. This is because the image patch appearance may change quite a bit.

Minimum Output Sum of Squared Error (MOSSE) uses discriminative correlation filter (DCF) for object tracking which produces stable correlation filters when initialized using a single frame. When the paper was published in 2010, it surprised the community because of it simplicity. It was an old idea that was modified slightly, and was able to outperform other algorithms that used heavy duty classifiers, complex appearance models, and stochastic search techniques. It was also substantially faster.

MOSSE tracker is robust to variations in lighting, scale, pose, and non-rigid deformations. It also detects occlusion based upon the peak-to-sidelobe ratio, which enables the tracker to pause and resume where it left off when the object reappears. MOSSE tracker also operates at a higher fps (450 fps and even more).
==============================================================================================
CSRT tracker
The CRST tracker extends the Discriminative Correlation Filter (DCF) idea in MOSSE with what the authors call Channel and Spatial Reliability (DCF-CSR). In particular, they are able to extend the search region over while the search is performed. This ensures enlarging and localization of the selected region and improved tracking of the non-rectangular regions or objects. It uses only 2 standard features (HoGs and Colornames). It also operates at a comparatively lower fps (25 fps) but gives higher accuracy for object tracking.
==============================================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
%matplotlib inline


import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
tracker_type = tracker_types[2]


if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()
elif tracker_type == "MOSSE":
    tracker = cv2.TrackerMOSSE_create()
else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
        print(t)
        
# Read video
video = cv2.VideoCapture(DATA_PATH + "videos/hockey.mp4")

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

# Define a few colors for drawing
red = (0,0,255)
blue = (255,128,0)


# Define an initial bounding box
# Cycle
bbox = (477, 254, 55, 152)

# ship
# bbox = (751, 146, 51, 78)

# Hockey
# bbox = (129, 47, 74, 85)

# Face2
# bbox = (237, 145, 74, 88)

# meeting
# bbox = (627, 183, 208, 190)     #CSRT
# bbox = (652, 187, 118, 123)       #KCF

# surfing
# bbox = (97, 329, 118, 293)

# surf
# bbox = (548, 587, 52, 87)

# spinning
# bbox = (232, 218, 377, 377)       #RED
# bbox = (699, 208, 383, 391)         #BLUE

# Car
# bbox = (71, 457, 254, 188)


# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

# Display bounding box.
p1 = (int(bbox[0]), int(bbox[1]))
p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
cv2.rectangle(frame, p1, p2, blue, 2, 1 )

plt.imshow(frame[:,:,::-1])
plt.title("Tracking")

# We will display only first 5 frames
count = 0

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()
    
    # The update method is used to obtain the location 
    # of the new tracked object. The method returns
    # false when the track is lost. Tracking can fail 
    # because the object went outside the video frame or 
    # if the tracker failed to track the object. 
    # In both cases, a false value is returned.
    
    # Update tracker
    ok, bbox = tracker.update(frame)
    
    # Calculate processing time and display results.
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
      # Tracking success
      p1 = (int(bbox[0]), int(bbox[1]))
      p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
      cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
      # Tracking failure
      cv2.putText(frame, "Tracking failure detected", (20,80), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.75,red,2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (20,20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue,2);
    
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (20,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2);

    # Display result
    plt.imshow(frame[:,:,::-1])
    plt.show()
    
    count += 1
    if count == 5:
        break

```


### Tracking Multiple Objects
```
OpenCV has a Multiobject Tracker class which has a very basic implementation of a multi object tracker. It processes the tracked objects independently without any optimization across the tracked objects

import cv2
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker
  
print("Default tracking algoritm is CSRT \n"
    "Available tracking algorithms are:\n")
for t in trackerTypes:
    print(t)

trackerType = "CSRT"

# Set video to load
videoPath = DATA_PATH + "videos/cycle.mp4"


# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()

# quit if unable to read the video file
if not success:
    print('Failed to read video')
    
## Select boxes
colors = []
for i in range(3):
    # Select some random colors
    colors.append((randint(64, 255), randint(64, 255),
                randint(64, 255)))
# Select the bounding boxes
bboxes = [(471, 250, 66, 159), (349, 232, 69, 102)]
print('Selected bounding boxes {}'.format(bboxes))

## Initialize MultiTracker
# There are two ways you can initialize multitracker
# 1. tracker = cv2.MultiTracker("CSRT")
# All the trackers added to this multitracker
# will use CSRT algorithm as default
# 2. tracker = cv2.MultiTracker()
# No default algorithm specified

# Initialize MultiTracker with tracking algo
# Specify tracker type

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    
# We will display only 5 frames
count = 0

# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 4, cv2.LINE_AA)

    # show frame
    if count % 10 == 0:
        plt.imshow(frame[:,:,::-1])
        plt.show()
    
    count += 1
    
    if count > 50:
        break
cap.release()

```


### kalman Filter
```

The Kalman filter is a method for tracking the internal state of the system based on internal dynamics and control inputs while fusing independent measurements of the state. It has two steps :

Predict: A prediction is made based on internal dynamics and control inputs.

Update : An update is made to the prediction based on independent measurements.

The KalmanFilter class in OpenCV implements the Kalman filter. In the tutorial below, we will use Kalman Filtering to track a person walking. We first detect the person using a HOG based person detector. We then initialize a Kalman filter to track the the top left corner (x, y) and the width (w) of the bounding box. We use a simple motion model where x, y and w have velocities but no acceleration. Consequently, our state has 6 elements ( x, y, w, vx, vy, vw ) and the measurement has 3 elements ( x, y, w ). There are no control inputs because in a recorded video we have not way to influence the motion of the object.

=============================================================================================

import os
import sys 
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Function to detect the rectangle with the maximum area
# To detect max object area in multiple array of object(x,y,w,h) 
def maxRectArea(rects):
  area = 0
  maxRect = rects[0].copy()
  for rect in rects:
    x, y, w, h = rect.ravel()
    if w*h > area:
      area = w*h
      maxRect = rect.copy()
  maxRect = maxRect[:, np.newaxis]
  return maxRect

# Initialize hog descriptor for people detection
# Initialize hog descriptor for people detection
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# Initialize HOG
hog = cv2.HOGDescriptor(winSize, blockSize, 
                      blockStride,cellSize, 
                      nbins, derivAperture,
                      winSigma, histogramNormType, 
                      L2HysThreshold,gammaCorrection, 
                      nlevels, signedGradient)

svmDetector = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(svmDetector)
#  Load video
cap = cv2.VideoCapture(DATA_PATH + "videos/boy-walking.mp4")

# Confirm video is open
if not cap.isOpened():
    print("Unable to read video")

# Variable for storing frames
frameDisplay = []


blue = (255, 0, 0)
red = (0, 0, 255)


# Initialize Kalman filter. 
# OpenCV Kalman filter is initialized using

KalmanFilter KF(numStateVariables, numMeasurements, numControlInputs, type);
In our Kalman filter, the state consists of 6 elements (x, y, w, vx, vy, vw) of the bounding box where,

x, y = Coordinates of the top left corner of the box

w = Width of the detected object

vx, vy = x and y velocities of top left corner of the box.

vw = rate of change of width with respect to time.

The height is not part of the state because height is always twice the width.

Hence, numStateVariables = 6.

The measurement matrix has 3 elements (x, y, w) which are simply the x and y coordinates of the top left corner of the detected object and the width of the object.

Hence numMeasurements = 3

There are no controlInputs because this is a recorded video and there is no way for us to change the affect the state of the person walking.

The type is set to float32

# Internal state has 6 elements (x, y, width, vx, vy, vw)
# Measurement has 3 elements (x, y, width ).
# Note: Height = 2 x width, so it is not part of the state
# or measurement.
KF = cv2.KalmanFilter(6, 3, 0)

Motion Model and Transition Matrix 
Because our motion model is

x = x + vx * dt

y = y + vy * dt

w = w + vw * dt

For simplicity, we assume zero accelaration. Therefore,

vx = vx

vy = vy

vw = vw

Therefore, the transition matrix is of the form

[

1, 0, 0, dt, 0, 0,

0, 1, 0, 0, dt, 0,

0, 0, 1, 0, 0, dt,

0, 0, 0, 1, 0, 0,

0, 0, 0, 0, 1, 0,

0, 0, 0, 0, 0, 1

]

We set it to identity and later add dt in a loop.

KF.transitionMatrix = cv2.setIdentity(KF.transitionMatrix)
print(KF.transitionMatrix)

Measurement matrix is of the form

[

1, 0, 0, 0, 0, 0,

0, 1, 0, 0, 0, 0,

0, 0, 1, 0, 0, 0,

]

because we are only detecting x, y and w. The measurement matrix picks those quantities and leaves vx, vy, vw.

KF.measurementMatrix = cv2.setIdentity(KF.measurementMatrix)
print(KF.measurementMatrix)

# Initializing variables to be used for tracking and bookkeeping
# Variable to store detected x, y and w
measurement = np.zeros((3, 1), dtype=np.float32)
# Variables to store detected object and tracked object
objectTracked = np.zeros((4, 1), dtype=np.float32)
objectDetected = np.zeros((4, 1), dtype=np.float32)

# Variables to store results of the predict and update 
# (a.k.a update step).
updatedMeasurement = np.zeros((3, 1), dtype=np.float32)
predictedMeasurement = np.zeros((6, 1), dtype=np.float32)

# Variable to indicate measurement was updated
measurementWasUpdated = False

# Timing variable
ticks = 0
preTicks = 0

# Read frames until object is detected for the first time
success=True
while success:
    sucess, frame = cap.read()
    objects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                        scale=1.05, hitThreshold=0, finalThreshold=1,
                                        useMeanshiftGrouping=False)

    # Update timer
    ticks = cv2.getTickCount()

    if len(objects) > 0:
        # Copying max object area values to Kalman Filter
        objectDetected = maxRectArea(objects)
        measurement = objectDetected[:3].astype(np.float32)

        # Update state. Note x, y, w are set to measured values.
        # vx = vy = vw because we have no idea about the velocities yet.
        KF.statePost[0:3, 0] = measurement[:, 0]
        KF.statePost[3:6] = 0.0

        # Set diagonal values for covariance matrices.
        # processNoiseCov is Q
        KF.processNoiseCov = cv2.setIdentity(KF.processNoiseCov, (1e-2))
        KF.measurementNoiseCov = cv2.setIdentity(KF.measurementNoiseCov, (1e-2))
        break

# Apply Kalman Filter

# dt for Transition matrix
dt = 0.0
# Random number generator for randomly selecting frames for update
random.seed(42)

# Loop over rest of the frames
# We will display output for only first 5 frames
count = 0
while True:
    success, frame = cap.read()
    if not success:
        break

    # Variable for displaying tracking result
    frameDisplay = frame.copy()
    # Variable for displaying detection result
    frameDisplayDetection = frame.copy()

    # Update dt for transition matrix.
    # dt = time elapsed.
    preTicks = ticks;
    ticks = cv2.getTickCount()
    dt = (ticks - preTicks) / cv2.getTickFrequency()

    KF.transitionMatrix[0, 3] = dt
    KF.transitionMatrix[1, 4] = dt
    KF.transitionMatrix[2, 5] = dt

    predictedMeasurement = KF.predict()

    # Detect objects in current frame
    objects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                            scale=1.05, hitThreshold=0, finalThreshold=1,
                                            useMeanshiftGrouping=False)
    if len(objects) > 0:
        # Find largest object
        objectDetected = maxRectArea(objects)

        # Display detected rectangle
        x1, y1, w1, h1 = objectDetected.ravel()
        cv2.rectangle(frameDisplayDetection, (x1, y1), (x1+w1, y1+h1), red, 2, 4)

    # We will update measurements 15% of the time.
    # Frames are randomly chosen.
    update = random.randint(0, 100) < 15

    if update:
        # Kalman filter update step
        if len(objects) > 0:
            # Copy x, y, w from the detected rectangle
            measurement = objectDetected[0:3].astype(np.float32)

            # Perform Kalman update step
            updatedMeasurement = KF.correct(measurement)
            measurementWasUpdated = True
        else:
            # Measurement not updated because no object detected
            measurementWasUpdated = False
    else:
        # Measurement not updated
        measurementWasUpdated = False

    if measurementWasUpdated:
        # Use updated measurement if measurement was updated
        objectTracked[0:3, 0] = updatedMeasurement[0:3, 0].astype(np.int32)
        objectTracked[3, 0] = 2*updatedMeasurement[2, 0].astype(np.int32)
    else:
        # If measurement was not updated, use predicted values.
        objectTracked[0:3, 0] = predictedMeasurement[0:3, 0].astype(np.int32)
        objectTracked[3, 0] = 2*predictedMeasurement[2, 0].astype(np.int32)

    # Draw tracked object
    x2, y2, w2, h2 = objectTracked.ravel()
    cv2.rectangle(frameDisplay, (x2, y2), (x2+w2, y2+h2), blue, 2, 4)

    # Text indicating Tracking or Detection.
    cv2.putText(frameDisplay, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
    cv2.putText(frameDisplayDetection, "Detection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)

    # Concatenate detected result and tracked result vertically
    output = np.concatenate((frameDisplayDetection, frameDisplay), axis=0)

    # Display result.
    plt.imshow(output)
    plt.show()
    count += 1
    if count == 5:
        break
cap.release()

Results
First, note the tracking results are much smoother than the detection results. In other words, noise in tracked motion is less than noise in repeated detection. Hence, the name Kalman “Filtering” where the noise is filtered out.

Second, the quality of tracking depends on the uncertainty in motion. If the motion is according to our motion model, tracking will produce very good results. However, if the object changes direction abruptly, prediction will be off until the update step is used.

Finally, Kalman filtering shown in this tutorial does not use pixel information at all for tracking. Can the results be improved if we use pixel information in addition to motion information? Of course, and that is exactly what we will learn in trackers in the next few sections.

```


### Meanshift
```
Meanshift is a non-parametric approach for finding the mode of a set of points. In other words, it finds the maxima of a density function. It was first presented by Fukunaga and Hostetler in 1975 in their paper. Variations of meanshift algorithm are used for applications like Image segmentation and edge preserving filtering.

Find the color histogram of the object of interest.

For every new frame, find a likelihood image ( which is similar to a density function ), whose pixels indicate how similar they are with the color distribution of the object of interest. This likelihood image can be obtained by histogram backprojection which is discussed in the next section.

Use Meanshift to find the maxima of this likelihood image cum density function, which gives the position of the object in the new frame.

Histogram backprojection is a way of finding the similarity between two images. It can be vaguely defined as a method of re-applying a pre-calculated histogram to a new image to find the similarity between the color distribution of the new image and the object of interest. Say, we have a histogram of the object of interest given by H. Then, for every pixel in the new image, it finds the bin in H that it should belong to and creates a new image with the pixel value being the value of bin count.

Step 1 : Find the histogram of the face region

First, we detect the face using Dlib’s face detector. Then compute the histogram of the face region. HSV color space is an intuitive color space as it represents color much like humans perceive it. Thus we use this to calculate the histogram of the face region. We use the Hue Channel only. However, both the H and S channels can be used. We use the calcHist() OpenCV function to compute the histogram and normalize the histogram to have values in the range [0, 255]. It should be kept in mind that color information is very sensitive to lighting variations.

Step 2 : Find Back Projected image

For every new frame, convert it to the same color space used for finding the histogram. Find the back projected image,  BP  using the calcBackProject() OpenCV function.

Step 3 : Apply Meanshift
Use meanshift to find the maxima in the back projected image in the neighborhood of the old position. We use meanShift() OpenCV function to get the new position. As explained in the earlier section, the algorithm finds the mode of the back projected image which is a confidence map of similarity between the color distribution of the object and the new image. The first few iterations for a particular frame of the meanshift process are shown in the figure below.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


filename = DATA_PATH + "videos/face1.mp4"
cap = cv2.VideoCapture(filename)

#nitialize the video feed and declare variables. Read a frame and find the face region using Dlib facial detector. Also convert the #lib rectangle to OpenCV rect.

# Read a frame and find the face region using dlib
ret,frame = cap.read()

# Detect faces in the image
faceCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_frontalface_default.xml')

frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
faces = faceCascade.detectMultiScale(frameGray,1.3,5)
x,y,w,h = faces[0]

currWindow = (x,y,w,h)


#Get the face region and convert to HSV color space. Use the inRange function to get rid of spurious noise and create a mask which will #be used for computing the histogram.

# get the face region from the frame
roiObject = frame[y:y+h,x:x+w]

hsvObject =  cv2.cvtColor(roiObject, cv2.COLOR_BGR2HSV)

# Get the mask for calculating histogram of the object and 
# also remove noise
mask = cv2.inRange(hsvObject, np.array((0., 50., 50.)), 
                  np.array((180.,255.,255.)))

plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
plt.title("Mask of ROI")
plt.imshow(mask)
plt.subplot(1,2,2)
plt.title("ROI")
plt.imshow(roiObject[:,:,::-1])
plt.show()


#We use 180 bins for each hue value. Use calcHist function to compute the histogram. We also normalize the histogram values to lie #between and 255.

# Find the histogram and normalize it to have values 
# between 0 to 255
histObject = cv2.calcHist([hsvObject], [0],
                        mask, [180], [0,180])           
cv2.normalize(histObject, histObject, 0, 
              255, cv2.NORM_MINMAX);
              
# Setup the termination criteria, either 10 iterations or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# We will process only first 5 frames
count = 0
while(1):
    ret , frame = cap.read()
    if ret == True:
        # Convert to hsv color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # find the back projected image with the histogram obtained earlier
        backProjectImage = cv2.calcBackProject([hsv], [0], histObject, [0,180], 1)

        # Compute the new window using mean shift in the present frame
        ret, currWindow = cv2.meanShift(backProjectImage, currWindow, term_crit)

        # Display the frame with the tracked location of face
        x,y,w,h = currWindow
        frameClone = frame.copy()

        if count % 20 == 0:
            plt.figure(figsize=(12,12))
            plt.subplot(1,2,1)
            plt.imshow(backProjectImage)
            plt.title("Back Projected Image")
            cv2.rectangle(frameClone, (x,y), (x+w,y+h), (255,0,0), 2, cv2.LINE_AA)
            plt.subplot(1,2,2)
            plt.imshow(frameClone[:,:,::-1])
            plt.title('Mean Shift Object Tracking Demo')
            plt.show()
    else:
        break
    count += 1
    if count > 100:
        break 
        
cap.release()
```

### CamShift
```
Camshift() finds an object center using meanshift and then adjusts the window size. In addition, it finds the optimal rotation of the object. The function returns the rotated rectangle structure that includes the object position, size, and orientation. In the figure shown below, the blue rectangle shows the current window of interest and green rectangle shows the rotated rectangle.

We follow the steps 1 and 2 used for Object tracking with Meanshift. For the third step, after we have found the back projected image, we use CamShift() OpenCV function to track the position of the object in the new image

#Initialize the video feed and declare variables. Read a frame and find the face region using Dlib facial detector. Also convert the #dlib rectangle to OpenCV rect.

filename = DATA_PATH + "videos/face1.mp4"
cap = cv2.VideoCapture(filename)

# Read a frame and find the face region using dlib
ret,frame = cap.read()

# Detect faces in the image
faceCascade = cv2.CascadeClassifier(DATA_PATH + 'models/haarcascade_frontalface_default.xml')

frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
faces = faceCascade.detectMultiScale(frameGray,1.3,5)
x,y,w,h = faces[0]
currWindow = (x,y,w,h)

#Get the face region and convert to HSV color space. Use the inRange function to get rid of spurious noise and create a mask which will #be used for computing the histogram.

# get the face region from the frame
roiObject = frame[y:y+h,x:x+w]
face_height,face_width = roiObject.shape[:2]

hsvObject =  cv2.cvtColor(roiObject, cv2.COLOR_BGR2HSV)

# Get the mask for calculating histogram of the object and also remove noise
mask = cv2.inRange(hsvObject, np.array((0., 50., 50.)), np.array((180.,255.,255.)))

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(mask)
plt.title("Mask")
plt.subplot(1,2,2)
plt.imshow(roiObject[:,:,::-1])
plt.title("Object")
plt.show()

We use 180 bins for each hue value. Use calcHist function to compute the histogram. We also normalize the histogram values to lie between and 255.

# Find the histogram and normalize it to have values between 
# 0 to 255
histObject = cv2.calcHist([hsvObject], [0],
                         mask, [180], [0,180])           
cv2.normalize(histObject, histObject,
             0, 255, cv2.NORM_MINMAX);
             
#Read a frame and convert to HSV color space and find the back projected image using the histogram calculated earlier.

# Setup the termination criteria, either 10 iterations or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
# We will display only first 5 frames
count = 0
i=0
while(1):
    ret, frame = cap.read()
    if ret == True:
        # Convert to hsv color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # find the back projected image with the histogram obtained earlier
        backProjectImage = cv2.calcBackProject([hsv], [0], histObject, [0,180], 1)



        # Compute the new window using CAM shift in the present frame
        rotatedWindow, currWindow = cv2.CamShift(backProjectImage, currWindow, term_crit)

        # Get the window used by mean shift
        x,y,w,h = currWindow

        # Get the rotatedWindow vertices
        rotatedWindow = cv2.boxPoints(rotatedWindow)
        rotatedWindow = np.int0(rotatedWindow)
        frameClone = frame.copy()

        # Display the current window used for mean shift
        cv2.rectangle(frameClone, (x,y), (x+w,y+h), (255, 0, 0), 2, cv2.LINE_AA)

        # Display the rotated rectangle with the orientation information
        frameClone = cv2.polylines(frameClone, [rotatedWindow], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frameClone, "{},{},{},{}".format(x,y,w,h), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frameClone, "{}".format(rotatedWindow), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)

        if count % 20 == 0:
            plt.figure(figsize=(12,12))
            plt.subplot(1,2,1)
            plt.imshow(backProjectImage)
            plt.title("Back Projected Image")
            plt.subplot(1,2,2)
            plt.imshow(frameClone[:,:,::-1])
            plt.title('CAM Shift Object Tracking Demo')
            plt.show()

        i+=1
    else:
        break
    count += 1
    if count > 100:
        break
        
        
cap.release()

```
              










