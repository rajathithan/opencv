# OpenCV
OpenCV respository


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

| **Operation** | **Input-1** | **Input-2** | **Output** |
| --- | --- | --- | --- |
| **AND** | 0 | 0 | 0 |
| **AND** | 0 | 1 | 0 |
| **AND** | 1 | 0 | 0 |
| **AND** | 1 | 1 | 1 |
| **OR** | 0 | 0 | 0 |
| **OR** | 0 | 1 | 1 |
| **OR** | 1 | 0 | 1 |
| **OR** | 1 | 1 | 1 |
| **NOT** | 0 | NA | 1 |
| **NOT** | 1 | NA | 0 |
| **XOR** | 0 | 0 | 0 |
| **XOR** | 0 | 1 | 1 |
| **XOR** | 1 | 0 | 1 |
| **XOR** | 1 | 1 | 0 |

They will be highly useful while extracting any part of the image , defining and working with non-rectangular ROI etc.


```















