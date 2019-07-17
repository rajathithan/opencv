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





