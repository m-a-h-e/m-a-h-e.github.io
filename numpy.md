---
add-to-toc: true
toc-categories: [NumPy Basics]
---
# NumPy Image Data Conventions

## Image Data Axes

NumPy and other Python based frameworks - like TensorFlow - define image data as arrays of pixels.

This means, that an image is represented by a 3-dimensional array with:
- axis 0 representing the image height,
- axis 1 representing the image width,
- axis 2 representing the pixel depth (number of color channels)

Following this convention, the shape of an image looks like that:
- (height, width, depth)

## Reading an Image File into a NumPy Array

```python
from PIL import Image
import numpy as np

img = Image.open(f)
data = np.array(img)
```

## Reading all Image Files from a Zip package

```python
from zipfile import ZipFile
from PIL import Image
import numpy as np

zf = ZipFile("test.zip")

data = []
for file_name in zf.infolist():
    f = zf.open(file_name)
    img = Image.open(f)
    data.append(np.array(img))
```
