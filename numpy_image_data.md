---
---
# NumPy Image Data Conventions

## Image Data Axes

NumPy and other Python based frameworks - like TensorFlow - define image data as arrays of pixels.

This means, that an image is represented by a 3-dimensional array with:
- axis 0 representing the image height,
- axis 1 representing the image width,
- axis 2 representing the pixel depth (number of color channels)

Following this convention, the Python data shape of an image looks like this:
- (height, width, depth)

## References

- [Image axes in TensorFlow](https://www.tensorflow.org/api_docs/python/tf/io/decode_image)
- [Unfortunately, PyTorch defines image axes in a different way](https://discuss.pytorch.org/t/dimensions-of-an-input-image/19439/2)
- [Python Pillow - Image Module](https://pillow.readthedocs.io/en/latest/reference/Image.html)

## Reading an Image File into a NumPy Array

```python
from PIL import Image
import numpy as np

img = Image.open("test.png")
data = np.array(img)
```

## Reading all Image Files from a Zip Package

```python
from zipfile import ZipFile
from PIL import Image
import numpy as np

zf = ZipFile("test.zip")

data = []
for zfi in zf.infolist():
    fi = zf.open(zfi)
    img = Image.open(fi)
    data.append(np.array(img))
```

