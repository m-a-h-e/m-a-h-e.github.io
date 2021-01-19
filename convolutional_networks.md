---
tags: python,numpy,neural-network,convolution,convolutional networks
mathjax: true
---
# Convolutional Neural Networks (CNNs)

{:.caption .img}
[![Convolutional Neural Networks, MIT 6.S191](https://img.youtube.com/vi/iaSUYvmCekI/0.jpg)](https://www.youtube.com/watch?v=iaSUYvmCekI)
[Alexander Amini](https://www.mit.edu/~amini/) - Convolutional Neural Networks, MIT 6.S191 (2020)

{:.caption .img}
[![Convolutional Neural Networks, Stanford University](https://img.youtube.com/vi/bNb2fEVKeEo/0.jpg)](https://www.youtube.com/watch?v=bNb2fEVKeEo)
[Serena Yeung](https://ai.stanford.edu/~syyeung/) - Convolutional Neural Networks, Stanford University (2017)

{:.caption .img}
[![CNN Architectures, Stanford University](https://img.youtube.com/vi/DAOcjicFr1Y/0.jpg)](https://www.youtube.com/watch?v=DAOcjicFr1Y)
[Serena Yeung](https://ai.stanford.edu/~syyeung/) - CNN Architectures, Stanford University (2017)

## CNN Implementation

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Conv2D(shape_in=(8, 8, 1), shape_out=(6, 6, 4), kernel_size=3, stride=1),
  npnn.LeakyReLU(6 * 6 * 4),

  npnn.MaxPool(shape_in=(6, 6, 4), shape_out=(3, 3, 4), kernel_size=2),
  npnn.Pad2D(shape_in=(3, 3, 4), pad_axis0=1, pad_axis1=1),
  npnn.Conv2D(shape_in=(5, 5, 4), shape_out=(3, 3, 4), kernel_size=3, stride=1),
  npnn.LeakyReLU(3 * 3 * 4),

  npnn.AvgPool(shape_in=(3, 3, 4), shape_out=(1, 1, 4), kernel_size=3),
  npnn.Reshape(shape_in=(1, 1, 4), shape_out=(4)),
  npnn.Softmax(4)
]

loss_layer = npnn.loss_layer.CrossEntropyLoss(4)
optimizer  = npnn.optimizer.Adam(alpha=1e-2)
dataset    = npnn_datasets.FourSteppedIcons()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w90}
<div class="video">
<video controls poster="assets/videos/four_stepped_icons_6x6_double_conv.png">
  <source src="assets/videos/four_stepped_icons_6x6_double_conv.webm" type="video/webm">
  <source src="assets/videos/four_stepped_icons_6x6_double_conv.ogv" type="video/ogg">
  <source src="assets/videos/four_stepped_icons_6x6_double_conv.mp4" type="video/mp4">
</video>
<p>Four Different Icon Pattern Classification using a CNN</p>
</div>

{:.w90}
<div class="video">
<video controls poster="assets/videos/four_stepped_icons_6x6_double_conv_2.png">
  <source src="assets/videos/four_stepped_icons_6x6_double_conv_2.webm" type="video/webm">
  <source src="assets/videos/four_stepped_icons_6x6_double_conv_2.ogv" type="video/ogg">
  <source src="assets/videos/four_stepped_icons_6x6_double_conv_2.mp4" type="video/mp4">
</video>
<p>Four Different Icon Pattern Classification using a CNN<br>
plot of network validation batch data target values (green) and 
predicted network output values (orange)</p>
</div>

## References

[A Beginner's Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8)
[Convolutional Neural Networks - A Beginner's Guide](https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022)

