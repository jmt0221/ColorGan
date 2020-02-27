# Coloring Photos With a Generative Adversarial Network

This Generative Network created in Python's Keras allows us to accurately transform photos from black and white to color. 



### Required Libraries
- Numpy
- PIL
- Keras
- Scikit-Image
- Matplotlib
- Glob
- Streamlit

# Datasets
I used two different datasets for this project, the first is a collection of beach and coastal pictures from MIT's Computational Visual Library Dataset. I chose this library since GAN's do a much better job at understanding symmetry and texture in photos than it is at understanding complicated geometrics. The coastal photos are very symmtrical and have simple textures that the GAN can replicate easier compared to the second dataset, the MPII Human Pose Dataset. This dataset consists of people performing actions in various enviorments which means the GAN will have a harder time understanding the features and boudaries in the photo. This means that it will be much more common to see some blurring or color bleeding effects during training as compared to the coastal photos. 

# Converting from RGB to LAB

The first step was converting the images from their standard RGB color channels into CIE-LAB where the 3 new channels consists of:
- L - Represents the white to black trade off of the pixels
- A - Represents the red to green trade off of the pixels
- B - Represents the blue to yellow trade off of the pixels

<img src="https://upload.wikimedia.org/wikipedia/commons/7/7d/CIELAB_color_space_front_view.png" width="600" height="300">

The L channel is going to be used as the input to the Generator since it is equivalent to a gray scale photo. The generator will then output both the AB channels which we combine with the input to create a colored image.


# Results
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/woman_together.png" width="600" height="300">
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/golf_together.png" width="600" height="300">

<pre>
     Left: Original Image                     Right: Fake Image
</pre>
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/kitchen_combined.png" width="600" height="300">
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/magic_combined.png" width="600" height="300">
