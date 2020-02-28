# Coloring Photos With a Generative Adversarial Network

This Generative Network created in Python's Keras allows us to accurately transform photos from black and white to color. 

Here is a blog covering the project in further detail: https://towardsdatascience.com/coloring-photos-with-a-generative-adversarial-network-a435c4403b5d

### Required Libraries
I ran this model in an AWS SageMaker GPU instance to handle the high computational requirements. I have included the code and imports needed if you want to run it in the same instance. Training time varied from 6-12 hours depending on the dataset and learning rate.
- Numpy
- PIL
- Keras
- Scikit-Image
- Matplotlib
- Glob
- Streamlit

# Datasets
I used two different datasets for this project; the first is a collection of beach and coastal pictures from MIT's Computational Visual Library Dataset. I chose this library since GAN's do a much better job of understanding symmetry and texture in photos than it is at understanding complicated geometrics. The coastal photos are very symmetrical and have simple textures that the GAN can replicate easier compared to the second dataset, the MPII Human Pose Dataset. This dataset consists of people performing actions in various environments which means the GAN will have a harder time understanding the features and boundaries in the photo. This means that it will be much more common to see some blurring or color bleeding effects during training as compared to the coastal photos. The MPII dataset varies in size so we are going to reshape them all to 256px256p which is the default size for the coastal images.

# Converting from RGB to LAB

The first step was converting the images from their standard RGB color channels into CIE-LAB where the 3 new channels consist of:
- L - Represents the white to black trade off of the pixels
- A - Represents the red to green trade off of the pixels
- B - Represents the blue to yellow trade off of the pixels
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/7/7d/CIELAB_color_space_front_view.png" width="300" height="300">
</p>
The L channel is going to be used as the input to the Generator since it is equivalent to a grayscale photo. The generator will then output both the AB channels which we combine with the input to create a colored image. The last thing we do in the transformation is scale all pixel values in the LAB color channels to the range -1 to 1.


# The Model

## Generator

The Generator can be broken down into two parts, the encoder and the decoder. The encoder consists of 4 layers in total; first, the L channel is passed through a 2-stride convolutional layer, starting with 64 feature maps and doubling each layer up to 512. The layer also halves the size of the image so by the time we reach the latent space (middle of the network) the feature maps are 16x16 in size. Lastly, each layer also uses a Leaky ReLU activation function and batch normalization.

<p align="center">
<img src="https://miro.medium.com/max/3636/0*7fgHtc8fEmoC_SiZ.png" width="600" height="300">
</p>

The Decoder is similar except it uses transpose convolutional layers which upsamples the size of the image and halves the number of feature maps. Like with the encoder, there are four layers and they all use the ReLU activation function and batch normalization. This feeds into the final layer which uses a normal convolutional layer which outputs 2 channels that we pass through a tanh activation function. The tanh function is used to replicate the range we scaled the images to when converting to LAB.

## Discriminator

The Discriminator is a much simpler model and has four 2-stride convolutional layers which all use Leaky ReLU, dropout, and batch normalization (except for the first layer). It takes in both real and fake AB color channels which are fed through those convolutional layers, and finally into an output layer. The output layer uses the sigmoid activation function to predict the probability of an image being real or fake. 

# Training

Both models use Binary Crossentropy for their loss functions and the Adam optimization function. Contrary to all the research I have found online, I had to use a lower learning rate for the discriminator to keep the model in equilibrium. The biggest difference between a GAN and most machine learning models is that the goal of a GAN isn't to minimize or maximize any function; instead, a GAN will keep the two competing networks in equillibrium during training, which can last anywhere from 6-12 hours on a GPU. During training, I shuffle my dataset and use mini epochs of 320 images. I train the discriminator first on two half batches of 160 real and 160 fake images, and then the generator is trained a full batch of 320. I would usually run this for about 3000 epochs, but further training produced better results occasionally.

<p align="center">
<img src="https://miro.medium.com/max/3232/1*siZC9SZPLHpr9C0ofoajpA.png" width="800" height="300">
</p>

As you can see from the graph, the training cycle can be highly erratic, making it difficult to perceive how well the model is actually doing. The best way to keep track of its progress is to print a fake image every 50-100 epochs and save the model about half as often.


# Results

These images were not part of either dataset and were images taken from the internet. I converted them to the black and white L channel and create the A and B channels via the generator. Finally, I concatenate it back together and print the images you see below. This allows me to recolor both black and white as well as colored images.

<pre>
     Left: Original Image                     Right: Fake Image
</pre>

<p align="center">
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/woman_together.png" width="600" height="300">
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/golf_together.png" width="600" height="300">
</p>



<p align="center">
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/kitchen_combined.png" width="600" height="300">
<img src="https://github.com/jmt0221/ColorGan/blob/master/images/magic_combined.png" width="600" height="300">
</p>
