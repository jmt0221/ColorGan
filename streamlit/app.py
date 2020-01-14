import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, UpSampling2D, Dropout, Flatten, Dense, Input, LeakyReLU, Conv2DTranspose,AveragePooling2D, Concatenate
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.compat.v1 import set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
from io import BytesIO
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

@st.cache
def model_load(dataset='people2'):
    '''
    Loads the model depending on which dataset we are working on
    '''
    if dataset == 'people1':
        model = load_model('models/generator_people_v1.h5')
    if dataset == 'people2':
        model = load_model('models/generator_people_v3.h5')
    elif dataset == 'coast':
        model = load_model('models/generator_v1.h5')
    return model

@st.cache
def read_img(file, size = (256,256)):
    '''
    reads the images and transforms them to the desired size
    '''
    img = image.load_img(file, target_size=size)
    img = image.img_to_array(img)
    return img


def read_img_url(url, size = (256,256)):
    """
    Read and resize image directly from a url
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((256, 256))
    img = image.img_to_array(img)
    return img

@st.cache
def read_multiple_images(im,dataset='people2'):
    '''
    Read and transforms an image then displays 
    '''
    img = read_img(im).astype('int64')
    l_channel = rgb_to_lab(img,l=True)
    model = model_load(dataset)
    fake_ab = model.predict(l_channel.reshape(1,256,256,1))
    fake = np.dstack((l_channel,fake_ab.reshape(256,256,2)))
    fake = lab_to_rgb(fake).astype('int64')
    multi = np.vstack((img,fake))
    return multi.reshape(2,256,256,3)






def rgb_to_lab(img, l=False, ab=False):
    """
    Takes in RGB channels in range 0-255 and outputs L or AB channels in range -1 to 1
    """
    img = img / 255
    l_chan = color.rgb2lab(img)[:,:,0]
    l_chan = l_chan / 50 - 1
    l_chan = l_chan[...,np.newaxis]

    ab_chan = color.rgb2lab(img)[:,:,1:]
    ab_chan = (ab_chan + 128) / 255 * 2 - 1
    if l:
        return l_chan
    else: 
    	return ab_chan

def lab_to_rgb(img):
    """
    Takes in LAB channels in range -1 to 1 and out puts RGB chanels in range 0-255
    """
    new_img = np.zeros((256,256,3))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pix = img[i,j]
            new_img[i,j] = [(pix[0] + 1) * 50,(pix[1] +1) / 2 * 255 - 128,(pix[2] +1) / 2 * 255 - 128]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img




def merge_real_fake(image,percentage,dataset):
    '''
    Transforms a photo and displays a percentage of each image merged together
    Percentage depends on slide setting
    '''
    img = read_img(image).astype('int64')
    l_channel = rgb_to_lab(img,l=True)
    model = model_load(dataset)
    fake_ab = model.predict(l_channel.reshape(1,256,256,1))
    fake = np.dstack((l_channel,fake_ab.reshape(256,256,2)))
    fake = lab_to_rgb(fake).astype('int64')
    real = (img*(1.0-percentage)).astype('int64')
    not_real = (fake*percentage).astype('int64')
    if percentage < 0.02:
        return img
    elif percentage > 0.98:
        return fake
    else:
        merged = real+not_real
        return merged

def url_generator(url,dataset='people2'):
    '''
    downloads the image from the url and creates the color channgels, then returns original and created
    '''
    img = read_img_url(url,size=(256,256)).astype('int64')
    l_channel = rgb_to_lab(img,l=True)
    model = model_load(dataset)
    fake_ab = model.predict(l_channel.reshape(1,256,256,1))
    fake = np.dstack((l_channel,fake_ab.reshape(256,256,2)))
    fake = lab_to_rgb(fake).astype('int64')
    return img, fake






def main():
    '''
    The main code for Streamlit
    '''
    #allows us to Pick what page we want to be on
    st.sidebar.title("What do you want to do?")
    app_mode = st.sidebar.selectbox("Choose the mode", [ "Best Image Transformation", "From Real to Fake", "Color Your Own Image","Model Architecture" ])

    if app_mode == "Best Image Transformation":
        #Displays the best image transformations on the first page
        with st.spinner("Showing you the Best..."):
            st.title("Coloring Images with a Generative Adversarial Network")
            st.markdown('Here are some of the best results: Original Left, Generated Right')
            images = read_multiple_images('images/sun27.jpg',dataset='coast')
            st.image(images,width=325)
            images2 = read_multiple_images('images/sun32.jpg',dataset='coast')
            st.image(images2,width=325)
            images3 = read_multiple_images('images/natu131.jpg',dataset='coast')
            st.image(images3,width=325)
            images4 = read_multiple_images('images/cdmc948.jpg',dataset='coast')
            st.image(images4,width=325)
            images5 = read_multiple_images('images/ski.jpg',dataset='people2')
            st.image(images5,width=325)
            images6 = read_multiple_images('images/magic.jpg',dataset='people1')
            st.image(images6,width=325)
            images7 = read_multiple_images('images/mbs.jpg',dataset='coast')
            st.image(images7,width=325)
            images8 = read_multiple_images('images/kitchen.jpg',dataset='people2')
            st.image(images8,width=325)
            images9 = read_multiple_images('images/bad_nyc.jpg',dataset='people1')
            st.image(images9,width=325)
            

    
    elif app_mode == "From Real to Fake":
        #merges together the original and generated photo at any percentage
        with st.spinner("Merging Photos..."):
            st.title("Seeing images transform from real to fake")
            st.markdown("Percentage of Original and Generated")
            percentage = st.slider("0 = Original 1 = Generated", 0.0, .99, (0.0))
            filter_by = st.radio('What image do you want to look at', ['Black and White Caricature','Black and White Photo','Black and White Golf','Coast','Motorcross Off Color','Beach Off Color'])
            if filter_by == 'Black and White Caricature':
                merged = merge_real_fake('images/bw1.png',percentage,dataset='people1')
                st.image(merged,width=325)
            elif filter_by == 'Black and White Photo':
                merged = merge_real_fake('images/b_w2.jpg',percentage,dataset='people2')
                st.image(merged,width=325)
            elif filter_by == 'Black and White Golf':
                merged = merge_real_fake('images/golf.jpg',percentage,dataset='people2')
                st.image(merged,width=325)
            elif filter_by == 'Coast':
                merged = merge_real_fake('images/b_w3.jpeg',percentage,dataset='coast')
                st.image(merged,width=325)
            elif filter_by == 'Beach Off Color':
                merged = merge_real_fake('images/natu912.jpg',percentage,dataset='coast')
                st.image(merged,width=325)
            elif filter_by == 'Motorcross Off Color':
                merged = merge_real_fake('images/motorcross.jpg',percentage,dataset='people2')
                st.image(merged,width=325)

    elif app_mode == "Color Your Own Image":
        #grab any photo's url and color it 
        with st.spinner("Coloring..."):
            st.title("Pick any image and lets color it in!")
            link = st.text_input('Put Image URL Here', 'Type Here')
            search = st.radio('What do we want to color:', ['coast', 'people1','people2'])
            if link != 'Type Here':
                real,fake = url_generator(link,dataset=search)
                st.image(real,width=325)
                st.image(fake,width=325)

                
    elif app_mode == "Model Architecture":
        #displays the architecture of the models
        with st.spinner('Gathering the data...'):
            st.title("Architecture of My Networks")
            st.markdown("Let me introduce you to the model and its architecture")
            st.image('combined_network.jpg')




if __name__ == "__main__":
    main()


