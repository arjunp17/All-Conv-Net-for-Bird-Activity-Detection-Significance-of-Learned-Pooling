from keras.models import Sequential, Model
import numpy as np
from keras.models import load_model
import os
from keras.optimizers import Adam
from PIL import Image

savepath = '/home/arjun/visualize_act_bird_allconv/conv2d_2'

#get_activation_maps_from_any_layer_using_the_model
base_model = load_model("all_convnet_BAD.h5")
opt = Adam(decay = 1e-6)
base_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) # define optimizer and run compile if optimizer is not loaded properly
base_model.summary()

model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_2').output) #choose the layer which you want to analyse


img = np.load('melspec.npy') # melspectrogram_feature_for_a_sample_that_we_want_to_analyse
img = np.array(img, dtype='float32')
input_img_data = np.reshape(img,(1,40,500,1))



layer_output = model.predict(input_img_data)
layer_output = np.squeeze(layer_output)

for i in range(layer_output.shape[2]):
	result = Image.fromarray((layer_output[:,:,i]*255).astype(np.uint8))
	result.save(os.path.join(savepath,str(i)+".jpg"))


