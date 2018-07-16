import os
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import librosa
import scipy.signal

#data_path_&_class_file
data_path = '../warblr_data' # data in wav format
class_file = np.loadtxt('../wblr_data.txt',dtype='str')

#class_name_&_class_label
class_names = class_file[:,0]
class_labels = class_file[:,1]

#array_to_store_feature_&_label
feature_file = []
label_file = []

def melspectrogram_feature_extract(path,class_names,class_labels):
   for i in range(len(class_names)):
      [fs, x] = wavfile.read(os.path.join(data_path,class_names[i])+".wav") # read wav file (fs = 44.1 kHz)
      D = librosa.stft(x,882*2,882,882*2,scipy.signal.hamming) # STFT computation (fft_points = 882*2, overlap= 50%, analysis_window=40ms)
      D = np.abs(D)**2 # magnitude spectra
      S = librosa.feature.melspectrogram(S=D,n_mels=40) # mel bands (40)
      S=librosa.power_to_db(S,ref=np.max)
      normS = S-np.amin(S) # normalization
      normS = normS/float(np.amax(normS))
      if int(normS.shape[1]) < 500: # 10 sec samples gives 500 frames
          z_pad = np.zeros((40,500))
          z_pad[:,:-(500-normS.shape[1])] = normS
          feature_file.append(z_pad)
          label_file.append(class_labels[i])
      else:
          img = normS[:,np.r_[0:500]] # final_shape = 40*500
          feature_file.append(img)
          label_file.append(class_labels[i])




#call_the_function
melspectrogram_feature_extract(data_path,class_names,class_labels)

feature_file = np.array(feature_file)
label_file = np.array(label_file)
feature_file = np.reshape(feature_file,(len(class_names),40,500,1))
np.save('BAD_wblr_feature',feature_file)
np.save('BAD_wblr_label',label_file)

#### similarly, using this function extract melspec features for other classes and concatenate all the features  ########
