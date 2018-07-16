import os
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import librosa
import scipy.signal



data_path_wav = '/home/arjun/decase_challenge/bad/warblr_data'
class_file = np.loadtxt('/home/arjun/decase_challenge/bad/wblr_data.txt',dtype='str')

class_names = class_file[:,0]
class_labels = class_file[:,1]

cls_name = []

for i in range(len(class_names)): 
   cls_name.append(class_names[i]+".wav")


feature_file = []
label_file = []

for i in range(len(cls_name)):
   [fs, x] = wavfile.read(os.path.join(data_path_wav,cls_name[i]))
   D = librosa.stft(x,1024,882,882,scipy.signal.hamming)
   D = np.abs(D)**2
   S = librosa.feature.melspectrogram(S=D,n_mels=40)
   S=librosa.power_to_db(S,ref=np.max)
#   print S.shape
   normS = S-np.amin(S)
   normS = normS/float(np.amax(normS))
   if int(normS.shape[1]) < 500:
#       print cls_name[i]
#       print normS.shape
       z_pad = np.zeros((40,500))
       z_pad[:,:-(500-normS.shape[1])] = normS
#       print z_pad.shape
#       print normS
#       print z_pad
       feature_file.append(z_pad)
       label_file.append(class_labels[i])
   else:
       img = normS[:,np.r_[0:500]]
       feature_file.append(img)
       label_file.append(class_labels[i])



feature_file = np.array(feature_file)
label_file = np.array(label_file)
feature_file = np.reshape(feature_file,(len(cls_name),40,500,1))
np.save('dbad_wblr_feature',feature_file)
np.save('dbad_wblr_label',label_file)
