# All Conv-Net for Bird Activity Detection Significance of Learned Pooling
Bird activity detection (BAD) deals with the task of predicting the presence or absence of bird vocalizations in a given audio recording. In this work, we propose an all-convolutional neural network (all-conv net) for bird activity detection. All the layers of this network including pooling and dense layers are implemented using convolution operations. The pooling operation implemented by convolution is termed as learned pooling. This learned pooling takes into account the inter featuremap correlations which are ignored in traditional max-pooling. This helps in learning a pooling function which aggregates the complementary information in various feature maps, leading to better bird activity detection. Experimental observations confirm this hypothesis. The performance of the proposed all-conv net is evaluated on the BAD Challenge 2017 dataset. The proposed all-conv net achieves state-of-art performance with a simple architecture and does not employ any data pre-processing or data augmentation techniques. This work is accepted for publication in INTERSPEECH 2018. 

     feature_extract.py is used to extract melspectrogram features.
     all_convnet_BAD.py is the all convolutional model architecture
     all_convnet_BAD_maxpool_variant.py is the maxpool variant of the all_conv_net model
     get_activation_map.py is used to analyse any layer activations in the model




more about the challenge and the data can be downloaded at here http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/
