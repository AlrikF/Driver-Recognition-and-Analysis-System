# Driver-Recognition-and-Analysis-System
Created a recognition system that can utilize low quality images for continuous monintoring and authorization of drivers. This includes models such as active learning module that utilizes entropy to indicate which images to be annotated for the highest information gain by the oracle resulting in a higher accuracy using lower number of images. The blur detection module utilizes 2 approaches one which makes use of DCT transformation for localizing blur and the other utilizeas a Mask RCNN model to model the blur map.
The open set recognition helps reduce the amount of false positive by utilizing atypical samples to model the unknown class during training that helps better represent the distribution of data 