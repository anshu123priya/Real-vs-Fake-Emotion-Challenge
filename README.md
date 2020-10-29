# Real-v-s-Fake-Emotion-Challenge

---
This project is currently in progress under the guidance of Dr. Himanshu Kumar

<!-- Put the link to this slide here so people can follow -->


### Problem statement
#### Task1 : To design a algorithm which can detect the **basic 7 emotions**:-
1. Happiness
2. Surprise
3. Anger
4. Sadness
5. Fear
6. Disgust
7. Neutral

![](https://i.imgur.com/KC8nj3c.png)
*Figure from paper Facial Expression Recognition Based on Facial Components Detection and HOG Features*

 #### 2. To design a algorithm which can detect the **Real v/s Fake Expression**:-
 It majorly focused on the recognition of fakeness and trueness for 6 basic emotions
 
 ![](https://i.imgur.com/iBrFwZp.png)
 *Figure from Fake Vs. True facial emotion recognition competition (ICCV'17)*
  

  

---

### Dataset

#### Task1:
**Dataset is taken from the Facial Expression Recognition Challenge of Kaggle**
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=train.csv

The data consists of 48x48 pixel grayscale images of faces.
The training set consists of 28,709 examples. Train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image.


#### Task2:
**Dataset is taken from the Fake Vs. True facial emotion recognition challenege website of ICCV'17**
http://chalearnlap.cvc.uab.es/dataset/27/data/47/description/For (request for access sent)

Dataset contains training, validation and test sets of **480, 60 and 60** RGB videos.The whole dataset contains videos of **50 different subjects**. For each subject, there are 12 videos about 3-5 seconds long representing 6 basic emotions (Anger, Happiness, Sadness, Disgust, Contempt,Surprise) for **real and fake expressions**.

A summary of the database is as follows:
![](https://i.imgur.com/Wipftxv.png)
![](https://i.imgur.com/gAGYZHU.png)


---
### Literature Survey


#### Task1:
***1. Deep Facial Expression Recognition: A Survey***
https://arxiv.org/pdf/1804.08348.pdf
* Provide a comprehensive survey on deep FER, including datasets and algorithms
* Provide the information about the available datasets that are widely used in the literature and provide accepted data selection and evaluation principles for these datasets.
* Describe the standard pipeline of a deep FER system with the related background knowledge and suggestions of applicable implementations for each stage. 




![](https://i.imgur.com/BByLMRm.png)
*The general pipeline of deep facial expression recognition systems.*

***2. MicroExpNet: An Extremely Small and Fast Model for Expression Recognition From Face Images***
https://arxiv.org/pdf/1711.07011v4.pdf
* This paper is aimed at creating extremely small and fast convolutional neural networks (CNN) for the problem of facial expression recognition (FER)
* First train a large model on two widely used benchmarks FER datasets, CK+ and Oulu-Casia, by using the Inception v3 model. 
* Then using the “knowledge distillation” (KD) method, created family of small and fast models. 
* In the KD method, there is a large, cumbersome model called the teacher (Inception v3 here) and a relatively much smaller model called the student. 
* The student is trained to “mimic” the softmax values of the teacher via a temperature hyperparameter 

Cons of paper:
1) A ﬁne-grained grid search is needed for tuning the temperature hyperparameter
2) To ﬁnd the optimal size-accuracy balance, one needs to search for the ﬁnal network size (or the compression rate).


***3. Real-time Convolutional Neural Networks for Emotion and Gender Classification***
https://arxiv.org/pdf/1710.07557.pdf
* Propose and implement a general convolutional neural network (CNN) building framework for designing real-time CNNs. 
* Eliminated completely the fully connected layers and by reducing the amount of parameters in the remaining convolutional layers via depth-wise separable convolutions
* Developed a vision system that performs face detection, gender classiﬁcation and emotion classiﬁcation in a single integrated module.
* Achieved human-level performance in classiﬁcations tasks using a single CNN that leverages modern architecture, report accuracies of 96% in the IMDB gender dataset and 66% in the FER-2013 emotion dataset. Along with this we
* Also presented a visualization of the learned features in the CNN using the guided back-propagation visualization. This visualization technique is able to show us the high-level features learned by models and discuss their interpretability.

![](https://i.imgur.com/aXwr6lK.png)
*proposed model for real-time classiﬁcation*

![](https://i.imgur.com/sHvS4Kb.png)
*Results of the provided real-time emotion classiﬁcation*



***4.Training an emotion detector with transfer learning (blog post on towards data science)***
https://towardsdatascience.com/training-an-emotion-detector-with-transfer-learning-91dea84adeed
* An easy way to train a emotion detector using pre-trained computer vision models, transfer learning
* Also provede a way to create a our own custom dataset using Google Images.
* Pipeline of this work is 
1.scrape a dataset from Google Images and leverage the queries to label the images
2.Apply and fine-tune pretrained models to detect faces and serve as the starting point for the model
3.Use the trained model to detect and identify emotions in images and videos

![](https://i.imgur.com/5d0cezX.png)
*crops the face from the raw image using an pre-trained face detection model and resizes the image and transforms it to grayscale*

---
#### Task2:

*Note: Most of the existing papers for this task was the one provided by Fake Vs. True facial emotion recognition challenege of ICCV'17*

**1. Real vs. Fake Emotion Challenge: Learning to Rank Authenticity From Facial Activity Descriptors**
http://chalearnlap.cvc.uab.es/challenge/25/track/25/description/
* The method proposed of three steps. Firstly the estimate the intensity of facial action units (AU). 
* For each video frame the method applies face detection, facial landmark localization, face registration, Local Binary pattern (LBP) feature extraction, and finally predicts AU intensities with Support Vector Regression (SVR) ensembles.
* Next they condense the obtained time series to descriptors. The time series are smoothed with first order Butterworth filter. After that the second derivative is calculated and from repeatedly smoothed time series 17 statistics are extracted. In total a 440-dimensional feature space are obtained on this stage.
* Finally classify the videos with Rank-SVM  For a pair of videos the Rank-SVM decides which of the videos shows a more real emotion than the other one.
* The advantage of the proposed method is that the number of model parameters to optimize during training is very low in compared to e.g standard deep learning methods. The time needed for all stages including face detection, features extraction, training and predicting labels for test set is around 3.5 hours

![](https://i.imgur.com/JM4Njfo.png)
*Overview of above method*

**2. Discrimination between genuine versus fake emotion using long-short term
memory with parametric bias and facial landmarks**
https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Huynh_Discrimination_Between_Genuine_ICCV_2017_paper.pdf

* Firstly facial landmarks from each frame were extracted using the DLIB library. 
* Next Trained a LSTM-PB network for each emotion class. The LSTM-PB network is a modification of network, where the Recurrent Neural Network (RNN) is replaced with Long short-term memory (LSTM). 
* For learning a two-stage training procedure was used: finding the optimal weights of LSTM-PB network by a back-propagation algorithm, and learning of the optimal values of parametric bias by accumulating gradients of the previous stage. 
* Gradient boosting is used to train a Real/Fake discrimination in parametric bias space. 
 ![](https://i.imgur.com/8vhHIY2.png)
 
 
 
 **3. Relaxed Spatio-Temporal Deep Feature Aggregation for Real-Fake Expression Prediction**
https://arxiv.org/pdf/1708.07335.pdf
* The algorithm is build on the assumption, that brief emotional changes in eyes and mount movements can be distinct indicators for real/fake emotions recognition.
* The proposed method contains two stages: features extraction and classification.
* On the first stage the robust micro-emotional visual descriptors for each emotion type is obtained. 
* To compute descriptors from small temporal windows (i.e. 150 ms) of the videos, the authors used the robust video representation method with the long short-term memory model.
* For emotion detection high-level convolutional features were used. To obtain one global representation for each video, the computed descriptors were pooled with Compact Bilinear Pooling (CBP). 
* Finally a SVM classifier was applied to get final predictions.
* One of the highest contributions of this method is the novel video representation method, which can boost visual pooling by partially retaining sequential information in the representation. 
 ![](https://i.imgur.com/Bfq5Fmf.png)
*Flow of relaxed spatio-temporal feature aggregation method. Initially, frame-level deep features are extracted and aggregated
from multiple-frames (red). Later, temporal structure is captured by RNN*







---
### Further work Details
#### Task3 -  Dataset Download + Run open source code + Reach Baseline accuracy
1. Download both the dataset (task 2 required acces) and do all the preprocessing.
2. Run the open source code of some of the best papers or projects that we got from literature survey.
3. Reach the baseline accuracy as mentioned in their papers.




---
---

