# Real-Time_Facial_Expression_Detection
## A Lightweight Convolutional Neural Network for Real-Time Facial Expression Detection

## ABSTRACT:
 In this paper our group proposes and designs a lightweight convolutional neural network (CNN) for detecting facial emotions in real-time and in bulk to achieve a better classification effect. We verify whether our model is effective by creating a real-time vision system. This system employs multi-task cascaded convolutional networks (MTCNN) to complete face detection and transmit the obtained face coordinates to the facial emotions classification model we designed firstly. Then it accomplishes the task of emotion classification. Multi-task cascaded convolutional networks have a cascade detection feature, one of which can be used alone, thereby reducing the occupation of memory resources. Our expression classification model employs Global Average Pooling to replace the fully connected layer in the traditional deep convolution neural network model. Each channel of the feature map is associated with the corresponding category, eliminating the black box characteristics of the fully connected layer to a certain extent. At the same time, our model marries the residual modules and depth-wise separable convolutions, reducing large quantities of parameters and making the model more portable. Finally, our model is tested on the FER-2013 dataset. It only takes 3.1% of the 16GB memory, that is, only 0.496GB memory is needed to complete the task of classifying facial expressions. Not only can our model be stored in an 872.9 kilobytes file, but also its accuracy has reached 67% on the FER-2013 dataset. And it has good detection and recognition effects on those figures which are out of the dataset.

## New Abstract:
A semi weighted convolutional neural network (CNN) to improve the classification result through real-time and bulk facial emotion detection. We built a real-time vision system to test the effectiveness of our approach. This system completes the goal of emotion classification by using the CNN model architecture. Through the use of facial expression recognition software, individuals can assess their intentions and emotional states by interpreting their expressions. In this project a convolutional neural network (CNN) model. The recognition of facial expressions is done using this approach. We will start by creating a CNN model and learning it the local characteristics of the lips, eyebrows, and eyes. Ultimately, we are trying to achieve the best accuracy and detection rate for this model in real time facial expression recognition. 
•	Many of the existing systems they achieved max 98% percentage of the accuracy for this model. We are targeting to improve this at least 98.5% and best classification rate in real time.
•	For this Project we are using Grayscale photos from the FER 2013 dataset 
•	The output classes should be the five groups: happy, sad, neutral, fearful, and angry. 
Below are the requirements for the developing this model.
## SYSTEM REQUIREMENTS:
## HARDWARE REQUIREMENTS: 
•	System: Pentium i5 Processor.
•	Hard Disk: 1 TB.
•	Monitor: 15’’ LED
•	Input Devices: Keyboard, Mouse
•	Ram: 16 GB
SOFTWARE REQUIREMENTS: 
•	Operating system: Windows 11.
•	Coding Language: Python with ML and DL packages.

## Implementation:-
We propose an end-to-end deep learning framework, based on attentional convolutional network, to classify the underlying emotion in the face images. Often times, improving a deep neural network relies on adding more layers/neurons, facilitating gradient flow in the network (e.g. by adding adding skip layers), or better regularizations (e.g. spectral normalization), especially for classification problems with a large number of classes. However, for facial expression recognition, due to the small number of classes, we show that using a convolutional network with less than 10 layers and attention (which is trained from scratch) is able to achieve promising results, beating state-of-the-art models in several databases. Given a face image, it is clear that not all parts of the face are important in detecting a specific emotion, and in many cases, we only need to attend to the specific regions to get a sense of the underlying emotion. Based on this observation, we add an attention mechanism, through spatial 

 ![Picture1](https://github.com/user-attachments/assets/e313eeab-d714-4b87-9bfa-980ccba32839)


transformer network into our framework to focus on important face regions.

## About Facial Recognization:-

 ![Picture2](https://github.com/user-attachments/assets/907b1695-3dfa-4bef-b4bc-a3e9aed697c7)


Description of the corpus The evaluation of the proposed method for facial expression recognition was performed on two databases: • The JAFFE database (The Japanese Female Facial Expression) [LBA99a]: is widely used in the facial expressions research community. It is composed of 213 images of 10 Japanese women displaying seven facial expressions: the six basic expressions and the neutral one. Each subject has two to four examples for each facial expression. • The KANADE database [KCT00]: is composed of 486 video sequences of people displaying 23 facial expressions within the six basic facial expressions. Each sequence begins by a neutral expression and finish with the maximum intensity of the expression. For fair comparison between KANADE and JAFFE databases, we selected from the KANADE database the first image (neutral expression) and the last three images (with the maximum intensity of the expression) of 10 people chosen randomly. Moreover, we selected the six basic facial expressions and the neutral one.
![Picture3](https://github.com/user-attachments/assets/90a9821e-eddc-44fb-83ea-8a938d54647a)

 
Facial Expression Recognition basically performed in three major steps: 
• Face detection • Feature Extraction • Facial Expression Classification 
The primary need of Face Expression Recognition system is Face Detection which is used to detect the face. The next phase is feature extraction which is used to select and extract relevant features such as eyes, eyebrow, nose and mouth from face. It is very essential that only those features should be extracted from image that have highly contribution in expression identification. The final step is facial expression classification that classifies the facial expressions based on extracted relevant features. 
There are different methods of features extraction such as appearance based method, geometric based method, texture based method etc. and in the current research mostly used methods are geometric based method and appearance based method. Geometric based feature extraction method, extract feature information using shape, distance and position of facial components and appearance based feature extraction method uses appearance information such as pixel intensity of face image. After getting the features, classification methods are applied to recognize facial expression.
![Picture4](https://github.com/user-attachments/assets/ec3bdeec-ce76-4ce6-9dd4-3c187c1d6c2c)
 
The steps of automated facial landmark detection and active patch extraction are shown fig.2 all the active facial patches is evaluated in training stage and the one features having large number variation between pairs of expressions is selected. These selected features is projected into lower dimensional subspace and then classified into different expressions using artificial neural network and support vector machine. The training phase includes pre-processing, selection of all facial patches, extraction of 
appearance features and learning of the different classifiers in the case of an unseen image, the facial landmarks are first detected, then extraction of features from the selected facial patches and finally classifies the expression.

 ![Picture5](https://github.com/user-attachments/assets/26d66f47-dd8e-4f63-9e84-038df7037b35)

Framework for automated facial landmark detection and active patch extraction, (i) face detection, (ii) coarse ROI selection for both  eyes and nose, (iii) eyes and nose detection followed by coarse ROI selection for both eyebrows and lips, (iv) detection of corners of lip and  eyebrows, (v) finding the facial landmark locations, (vi) extraction of active facial patches

Modules:
Pre-Processing 
Image pre-processing often takes in the form of signal conditioning with the segmentation, location, or tracking of the face or facial parts. A low pass filtering condition is performing using a 3x3 Gaussian mask and it remove noise in the facial images fo llowed by face detection for face localization. Viola-Jones technique of Haar-like features is used for the face detection. It has lower computational complexity and sufficiently accurate detection of near -upright and near-frontal face images. Using integral image, it can detect face scale and location in real time. The localized face images are extracted. Then localized face image scaled  to bring it to a common resolution and this made the algorithm shift invariant.
Eye and Nose Localization 
The coarse regions of interests are eyes and nose selected using geometric position of face. Coarse region of interests can also use to reduce the computational complexity and false detection. Haar classifiers used to both eyes are detected separately and then haar classifier trained for each eye. The Haar classifiers are returns to the vertices of the rectangular area of detected eyes. The  centers  of the both eyes are computed as the mean of these coordinates. The position of eyes does not change with facial expression s.  Similarly, Haar cascades are used to detected nose position. In this case eyes or nose was not detected using Haar classifier s. In  our experiment, for more than 99.6 percent cases these parts were detected correctly. 
Lip Corner Detection 
Facial topographies is used to detected of lip and both eyebrow corners. The region of interests is used to lips and both eyebrows  are selected. The face width positioned with respect to the facial organs. The region of interest was extracted mouth using the  position of nose as shown figThe upper lip produces a distinct edge can be detected using a sobel edge detector . In images  different expressions, a lot of edges are obtained it can be detect further threshold by using Otsu method . In this proce ss a  binary  image are obtained. The binary image containing lot of connected regions. Connected component analysis used to the spurious components having an area less than a threshold is removed. Then morphological dilation operation was carried out onthe binary image. Finally, the connected components are selected as the upper lip region. The different stage of the process are shown fig.
![Picture6](https://github.com/user-attachments/assets/ab3013f3-abc2-4559-b620-0b244cf2c90f)

 
Lip corner detection algorithm steps are given below.
Algorithm 1. Lip Corner Detection 
1)  Step 1. Select coarse lips ROI using face width and nose position. 
2)  Step 2. Apply 3x3 Gaussian mask to the lips ROI.
3)  Step 3. Apply horizontal sobel operator for edge detection. 
4)  Step 4. Apply Otsu-thresholding.
5)  Step 5. Apply morphological dilation operation. 
6)  Step 6. Find the connected components in images. 
7)  Step 7. Remove the spurious connected components using threshold technique to the number of pixels. 
8)  Step 8 .Scan the image from the top and select the first connected component as upper lip position. 
9)  Step 9.Locate the left and right most positions of connected component as lip corners. 
In lip corner detection shadow due sometimes below the nose, the upper lip could not be segmented properly. In this case the upper lip was not segmented as a whole portion. The first connected component obtained from the end resembled half of the upper lip. The ends of this connected component analysis did not satisfy the bilateral symmetry property, i.e., the lip corners have been at more or less equal distances from vertical central line of face. In this condition was detected and then giving a threshold ratio of the distance between the lip corners to the maximum of d istances of the lip corners to  the vertical central line. In this case the second connected component obtained below the nose it consider as the other part of upper lip. The lip corners were detected only with the help of the two connected components. Connected component analysis is used minimize false detection of lip corner and computation time. The different stage of the process are shown fig

![Picture7](https://github.com/user-attachments/assets/53ae8ca4-efd6-48f4-8ff5-5764aa221b89)

 
Eyebrow Corner Detection 
The coarse region interest is used to both eyebrows are selected. The eyebrows corner detection following the same procedure  as that of upper lip detection. An adaptive threshold operation is applying before horizontal sobel opera tor it improved the accuracy of eyebrow corner detection. The horizontal edge detector can also use to reduce the false detection of eyebrow positions due   to partial occlusion by hair. In this process a binary image are obtained. The binary image containin g lot of connected regions. Connected component analysis used to the spurious components having an area less than a threshold is removed. Then morphological dilation operation was carried out on the binary image. Finally, the connected component analysis both eyebrows are detected. Connected component analysis is used minimize false detection of eyebrow corner and inner corners are detected accurately. The different stage of the process are shown fig
 ![Picture8](https://github.com/user-attachments/assets/9436d68c-e433-4422-87b1-8500e8263a94)

Eyebrow corner detection, (a) rectangles showing coarse ROI selection for eyes and plus marks showing the detection result, (b & g)  eye ROIs, (c & h) applying adaptive threshold on ROIs, (d & i) applying horizontal sobel edge detector followed by Otsu thres hold and  morphological operations,(e & j) connected components for corner localization,(f&k) final eyebrow corners localized.

Extraction of Active Facial Patches 
Facial expressions are usually formed  by the local facial appearance variations. However, it is more difficult to automatically  localize  these local active areas on a facial  image. Firstly the facial image is divided into N local patches, and then local binary  pattern (LBP) features are used to represent the local appearance of the patch. During an expression, the local patches are extracted from the face image depending upon the position of active facial muscles.  In our experiment are used active facial patches as shown in Fig. 6.  Patches are assigned by number.  The patches do not have very fixed position on the face image. The position  of patches varying with different expression. The location depends on the positions of facial landmarks. The sizes of all facial patches are equal and it was approximately one-eighth of the width of the face. In Fig  1P,4P,18P and 17P are directly extracted from the positions of lip corners and inner eyebrows respectively. 15P was  at the center of both the eyes; and 16P was the patch occur  just above 15P. 19P and 6P are located in the midway of eye and  nose. 3P was located just below 1P.9Pwas  located at the center of position of  3P and 8P. 14P and 11P are located just below eyes. 2P,10P, and 7Pare clubbed together and located at one side of nose position. It is similar to 5P 12P, and 13P are located.

## Objectives:
An artificial intelligence technique known as a neural network teaches computers to interpret data in a manner that is inspired by the human brain. In order to simulate a layered architecture of interconnected neurons or nodes is used in the human brain by a machine learning technique known as deep learning. [10]. Frank Rosenblatt, a psychologist, created the first artificial neural network in 1958. Its name, perceptron, was chosen to represent how the human brain interpreted visual information and developed object recognition skills. Similar ANNs have since been employed to examine human cognition by other researchers. Artificial neural networks (or ANNs) are a type of neural network technology that aims to mimic the functions of the human brain in a computer. It is created with the ultimate goal of its development is to enable a computer to learn from experience much like humans do 
CNNs are among the most widely-used models currently in use. This computational model of a multilayer perceptron version is used by a neural network, and it contains one or more convolutional layers that are either fully coupled or pooled. As a result of these convolutional layers, features maps are produced, which take a portion of the image and send it for nonlinear processing after being separated into rectangles. Advantages: 
1. Extremely high accuracy in issues involving image recognition. 
2. Detects key properties automatically and without human intervention. 
3. Weight distribution 
4. Performance Measurement on CNN. 
5. Image-based data. The first sincere attempts to create a face were made in the 1980s and 1990s using a technique known as Eigenfaces. A face recognizer assumes that each face is composed of several Eigenfaces images that have been layered one pixel at a time on top of each other in order create a blur image that resembles a face. However, this approach didn't really succeed. The following generation of face recognizers would then take each image of a face and identify significant features like a mouth corner or an eyebrows. These points' coordinates are referred to as feature points. Although this method is superior to the Eigenfaces method, it is still not perfect. We waste a great deal of information that may be useful: Hair colour, eye colour, and other face structures that a feature point does not capture etc. All of this is ignored by the current generation of face recognition software. Convolutional neural networks were utilised in this technique (CNNs).
 
![Picture9](https://github.com/user-attachments/assets/c0d6684d-dc7c-4f21-942c-93125fc275b3)

## Literature Survey:
Humans have always found it simple to identify emotions from facial expressions, but applying a computer programme to do the same thing is quite difficult. The classification of facial expressions using OpenCV, Keras, and Convolutional Neural Network is the main goal of the work presented in this study. The goal of facial expression recognition software is to distinguish between fundamental human emotions including happiness, sadness, anger, surprise, and neutrality. The goal of this paper is to bring advancement and development in the field of technology [3]. This includes extraction of the feature of image which becomes easy using deep learning algorithm, and classifier model to produce output when input is given. It achieves higher accuracy as compared to traditional classifier model. It was found that the model almost predicts the emotions like happy, sad but it rarely predicts disgust emotion. Model gives the highest accuracy while predicting happiness as compared to other emotions having lower accuracy. The result for surprise is almost good [8]. In the Facial Emotion Recognition Model (FERC) that NinadMehendale suggested, they used a method that was based on a two-level CNN architecture. The first level describes how to remove the background from an input image while still preserving the core expressional vector using a normal CNN network module (EV). Here, the EV is directly proportional to the changes in expressions on face. And the second level mainly concentrates on facial feature vector extraction. To achieve highest accuracy, they worked on a large dataset with the sample size of 10,000 images. Accuracy of this model is 96% [1]. In this model Ruhi Jaiswal used Depth- wise seperable convolutions composing of two different layers in which first layer is depth- wise convolutions which seperates the spatial cross- correlations from the channel crosscorrelations. And the second layer is point- wise convolutions which generates a prediction by using a soft-max activation function and global average pooling. They worked on FER2013 datasets which is a cleaned dataset with 28,709 sets. Disgust expression is hardly understood by the trained model and, currently it has an accuracy of 66% [2]. This architecture was proposed by Dr. D. Dhanya and their team which consist of five layers for facial emotion recognition. This model was prepared by using FER2013 datasets provided by Kaggle. The first layer accepts input of size 48x48 black and white and convolved with 5x5 kernel which reduces the spatial dimensions. The second layer convolves a 3x3 kernel with 64 pixels as input from the first layer's output. The third layer now receives input from the second's output and convolves it with a 3x3 kernel and 128 filters. Finally, the output is produced by the two dense layers using the soft-max activation function. This model has a 98.7666% accuracy rate. [3]. Arvind R and the team members proposed a model Facial Emotion Recognition Using CNN, in this model image is processing using Gabor Filter, Model training using CNN, saved model using json and use it for testing, virtualisation, with Metplotlib, Gabor filters and CNN. Accuracy of this model is best in LDA (96.25%) and lowest in CNN (93%) [5].
06-04-2024 --- Start
LIGHT WEIGHT CONVOLUTION NETWORK: The main idea of lightweight model design is to design a “network computing method”, for the convolution method. At first, there were various convolution methods were introduced, then six lightweight convolutional neural networks given excellent results in recent years and the innovations of the model are discussed. Following this, the accuracy and parameters of each model on the ImageNet data set are analyzed, and the lightweight techniques of each model are compared.

POOLING LAYERS: A pooling layer is a new layer added after the convolutional layer. Specifically, after a nonlinearity (e.g. ReLU) has been applied to the feature maps output by a convolutional layer; for example, the layers in a model may look as follows: Input Image, Convolutional Layer, Nonlinearity, Pooling Layer. It is a common pattern for adding a pooling layer next to convolutional layer which issued to ordering layers within a convolutional neural network that may be repeated one or more times in a given model

CNN is the most popular way of analyzing images. CNN is different from a multi-layer perceptron (MLP) as they have hidden layers, called Convolution Layers. Twolevel CNN framework is the method proposed here. The first layer recommended is background removal used to extract emotions form an image. 

The working of the two-level CNN network. In the First layer, CNN network module is used to extract primary expressional vector (EV) is generated by tracking down relevant facial points of importance. EV changes with changes in expression. These vectors are passed in the CNN model form which the results obtained.

 ![Picture10](https://github.com/user-attachments/assets/4354029a-b4f3-4404-a62e-988daecabba5)

Filters in Convolution Layers Within each layer, four filters were used. The input image fed to the first part CNN generally consists of shapes, edges, textures, and objects along with the face. The edge detector and corner detector are used at the Convolution layer.

 ![Picture11](https://github.com/user-attachments/assets/b3343e31-85b3-4774-a280-e0b95f3dcfa8)


facial emotion detection with minimal parameters. Anonymous emotion detection for online education is a great tool to evaluate and improve the online student journey. Emotional feedback is used to evaluate a school's course materials, teaching techniques, organization, and layout as students’ progress through each module in real time. Find and optimize points of attraction or course stumbling blocks using genuine facial responses and engagement levels. Since it uses very few para meters it makes it easy detect expressions during time and also helps the physically disabled people a great deal. It helps us to understand their mood fluctuations.
In this real time facial expression detection, we are using pre trained weights from the open source.
06-04-2024- end



References:
1. NinadMehendale, “Facial emotion recognition using convolutional neural networks (FERC)”, SN Applied Sciences (2020), https://doi.org/10.1007/s42452-020-2234-1. 2. Ruhi Jaiswal, “Facial expression classification using CNN and its applications”, 15th (IEEE) 2020, 3. Dr. D. Dhanya et.al, “Emotion analysis using CNN”, ICCIDT (2022), ISSN: 2278-0181 4. N. Swapna Gond et.al, “Facial emoji recognition”, IJTSRD (2019), ISSN: 2456-6470 5. Arvind R et.al, “Facial Emotion Recognition Using CNN, IJRASET (2022), http://doi.org/10.22214/ijraset.2022.41536. 6. SoadAlmabdy and LamiaaElrefaei, “Deep Convolutional Neural Network-Based Approaches For Face Recognition, MDPI Applied Sciences (2019), 9, 4397; http://doi:10.3390/app9204397. 7. Li, Chieh-En James et.al, “Emotion Recognition Using CNN (2019), https://docs.lib.purdue.edu/purc/2019/Posters/63. 8. Akash Saravanan et.al, “Facial Emotion Recognition using CNN, 12 oct 2019, arXiv:1910.05602vl. 9. ByoungChul Ko, “A Brief Review of Facial Emotion Recognition Based on Visualisation”, 2018, http://doi:10.3390/s18020401. 10. Shivam Singh and Prof s Graceline Jasmine, “Facial Recognition System”, IJERT (2019), ISSN:22780181. 11. Steve Lawrence et.al, “Face Recognition: A Convolutional Neural Network Approach, IEEE 1997 12. Sameer Aqib Hashmi, “Face Detection in Extreme Conditions: A Machine-Learning Approach, email: Sameer.aqib@northsouth.edu








