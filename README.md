# Neural-Network-from-Scratch-using-NumPy

This actual neural network was made from scratch using NumPy without any deep learning frameworks. The data was processed seperately from tensorflow and pytorch and converted to numpy arrays. 

This neural network uses L2 regularization and he initialization

The first application of the neural network was a clustering classifer which created a decision boundary between the blue and red points on the graph. 

![image](https://user-images.githubusercontent.com/108239710/194942939-4d45c220-cfd6-4de9-867d-99a86585b8b1.png)

This image is the data plotted using matplotlib

Results of the NN:

![image](https://user-images.githubusercontent.com/108239710/194943371-9085f7e0-46c4-42d0-b9c4-fa24cbecbec6.png)

Graph of Cost Function:

![image](https://user-images.githubusercontent.com/108239710/194943405-178f8494-8ae7-42c8-9657-44eaa0ea5586.png)

The second application of the neural network was on the fashion-mnist dataset. 

![image](https://user-images.githubusercontent.com/108239710/194943753-39e77214-17b1-4ab1-abc5-88ec11fdf81e.png)

The basic neural network was able to obtain around 90% accuracy. Normally this data is used by a convolutional neural network, so generalizing between different types of shirts was the main flaw in learning this data set from the simple neural network.

![image](https://user-images.githubusercontent.com/108239710/194943786-86091214-5435-48b2-bba9-6e5b129a600b.png)
![image](https://user-images.githubusercontent.com/108239710/194943805-ae5ec3b7-5a73-47c0-954c-bb019c9ac02c.png)

As you can see, the neural network did most of the learning in the first epoch, and stayed at around 90% accuracy. 

Graph of Cost Function:

![image](https://user-images.githubusercontent.com/108239710/194943963-cc6012e5-6422-4f74-af86-73c075b1dd21.png)

The artitectures for both applications can be found in the source code



