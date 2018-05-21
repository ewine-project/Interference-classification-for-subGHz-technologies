# Interference classification for subGHz technologies
## Summary
We provide a Convolutional Neural Network (CNN)-based classifer for classification of 3 subGHz technologies including Sigfox, LoRA and IEEE802.15.4 (subGHz). Among the technologies, we consider 8 signal classes which include 1 class of sigfox transmitting randomly on the 400 sigfox channels, 6 classes of LoRA with spreading factors (SFs) 7, 8, 9, 10, 11, and 12, and 1 class of IEEE 802.15.4 g. We design the CNN-based classifier from the raw IQ samples from 8 signal classes. The dataset comprises of IQ samples of the eight signal classes that we generated and used for classification and can be found at [1]. For the classification task, we consider a usecase of Europe, in which all the considered technologies transmitters were made to operate at a center frequency of 868 MHz. 
## Data set preprocessing and model description
In order to make the streamed IQ samples compatible with CNN, we divide the IQ samples into sensing snapshots, where each sensing snapshot comprises of 500 IQ samples of duration 500 µs. The network structure that we used for classification of 8 signal classes is shown in Table 1. In the designed CNN-based classifier, the input to CNN are the snapshots which are arranged in a two-dimensional matrix as 2 x 500, where the first row corresponds to I-components while the second row corresponds to Q-components. The used CNN structure comprises of two convolutional layers and two dense layers. The last layer is the softmax layer with 8 neurons which corresponds to the fact that the CNN is able to classify 8 signal classes of the three technologies. The total sensing snapshots comprises of 240,000 which means that for each signal class (out of 8 classes of signals) there are in total 240,000 sensing snapshots. Furthermore, we divide the whole dataset into a training and a validation set with a ratio of 70 and 30. In order to have a better training and validation accuracy, the data is normalize in the range from -1 to 1. For training of the proposed CNN approaches, the Adam optimizer [2] is used which give best training and validation accuracies as compared to other optimizers, similarly as in [3]. All the default values of the Adam optimizer are used except the learning, which was reduced from 0.001 to 0.0001 to have better accuracy results. A batch size of 1024 is used in the design of the CNN models, which is nearly equivalent to the memory constraint of the GPUs. The details of the CNN-based classifier can be found in the eWINE deliverable D5.3 [4].

Table 1: Structure of CNN-based classifier. <p align="center">

| Layer | Input size | parameters | Activation function |
| --- | --- |  --- |  --- | 
| Convolutional layer | 2 x 500 | 2 x 3 filter kernel, 64 feature maps, Dropout 60% | Rectified linear |
| Convolutional layer | 2 x 496 x 64 | 1 x 3 filter kernel, 16 feature maps, Dropout 60% | Rectified linear |
| Dense layer | 1 x 1015936 | 500 neurons, Dropout 60% | Rectified linear |
| Dense layer | 1 x 500 | 8 neurons | Softmax |



## References
[1] https://github.com/ewine-project/SubGHz-technologies-dataset-Sigfox-LoRA-and-IEEE802.15.4g-subGHz-<br/>
[2] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.<br/>
[3] J. C. T. J. OShea, and T. C. Clancy, "Convolutional radio modulation recognition networks," in International Conference on Engineering Applications of Neural Networks, , 2016.<br/>
[4] "eWINE deliverable D5.3," https://ewine-project.eu/wp-content/uploads/eWINE_D5.3_Final.pdf.<br/>

## Contact
If you need any further details about the model, then you can contact at adnan.shahid@ugent.be
