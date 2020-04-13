[image1]: ./images/encoder-decoder-model.png "model"
[image2]: ./images/airplane.png "airplane"
[image3]: ./images/airplane_2.png "airplane2"
[image4]: ./images/baseball.png "baseball"
[image5]: ./images/birthday.png "birthday"
[image6]: ./images/snowboard.png "snowboard"
[image7]: ./images/surfer.png "surfer"
[image8]: ./images/tennis.png "tennis"
[image9]: ./images/train.png "train"
[image10]: ./images/woman_laptop.png "woman_laptop"
[image11]: ./images/woman_dog.png "woman_dog"
[image12]: ./images/coco-examples.jpg "coco_example"

# Image Captioning
The goal of this project is to combine CNNs and RNNs to build a complex, automatic image captioning model.
| Sample results             |                           |
| -------------------------- | ------------------------- |
| ![Airplane][image2]        | ![Airplane2][image3]      |
| ![Baseball_player][image4] | ![Birthday_Party][image5] |
| ![Snowboard][image6]       | ![Surfer][image7]         |
| ![tennis][image8]          | ![train][image9]          |
| ![woman][image10]          | ![woman_2][image11]       |

 ## Code
 `main.py` contains the source code to process an input image

 `model.py` contains the source code defining the network architectures used in the project  

 `data_loader.py` contains the source code to load and preprocess the data

 `vocabulary.py` contains the source code for processing the vocabulary

 `train.py` contains the source code for training and saving the network

 `vocab.pkl` pickle file containing the vocab file generated from the training data annotations

## Model
You can use the pre-trained model placed in the `saved_models` folder, or train your own network. The pre-trained model is an Encoder Convolutional Neural Network (CNN) and a Decoder Recurrent Neural Network (RNN) based on the [Neural Image Caption](https://arxiv.org/pdf/1411.4555.pdf) model. ![Encoder-Decoder Model][image1]

The EncoderCNN uses a pre-trained ResNet-50 architecture, with the final fully-connected layer removed to extract features from the pre-processed images. The output is then passed through a batch normalization layer, and then flattened to a vector, and passed through a Linear layer to transform the feature vector to have the same size as the word embedding, and then passed through another batch normalzation layer. The DecoderRNN is composed of multiple Long Short-Term Memory(LSTM), and a Linear layer. The fully-connected weights have been initialized as the xavier normal weights. The LSTM forget gate bias is initialized to 1 for better perforemance, as sugested in this [paper](http://proceedings.mlr.press/v37/jozefowicz15.pdf). The pre-trained model is trained for 3 epochs with the hyper-parameters set in the train.py file.

 ### Dependencies
- [OpenCV](http://opencv.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [Torch](http://PyTorchpytorch.org)
- [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
- [Pandas](https://pandas.pydata.org/)
- [cpython](https://github.com/python/cpython)
- [Natural Language Toolkit](https://www.nltk.org/)
- [Pillow](https://pypi.org/project/Pillow/)


 ## Data
 ![Dataset][image12]

 In this project the [Microsoft Common Objects in COntext (MS COCO)](http://cocodataset.org/#home) dataset is used to train the network. To get the dataset, and setup the coco API  run the following commands in the project directory:

```
mkdir opt

cd opt

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

make

sudo make install

sudo python setup.py install

cd ..

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip -n annotations_trainval2014.zip -d .

wget http://images.cocodataset.org/zips/train2014.zip

unzip -n train2014.zip -d ./images
```
This dataset consists of:
- **2014 Train/Val annotations [241MB]** annotations.
- **2014 Train images [83K/13GB]** images.

