from model import EncoderCNN, DecoderRNN
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from data_loader import get_loader
from vocabulary import Vocabulary
from pycocotools.coco import COCO
import sys
sys.path.append('./opt/cocoapi/PythonAPI')

# Define a transform to pre-process the testing images.
transform_test = transforms.Compose([transforms.Resize((224, 224)),                   # resize image
                                     transforms.ToTensor(),                           # convert the PIL Image to a tensor
                                     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                                          (0.229, 0.224, 0.225))])


# Convert image to tensor and pre-process using transform
PIL_image = Image.open('test_images/airplane.jpg').convert('RGB')
orig_image = np.array(PIL_image)
image = transform_test(PIL_image)
image = image.view(1, image.size()[0], image.size()[1], image.size()[2])

# Visualize sample image, before pre-processing.
plt.imshow(np.squeeze(orig_image))
plt.title('Original image')
ax = plt.axes()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the saved models to load.
encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

# Select appropriate values for the Python variables below.
embed_size = 512
hidden_size = 512

# Initialize Vocabulary
vocab = Vocabulary(vocab_threshold=None, vocab_file='./vocab.pkl', start_word="<start>",
                   end_word="<end>", unk_word="<unk>",
                   vocab_from_file=True)

# The size of the vocabulary.
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
decoder.load_state_dict(torch.load(os.path.join('./saved_models', decoder_file), map_location=device))
encoder.load_state_dict(torch.load(os.path.join('./saved_models', encoder_file), map_location=device))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)

sentence = vocab.convert_sentence(output)

# display final result
ax = plt.axes()

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide ticks
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())

plt.imshow(np.squeeze(orig_image))
plt.xlabel(sentence, fontsize=12)
plt.show()
