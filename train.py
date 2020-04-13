import nltk
import torch.utils.data as data
import time
import os
import numpy as np
import math
from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from pycocotools.coco import COCO
import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
nltk.download('punkt')

# Hyperparameters.
# batch size
batch_size = 64

# minimum word count threshold
vocab_threshold = 5

# if True, load existing vocab file
vocab_from_file = False

# dimensionality of image and word embeddings
embed_size = 512

# number of features in hidden state of the RNN decoder
hidden_size = 512

# number of training epochs
num_epochs = 3

# determines frequency of saving model weights
save_every = 1

# determines window for printing average loss
print_every = 100

# name of file with saved training loss and perplexity
log_file = 'training_log.txt'


transform_train = transforms.Compose([

    # smaller edge of image resized to 256
    transforms.Resize(256),

    # get 224x224 crop from random location
    transforms.RandomCrop(224),

    # horizontally flip image with probability=0.5
    transforms.RandomHorizontalFlip(),

    # convert the PIL Image to a tensor
    transforms.ToTensor(),

    # normalize image for pre-trained model
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function.
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model.
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer.
optimizer = torch.optim.Adam(params)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

# Open the training log file.
f = open(log_file, 'w')

for epoch in range(1, num_epochs+1):

    for i_step in range(1, total_step+1):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()

        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, num_epochs, i_step, total_step, loss.item(),
            np.exp(loss.item()))

        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()

        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()

        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)

    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join(
            './saved_models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join(
            './saved_models', 'encoder-%d.pkl' % epoch))

# Close the training log file.
f.close()
