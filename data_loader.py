import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json


def get_loader(transform,
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=False,
               num_workers=0,
               cocoapi_loc='./opt'):
    """Returns the data loader

    Arguments:
        transform -- Image transform to be applies

    Keyword Arguments:
        batch_size -- Batch size (if in testing mode, must have batch_size=1).
                    (default: {1})
        vocab_threshold -- Minimum word count threshold. (default: {None})
        vocab_file -- File containing the vocabulary.
                    (default: {'./vocab.pkl'})
        start_word -- Special word denoting sentence start.
                    (default: {"<start>"})
        end_word -- Special word denoting sentence end.
                    (default: {"<end>"})
        unk_word -- Special word denoting unknown words.
                    (default: {"<unk>"})
        vocab_from_file -- If False, create vocab from scratch &
                                override any existing vocab_file.
                             If True, load vocab from from existing vocab_file,
                                if it exists. (default: {False})
        num_workers --  Number of subprocesses to use for
                        data loading (default: {0})
        cocoapi_loc -- The location of the folder containing the COCO API:
         https://github.com/cocodataset/cocoapi (default: {'./opt'})

    Returns:
        [type] -- [description]
    """

    # Obtain img_folder and annotations_file.
    if vocab_from_file:
        assert os.path.exists(
            vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_train2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    # Randomly sample a caption length, and sample indices with that length.
    indices = dataset.get_train_indices()

    # Create and assign a batch sampler to retrieve a batch with the sampled indices.
    initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)

    # data loader for COCO dataset.
    data_loader = data.DataLoader(dataset=dataset,
                                  num_workers=num_workers,
                                  batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                          batch_size=dataset.batch_size,
                                                                          drop_last=False))

    return data_loader


class CoCoDataset(data.Dataset):

    def __init__(self, transform, batch_size, vocab_threshold,
                 vocab_file, start_word, end_word, unk_word,
                 annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file,
                                vocab_from_file)
        self.img_folder = img_folder
        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        print('Obtaining caption lengths...')
        all_tokens = [nltk.tokenize.word_tokenize(str(
            self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, index):
        # obtain image and caption
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(
            self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        # return pre-processed image and caption tensors
        return image, caption

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(
            len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)
