The file SingleBand_Unet.py will adapt the "Default" VGG19 network from pytorch from 3 channels to 1 channel, changing the first conv from 3x64 to 1x64.
the ImageNet_class.py contains a dictinoary with all the 1000 classes with it's names macthing the folders of the dataset.
it's necessary to download de imaget 540k in https://www.kaggle.com/datasets/dimensi0n/imagenet-256.
