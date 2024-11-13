from fastai.vision.all import *

# Load a dataset (e.g., Oxford Pets dataset)
path = untar_data(URLs.PETS)

# Set up a DataLoader for the dataset
dls = ImageDataLoaders.from_name_re(path, get_image_files(path), 
                                    valid_pct=0.2, item_tfms=Resize(224), 
                                    batch_tfms=aug_transforms())

# Create a CNN model using transfer learning (resnet34 as base)
learn = cnn_learner(dls, resnet34, metrics=accuracy)

# Train the model
learn.fine_tune(1)

# Save the model
learn.save('model')
