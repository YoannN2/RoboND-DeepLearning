
## Intro:

The goal of the project is to have a drone follow a hero, to achieve that we needed the drone to be able to identify the hero, and non-hero, using semantic segmentation which was implemented via a FCN to classify each pixels of an image captured by the drone's camera as either hero, non-hero or the background.

![Drone Following the hero in the simulation](docs\misc\sim_screenshot.png)

## 1. NN Architecture:
The Neural Network's architecture goes as follows:
* Input images are fed into the encoder (layers E1, E2, E3) to extract spatial features, as a normal CNN would, with an increase in the complexity of the features extracted as we go deeper into the layers.
* The output of the encoder is fed into a 1x1 convolution, which represents and stores a latent vector space that contains the extracted features.
* Then the decoder block takes the information from both the 1x1 convolution layer and the encoder's layer, using skip connections to recover spatial informations that might get lost in the encoder and does the semantic segmentation through its layers (D1, D2, D3).
* Finally we get an output image with 3 channels (hero, not-hero, background).

![NN Architecture](docs\assets\NN architecture.png)

A filter size of 32 was used for the first encoder's layer, and was doubled for each following layers in the encoder, allowing more features to be learnt. The 1x1 Convolution layer's filter size was set to 512, instead of doubling it like with the encoder layers, to help the network's overall accuracy. Finally for the decoder the filter size was halved at every layer, while remaining big enough to be able to do the semantic segmentation.

The layers fed into the decoder through the skip connections are concatenated with the previous layer, which are upsampled using bilinear upsampling, before going throught a convolution with a stride of 1 and a kernel size of 3.

For the encoder layers a stride of 2 and a kernel size of 3 was used, whereas in the 1x1 convolution a stride of 1 and a kernel size of 1 was used.

In terms of activation functions ReLU everywhere and softmax for the output.

Batch normalization was used for both the decoder and the encoder which helped regularize the network and gave some wiggle room during the tuning.

```Python
# Encoder block:
def encoder_block(input_layer, filters, strides):    
    layer = separable_conv2d_batchnorm(input_layer, filters, strides=2)    
    return layer

# Decoder block:
def decoder_block(small_ip_layer, large_ip_layer, filters):

    # 1. Upsample the small input layer using the bilinear_upsample() function.
    upsampled_ip_layer = bilinear_upsample(small_ip_layer)

    # 2. Concatenate the upsampled and large input layers using layers.concatenate
    layer = layers.concatenate([upsampled_ip_layer, large_ip_layer])

    # 3. Add some number of separable convolution layers
    layer = separable_conv2d_batchnorm(layer, filters, strides=1)

    return layer
```


## 2. Hyperparameters:
The values were chosen through brute force while experimenting with different architectures, while insight was gained through this method, which helped the final tuning of the network but is by no mean the proper way to do it, but it helps get the "feel of it" at the very least.


* Learning rate: 0.005

The learning rate was proportional to the filter size chosen, and depended on how deep the network was (# of layers) and the batch size.


* Batch size: 64 (a smaller value would work aswell)

The batch size was chosen mostly based on hardware limitations (also affected by both the depth of the FCN and the filter size), my computer could only run batches of 16 images while the AWS instance could do pretty much whatever. After going bananas I settled with 64, but 32 would probably still yield good results (adjust LR accordingly though).


* Epochs: 25

While testing different architectures a smaller epochs of 5 was used to test the convergence of different architectures, for the final version I used 25 epochs, which was enough to reach the final score without overfitting.


* Step per epochs: 200 (default)
* Validation step: 30

To be completely honest, changing those parameters didn't change all that much, compared to the other parameters, which is why they were left to their default value, except the validation step which was set to 30 (from 50 by default) to reduce the amount of operations that needed to be done in each epochs that wasn't directly affecting the training while still retaining its role as a metrics for training the neural network.

![overfitting?](docs\assets\training plot.png)

This is the training curves I got using those parameters for comparison, and the last loss values were: train_loss: 0.0126, val_loss: 0.0272, refer to the .ipynb for more info.


## 3. Limitations of this NN and Dataset:
This model would not be able to work with different subjects (dog, cat, car, etc.) as it has only learnt from images of humans. To make it able to segment out different subjects, the network would have to be trained using data that contains these subjects.

## 4. Future enhancements:
Current issues with the FCN:
* It could be better tuned for sure, using an evolutionary algorithm for tuning the model would be a good experiment.


* Data collection, during the training I gathered more data of the drone patrolling with the target in sight, but the quantity wasn't near enough to have a significant impact, so that would be a potential improvement to make.

![more data!](docs\assets\data_collection.jpg)

Potential enhancements to make:
* The drone's camera captures the depth, yet our network isn't using it.
* The network can differentiate between non-hero and the hero but isn't able to differentiate non-heroes from each others, how can I alter the network for it to properly segment people and detect them as different entities? Using the techniques used in project 3 would be a good place to start.
