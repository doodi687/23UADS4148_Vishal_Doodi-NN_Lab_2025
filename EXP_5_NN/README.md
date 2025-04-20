
<h1>Object 5</h1>
<h4> WAP  to  train  and  evaluate  a  convolutional  neural  network  using  Keras  Library  to 
classify  MNIST  fashion  dataset.  Demonstrate  the  effect  of  filter  size,  regularization, 
batch size and optimization algorithm on model performance. </h4>
<hr>

<h2> Model Description </h2>


This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify images from the **Fashion MNIST** dataset. 



### Dataset
- **Dataset:** Fashion MNIST (28x28 grayscale images across 10 classes)
- **Preprocessing:**
  - Normalized pixel values to [0, 1]
  - Reshaped to (28, 28, 1)
  - One-hot encoded the labels
  - Used 'ImageDataGenerator' for data augmentation:

###  Model Architecture

- **Input Layer:** (28, 28, 1)
- **Conv Block 1:**
  - 2 × Conv2D → BatchNorm → ReLU
  - MaxPooling2D
  - Dropout
- **Conv Block 2 (Residual Block):**
  - 2 × Conv2D → BatchNorm → Add(Shortcut) → ReLU
  - MaxPooling2D
  - Dropout
- **Conv Block 3:**
  - Conv2D
  - GlobalAveragePooling2D
- **Dense Layers:**
  - Dense(512) → BatchNorm → ReLU → Dropout
  - Output: Dense(10) with Softmax


### Training Hyperparameters

- **Optimizer:** 'Adam' with Exponential Learning Rate Decay
- **Loss Function:** 'CategoricalCrossentropy' with 'label_smoothing=0.1'

### Evaluation

- Loads the best saved model weights using 'ModelCheckpoint'.
- Evaluates on the test set
- Plots training and validation loss and accuracy over epochs

<hr>

<h2> Code Description </h2>

### Dataset

- **Fashion MNIST**: 28x28 grayscale images of fashion items.
- **Classes**: 10 categories (e.g., T-shirt/top, Trouser, Pullover, etc.)
- **Preprocessing**:
  - Normalized pixel values to [0, 1].
  - Reshaped data to (28, 28, 1) for CNN compatibility.
  - One-hot encoded labels.
  - Data augmentation using ImageDataGenerator:
    - Rotation, width/height shift, zoom, shear, etc.

### Model Architecture

- **Input Layer**: (28, 28, 1)
- **Conv Block 1**:
  - 2 × Conv2D + BatchNormalization + ReLU
  - MaxPooling2D
  - Dropout(0.3)
- **Conv Block 2 (with residual)**:
  - 2 × Conv2D + BatchNormalization + Residual Connection + ReLU
  - MaxPooling2D
  - Dropout(0.3)
- **Conv Block 3**:
  - 2 × Conv2D + BatchNormalization + ReLU
  - GlobalAveragePooling2D
- **Fully Connected Layers**:
  - Dense(512) + BatchNormalization + ReLU + Dropout(0.6)
  - Dense(10) with Softmax activation

### Training Strategy

- **Optimizer**: Adam with an **Exponential Decay Learning Rate Schedule**.
  ```python
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.001,
      decay_steps=1000,
      decay_rate=0.9
  )

### Training 

- Training of the model using augmented data is carried out.
- Evaluated the accuracy for validation set for every epoch.
- Best weights are saved to 'best_model.h5'.

### Evaluation 

- Loads the best weights.
- Evaluates the final model on the test dataset.

### Visualization 

- **Loss Curve:** Shows train and validation loss.
- **Accuracy Curve:** Shows train and validation accuracy.
<h2>My Comments :-</h2>

<ul>
<li>The maximum test accuracy achieved is 94.53.</li><br>
<li>Adding the Batch Normalisation and dropout layers along with applying the data augmentation proved effective in bringing around 95% accuracy.</li><br>
<li>Additionally in the previous model the gap between the validation and training set was prominent indicating model overtraining which is reduced to a significant extent in this model eventually fixing the problem of model overtraining.</li>