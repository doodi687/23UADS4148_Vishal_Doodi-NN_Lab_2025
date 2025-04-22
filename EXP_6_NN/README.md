<h1>Object 6</h1>
<h4>  WAP  to  train  and  evaluate  a  Recurrent  Neural  Network  using  PyTorch  Library  to 
predict the next value in a sample time series dataset.  </h4>
<hr>


<h3>Description of the model:-</h3>
<b>1. Data Set : </b><p>The dataset used for training the model is the International Airline Passengers dataset, which records the monthly total number of airline passengers from January 1949 to December 1960. 
</p>
<b>2. Model Architecture : </b><p>This code defines a simple Recurrent Neural Network (RNN) model for handling sequential data. It first processes the input through an RNN layer, which captures patterns across the sequence over time. After going through the sequence, it takes the output from the final time step, assuming it carries the most important information. This output is then passed through a fully connected layer to generate the final prediction.</p>

<b>4. Experimental Variations : </b><p>
<ul>
<li>The input and output sizes are both set to one, as the task involves predicting a single continuous value at each step.</li> <li>The model uses a hidden layer with 32 units and a single recurrent layer. </li>
<li>The loss function is Mean Squared Error (MSE) and the model is optimized using the Adam optimizer with a learning rate of 0.01. </li>
<li>Number of epochs are 100. </li></ul>
</p>





<h3>Description of the code:-</h3> 
<ol>
<li><b>Import Required Libraries : </b><ul>
<li>torch and torch.nn:These are the core PyTorch libraries for creating the model and defining layers like RNN and Linear.</li>
<li>pandas: Used for reading and handling the dataset.</li>
<li>numpy: Provides efficient operations on arrays, particularly for data processing and manipulation.</li>
<li>matplotlib.pyplot: For visualizing the data, loss curves, and accuracy curves.</li>
<li>MinMaxScaler from sklearn.preprocessing: This is used to scale the data between 0 and 1, as neural networks typically perform better with normalized data.</li>
</ul></li><br>



<li><b>Load and Preprocess the Data: </b><ul><li>The dataset 'airline-passengers.csv' is loaded using pandas.</li>
<li>Only the 'Passengers' column is used — representing monthly international airline passenger numbers.</li>
<li>The passenger numbers are normalized between 0 and 1 using MinMaxScaler from scikit-learn.</li></ul>
</li><br>

<li><b>Create Sequences for RNN Input: </b><ul>
<li>A helper function create_dataset prepares sequences of length seq_length (set to 10).</li>
<li>For each sequence, the model learns to predict the next passenger count.</li>
<li>Features X and labels y are created and converted into PyTorch tensors.</li>
</ul>
</li><br>

<li><b>Define the RNN Model:</b><ul><li>A class RNNModel is created by extending nn.Module.</li>
<li>This consists of :</li><ul>
<li>An RNN layer (nn.RNN) with specified input_size, hidden_size, and num_layers.</li>
<li>A fully connected (Linear) layer to output the final prediction.</li></ul>
<li>In the forward pass:</li><ul>
<li>The sequence output from the RNN is taken.</li>
<li>Only the last time step’s output is passed through the linear layer to predict the next value.</li></ul>
</ul>

<br>
<li><b>Train the Model:
</b>
<ul>
    <li>Loss function: Mean Squared Error Loss (nn.MSELoss).</li>
    <li>Optimizer: Adam optimizer (torch.optim.Adam)..</li>
    <li>Training runs for 100 epochs.</li>
    <li>n each epoch:</li><ul>
    <li>Forward pass through the model.</li>
    <li>Compute loss and backpropagate gradients.</li>
    <li>Update model parameters.</li>
    </ul>
    <li>Loss and a custom Accuracy (based on Mean Absolute Error) are calculated and stored for each epoch.</li>
    <li>Every 10 epochs, the current loss and accuracy are printed.</li>
</ul>
</li><br>

<li><b>Final Evaluation</b></li>
<ul>
    <li>After training, the model is evaluated.</li>
    <li>Final Mean Squared Error (MSE) and custom accuracy are calculated between the predicted and actual passenger counts.</li>
    <li>Predictions and actual values are de-normalized (inverse of MinMax scaling) before evaluation.</li>
    <li>Plots for "Predictions vs Actual Values", "Loss Curve" and  "Accuracy Curve" are displayed. </li>
</ul>
<hr>


<h3>My Comments</h3>
<ul><li>
The batch size is not defined here instead the whole dataset is fed into the network because the size of the dataset is small.</li>
<li>Here plain RNN is used which works well for only small datasets, in case of lager ones the RNN does not perform well because of vanishing gradient problem. In such cases models like "Long Short-Term Memory (LSTM)" should be preferred. </li>
<li>In this case using LSTM gives even lower accuracy this is because this dataset being smaller in size does not require storing information and as such it perform well even on plain RNN. </li>
</ul>










