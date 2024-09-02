# Natural-Language-Processing

# Project Overview

This project involves training and evaluating machine learning models using various techniques, including Word2Vec embeddings, Bi-LSTM networks, and neural network classifiers. The tasks are focused on natural language processing (NLP) with a specific emphasis on word similarity, named entity recognition (NER), and out-of-vocabulary (OOV) handling.

## Part 1

### Question 1.1: Word2Vec Model and Semantic Similarity
The Word2Vec model was trained to capture semantic similarities between words. We used the `most_similar` method to compute the cosine similarity between specified words and others in the vocabulary. The results are as follows:
- **Most similar word to 'student'**: `students` (cosine similarity: 0.729)
- **Most similar word to 'Apple'**: `Apple_AAPL` (cosine similarity: 0.746)
- **Most similar word to 'apple'**: `apples` (cosine similarity: 0.720)

### Question 1.2: Data Description and B-I-O Tagging
#### Part a: Dataset Description
- **Training set**: 14,989 sentences with labels including `I-PER`, `I-MISC`, `B-LOC`, `I-LOC`, `O`, `B-ORG`, `I-ORG`, `B-MISC`.
- **Development set**: 3,468 sentences with labels including `I-PER`, `I-MISC`, `I-LOC`, `O`, `I-ORG`, `B-MISC`.
- **Test set**: 3,683 sentences with labels including `I-PER`, `I-MISC`, `B-LOC`, `I-LOC`, `O`, `B-ORG`, `I-ORG`, `B-MISC`.

#### Part b: B-I-O Tagging Scheme
Implemented a logic to identify multi-word entities using the B-I-O tagging scheme. The approach stored and recognized multi-word entities based on the presence of 'B' or 'I' prefixes and label changes.

### Question 1.3: Handling OOV Words and Bi-LSTM Architecture
#### Part a: OOV Words
For OOV words, we averaged the embeddings of surrounding words to derive a representation. If no surrounding words were in the pre-trained embeddings, a zero vector was used. This method handled 8% of cases with zero vectors, balancing computational efficiency with accuracy.

#### Part b: Bi-LSTM Neural Network Architecture
1. **Embedding Layer**: Transforms word indices into dense vectors using pre-trained Word2Vec embeddings.
2. **Bi-LSTM Layer**: Processes word embeddings in both forward and backward directions to capture contextual information.
3. **Linear Layer**: Projects LSTM outputs to the label space.
4. **Softmax Function**: Converts scores into probabilities.
5. **Parameter Updates**: Weights and biases in Bi-LSTM and linear layers are updated during training.
6. **Vector Representation**: Length corresponds to the number of unique labels.

#### Part c: Training and Early Stopping
- **Epochs**: 10 (training halted after 5 epochs due to early stopping).
- **Average Time per Epoch**: 167.84 seconds.

### Question 2: Classification and Aggregation Methods
#### Part a: Random Class Selection
Used `numpy.random.choice` to pseudo-randomly select 5 classes ('0', '1', '5', '2', and 'OTHERS') for classification. Seed fixed for reproducibility.

#### Part b: Aggregation Methods
Evaluated different aggregation methods (Summation, Max, Mean) to represent sentences:
- **Mean** was chosen for its robustness to sentence length and consistent performance.

#### Part c: Neural Network Architecture
A simple feedforward neural network with one hidden layer was used, including:
- **Input Layer**: Size of word embeddings.
- **Hidden Layer**: 128 neurons.
- **Output Layer**: 5 neurons.
- **Activation Function**: ReLU.
- **Dropout Rate**: 0.5.
- **Loss Function**: NLLLoss.
- **Optimizer**: Adam with learning rate 0.001 and L2 regularization.

#### Part d: Training
- **Epochs**: 20 with early stopping after 3 consecutive epochs without improvement.
- **Training Time**: ~16.04 seconds.

#### Part e: Model Accuracy
- **Development Set Accuracy**: Improved to 0.8140 by epoch 16.
- **Test Set Accuracy**: 0.8540, indicating good generalization.

