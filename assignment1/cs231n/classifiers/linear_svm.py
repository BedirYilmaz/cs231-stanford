import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1
  # compute the loss and the gradient
  # see https://math.stackexchange.com/questions/2572318/derivation-of-gradient-of-svm-loss/2572319#2572319?
  # see https://bruceoutdoors.wordpress.com/2016/05/06/cs231n-assignment-1-tutorial-q2-training-a-support-vector-machine/
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  count = 0
  margins = np.zeros((num_train,num_classes))
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      margins[i,j] = scores[j] - correct_class_score + delta # note delta = 1
      
      if j == y[i]:
        continue
        
      if margins[i,j] > 0:
        loss += margins[i,j]

  sum_incorrect_predictions = np.sum(margins > 0, axis=1)
  for j in xrange(num_classes):
    wj = np.sum(X[margins[:, j] > 0], axis=0) # gradient of loss for incorrect classes, the > operator will act as the indicator func
    wyi = np.sum(sum_incorrect_predictions[y == j][:, np.newaxis] * X[y == j], axis=0) # gradient of loss for the correct class
    dW[:, j] = wj - wyi
      
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  delta = 1
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_scores = scores[xrange(num_train), y]
  margins = scores - correct_class_scores[:, np.newaxis] + delta
  margins[xrange(num_train), y] = 0
  loss = np.sum(margins[margins > 0])
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  sum_incorrect_predictions = np.sum(margins > 0, axis=1)
  for j in xrange(num_classes):
    wj = np.sum(X[margins[:, j] > 0], axis=0) # gradient of loss for incorrect classes, the > operator will act as the indicator func
    wyi = np.sum(sum_incorrect_predictions[y == j][:, np.newaxis] * X[y == j], axis=0) # gradient of loss for the correct class
    dW[:, j] = wj - wyi
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)

  return loss, dW
