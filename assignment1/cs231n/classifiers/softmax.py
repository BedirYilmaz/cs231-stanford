import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  b = np.zeros((1,W.shape[1]))

  step_size = 0.5
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = XdotW = np.dot(X, W)
  num_train = X.shape[0]    
  f -= np.max(f, axis=0)

  # l_i = -f_y + logE_j e_f_j version   
  for i in xrange(num_train):
    incorrect_predictions_exp = np.exp(f[i,:])
    correct_prediction = f[i,y[i]]
    incorrect_predictions_exp[y[i]] = 0
    sum_of_incorrects = np.sum(incorrect_predictions_exp)
    loss += -correct_prediction + np.log(sum_of_incorrects) 
   
  #probabilistic version
  #correct_predictions = f[np.arange(num_train),y]
  #correct_predictions_exp = np.exp(correct_predictions)
  #incorrect_predictions = f
  #incorrect_predictions = np.ma.array(incorrect_predictions, mask=False)
  #incorrect_predictions.mask[np.arange(num_train),y] = True
  ##incorrect_predictions[np.arange(num_train),y] = 0
  #incorrect_predictions_exp = np.exp(incorrect_predictions)
  #sum_incorrect_predictions_exp = np.sum(incorrect_predictions_exp, axis=1)
  #probs_i = - np.log(correct_predictions_exp/sum_incorrect_predictions_exp)
  #loss = np.sum(- np.log(correct_predictions_exp/sum_incorrect_predictions_exp))
  
  #print(W.shape)
  #print(scores.shape)
  #print(sum_of_incorrects.shape)
  #print(X.shape)
  #print(W.shape)

  
  #XdotW -= np.max(XdotW,axis=1)[:,np.newaxis]
  #num = np.exp(XdotW+b) 
  #XdotW = np.ma.array(XdotW, mask = False)
  #XdotW.mask[np.arange(num_train),y] = True
  #denom = np.sum(num,axis=1)
  #probs = num/denom

  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 

  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  corect_logprobs = -np.log(probs[range(num_train),y])
  data_loss = np.sum(corect_logprobs)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  
  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  #loss /= num_train
  #loss += 0.5 * reg * np.sum(W * W)
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)  
    
  b = np.zeros((1,W.shape[1]))
  step_size = 0.5
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #probabilistic version

  f = XdotW = X.dot(W)
  num_train = X.shape[0]    
  f -= np.max(f, axis=0)

  correct_predictions = f[np.arange(num_train),y]
  correct_predictions_exp = np.exp(correct_predictions)
  incorrect_predictions = f
  incorrect_predictions = np.ma.array(incorrect_predictions, mask=False)
  incorrect_predictions.mask[np.arange(num_train),y] = True
  #incorrect_predictions[np.arange(num_train),y] = 0
  incorrect_predictions_exp = np.exp(incorrect_predictions)
  sum_incorrect_predictions_exp = np.sum(incorrect_predictions_exp, axis=1)
  probs_i = - np.log(correct_predictions_exp/sum_incorrect_predictions_exp)
  loss = np.sum(- np.log(correct_predictions_exp/sum_incorrect_predictions_exp))
  
  # for gradient see http://cs231n.github.io/neural-networks-case-study/
  scores = np.dot(X, W) + b 

  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]  
  corect_logprobs = -np.log(probs[range(num_train),y])
  data_loss = np.sum(corect_logprobs)/num_train  
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  
  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  
  return loss, dW

