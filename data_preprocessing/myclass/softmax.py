import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_vectorized(W, X, y):
    """
      Softmax loss function, vectorized version.

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability.                         #
    #############################################################################
    
    N = X.shape[0]
    shift = 1000
    S=np.zeros((N,33))
    for i in range(33):
        x_tmp = X[:,range(i,165,33)]
        if i<11:
            S[:,i]=W[0]*x_tmp[:,0]+W[1]*x_tmp[:,4]+W[2]*x_tmp[:,2]+W[3]*x_tmp[:,1]+W[4]*x_tmp[:,3]**W[6]
        elif (i>=11) and (i<22):
            S[:,i]=W[0]*x_tmp[:,0]+W[1]*x_tmp[:,4]+W[2]*x_tmp[:,2]+W[3]*x_tmp[:,1]
        else:
            S[:,i]=W[0]*x_tmp[:,0]+W[1]*x_tmp[:,4]+W[2]*x_tmp[:,2]+W[3]*x_tmp[:,1]+W[5]*x_tmp[:,3]**W[7]
        
    exps = np.exp(S-np.max(S,axis=1,keepdims=True))
    softm = (exps.T / np.sum(exps, axis=1)).T
    loss = sum(-np.log10(np.diag(softm[:,y-1])+shift))/N
    dW[0] = (sum(X[range(N),y-1])+np.sum(np.sum(X[:,0:33].dot(S.T)))/np.sum(np.sum(softm)))/N


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW