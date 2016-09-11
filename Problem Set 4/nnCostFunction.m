function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 - Forward Prop
% Initialize layer one = X with the bias unit added
a1 = [ones(m,1) X];
z2 = (a1*Theta1'); % Raw input using Theta1 weights
a2 = [ones(size(z2,1),1) sigmoid(z2)]; % Add a bias and sigmoid the raw input

h_theta = sigmoid(a2*Theta2'); % This is useful later
a3 = h_theta; % the output of the last layer is the same as h_theta
% NOTE THAT h_theta IS WHAT WE ACTUALLY OUTPUT AS OUR PREDICTION

% I believe this creates the output as proper vectors
y_matrix = eye(num_labels)(y,:); % I'm not positive how this works

J = (sum(sum((-1).*y_matrix.*log(h_theta)) - sum((1-y_matrix).*(log(1-h_theta)))))/m;

% By Dr. Ng's recommendation we add this separately

% take note of 2:end. We don't regularize the biases
regterm = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2)));
J = J + regterm;

% Back Prop

% Note that d3 is the attempted output minus the actual output 
d3 = a3 - y_matrix;                                             % has same dimensions as a3

% Get the delta's of layer 2 by multiplying outgoing weight by the derivative of our layer
d2 = (d3*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)];     % has same dimensions as a2

% calculates capital delta all at once
D1 = d2(:,2:end)' * a1;    % has same dimensions as Theta1
D2 = d3' * a2;    % has same dimensions as Theta2

% now we get the actual matrices of gradients
% Note that lambda = 1 here so it is omitted
Theta1_grad = Theta1_grad + (1/m) * D1;
Theta2_grad = Theta2_grad + (1/m) * D2;

% below is my first attempt. Not bad!
%Theta1_grad(:,2) = lambda.*(Theta1(:,2));
%Theta2_grad(:,2) = lambda.*(Theta2(:,2));

% here is the actual answer
% REGULARIZATION OF THE GRADIENT

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
