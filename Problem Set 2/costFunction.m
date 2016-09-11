function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% this makes implementation really easy in the cost function
% note that X is n x 3: col1 = test1 col2 = test2
% X goes first since we need to multiple each column (features) but theta (params)
h_theta = sigmoid(X*theta);

% This takes the sum over each value of y and the result of the above.
% log is simply used for its convenient and concave shape
% note that since v only contains 1's or 0's, one of the terms disappears
J = (1/m)*sum(-y.*log(h_theta) - (1 - y).*log(1-(h_theta)));

% and this is simply the derivative
% NOTE that this results in a vector in R3
grad = (1/m)*(X')*(h_theta-y);
grad(2:size(theta,1)) = (1/m) * (X'(2:size(X',1),:)*(h_theta - y) + lambda*theta(2:size(theta,1),:));

% =============================================================

end
