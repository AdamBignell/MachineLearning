function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h_theta = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(h_theta) - (1-y).*log(1-(h_theta))) + (lambda/(2*m))*sum(theta(2:length(theta)).^2);

% vectorization removes the need for a sum here
% This is the gradient for the first un-regularized term
% Just believe with all your heart that this is the derivative unless you feel
% like spending some serious calulation time figuring out the above.
grad(1) = (1/m)*(X'(1,:))*(h_theta - y);

% This nasty code basically performs above but from the second feature on to the last
% Note that regularization is performed during/by the (labda/m...) term
grad(2:size(theta,1)) = (1/m) * (X'(2:size(X',1),:)*(h_theta - y) + lambda*theta(2:size(theta,1),:));


% =============================================================

end
