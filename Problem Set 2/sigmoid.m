function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Note that the dot is extremely important as it performs the operation element-wise
% Our clever treatment of z as a matrix no matter what allows this
% Scalar z is simply a 1 x 1 matrix.
g = 1./(1 + exp(-z));


% =============================================================

end
