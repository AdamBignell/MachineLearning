function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % alpha: learning rate, determines how big of a set the descent takes
    % (1/m) is an artifact of the derivative. We initialized (1/2m) to handle least-square function
    % (((X * theta) - y)' * X)' virtually MUST be drawn out:
    % X * theta = (m x 2) * (2 x 1) = (m x 1)
    % (m x 1) - y = (m x 1) -> transpose so error stored as vector, call this C = (1 x m)
    % C * X = (1 x m) * (m x 2) = (1 x 2) = Our pre-learning values of theta, transpose to get vector

    theta = theta - (alpha * (((X * theta) - y)' * X)' * (1/m));
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    
end

end
