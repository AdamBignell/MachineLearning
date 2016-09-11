function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

disp(size(X));

% Note a1 is the input X and contains 400 features
a2 = sigmoid(X * Theta1'); % (5000 x 401) * (401 * 25) = (5000 x 25)
a2 = [ones(m, 1) a2]; % add the bias unit
fprintf('a2 has dimensions:');
disp(size(a2));
a3 = sigmoid(a2 * Theta2'); % (5000 x 26) * (26 * 10) = (5000 x 10)
h_theta = a3; % This corresponds to 5000 rows with a probability of each row being the number

[value, p] = max(h_theta, [], 2);


% =========================================================================


end
