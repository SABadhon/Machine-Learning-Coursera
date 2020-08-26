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
n = size(theta,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);  %hypothesis function

part1 = sum(y .* log(h));
part2 = sum((1-y) .* log(1-h));
part3 = sum(theta(2:n,1) .^2);
part4 = (lambda/m)*0.5*part3;

whole = part1 + part2;

J = -(1/m)*whole + part4; %Cost Function

calError = h - y;
sumOfError = sum(calError .* X);
temp1 =  (1/m) * sumOfError(1,2:n);

temp2 = (lambda/m)*theta(2:n,1);
temp2 = temp2';

temp3 = (1/m) * sumOfError(1,1);
temp4 = temp1 + temp2; %Grad
grad = [temp3 temp4];


% =============================================================

end
