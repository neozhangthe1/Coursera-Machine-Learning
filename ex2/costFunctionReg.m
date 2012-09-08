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

% non-regularized bits first
h = sigmoid(X * theta);
costPos = -y' * log(h);
costNeg = (1 - y') * log(1 - h);
% nonreg is equal to the non-regularized J in costFunction.m
nonreg = (1/m) * (costPos - costNeg);

% pop off the theta(1) param
regTheta = theta(2:end, :);
% calculate the regularization bit
reg = (lambda / (2*m)) * (regTheta' * regTheta);

% J is the the non-regularized cost plus the regularized parameters
J = nonreg + reg;

% g0 is equal to the non-regularized grad in costFunction.m
g0 = (1/m) * (X' * (h - y));
% get regularized bits
gTheta = (lambda / m) * theta;

% zero out the theta(1) term
gTheta(1) = 0;

grad = g0 + gTheta;

% =============================================================

end