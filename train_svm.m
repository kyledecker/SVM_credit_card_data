function [ lambda, lambda_0 ] = train_svm( x_train, y_train )
%Train Hard-margin SVM train fxn
% Set up Solver
n = size(x_train,1);
X = zeros(size(n));
for i = 1:n
    for j = 1:n
        X(i,j) = x_train(i,:)*x_train(j,:)';
    end
end

Y = y_train*y_train';
I = eye(n);
zero = zeros(1,n);
D = Y.*X;

% Solve for alpha
H = D;
f = ones(size(zero));
A = I;
b = zero;
Aeq = Y;
beq = zero;
lb = zero;
ub = f*10;
alpha = quadprog(H,-f,-A,b,Aeq,beq);

% Find Support Vectors and Determine Lambda and Lambda_0
sv_index = alpha>1e-5;
sv_index1 = find(sv_index);
alpha = alpha.*sv_index;

lambda = zeros(1,size(x_train,2));
for i = 1:size(y_train,1)
    lambda = lambda + (alpha(i)*y_train(i)*x_train(i,:));
end
lambda_0 = 1 - lambda*x_train(sv_index1(1),:)';



end

