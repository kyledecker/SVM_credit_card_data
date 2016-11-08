%% Part A
% Hard-margin SVM
clear; clc; close all;
% Load data
load('fisheriris.mat')
x = meas(1:100,:);
y = zeros(100,1);
y(1:50) = 1;
y(51:end) = -1;

% Random Permutation then split into tranining and testing
rand_index = randperm(size(y,1));
x_shuffle = x(rand_index,:);
y_shuffle = y(rand_index);

train_size = 9*round(size(y,1)/10);
x_train = x_shuffle(1:train_size,:);
x_test = x_shuffle(train_size+1:end,:);
y_train = y_shuffle(1:train_size);
y_test = y_shuffle(train_size+1:end);

% Train SVM
[lambda, lambda_0] = train_svm(x_train,y_train);
% Predict SVM
[ y_pred ] = predict_svm( lambda,lambda_0,x_test );

%% Part B SVM w/ k(x,z) = x'z on credit card data
clear; clc;
load('creditCard.mat')
data = double(creditCard);
x = data(:,1:9);
y = data(:,10);
index = (y==0);
y(index) = -1;

% Random Permutation then split into tranining and testing
rand_index = randperm(size(y,1));
x_shuffle = x(rand_index,:);
y_shuffle = y(rand_index);

train_size = 9*round(size(y,1)/10);
x_train = x_shuffle(1:train_size,:);
x_test = x_shuffle(train_size+1:end,:);
y_train = y_shuffle(1:train_size);
y_test = y_shuffle(train_size+1:end);

% Train Linear SVM
K = 'linear';
svmModel_linear = fitcsvm(x_train, y_train, 'Standardize', true, 'KernelFunction', K);
svmModel_linear = fitPosterior(svmModel_linear);
[~, Yscores] = predict(svmModel_linear, x_test);

y_predict = double(Yscores(:,2)>0.5);
index = (y_predict==0);
y_predict(index) = -1;
accuracy = (sum(abs(y_predict == y_test))/length(y_test))*100;
            
% Compute the standard ROC curve and the AUROC
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(y_test, Yscores(:, 2), 1);

fprintf('Classifcation Results for Linear SVM... \n')
figure;
plot(Xsvm, Ysvm,'-')
xlabel('false positive rate');
ylabel('true positive rate');
axis([-0.01,1.01,0,1.01])
title('ROC Curve for Linear SVM')
fprintf('Classifcation Accuracy on Test Set: %f %% \n', accuracy);
fprintf('AUC : %f \n', AUCsvm);

%% Part C SVM w/ radial basis kernel on credit card data

% Sigma^2 = 2

% Train RBF SVM
K = 'rbf';
svmModel_rbf_2 = fitcsvm(x_train, y_train, 'Standardize', true, 'KernelFunction', K,'KernelScale',sqrt(2));
svmModel_rbf_2 = fitPosterior(svmModel_rbf_2);
[~, Yscores] = predict(svmModel_rbf_2, x_test);

y_predict = double(Yscores(:,2)>0.5);
index = (y_predict==0);
y_predict(index) = -1;
accuracy = (sum(abs(y_predict == y_test))/length(y_test))*100;
            
% Compute the standard ROC curve and the AUROC
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(y_test, Yscores(:, 2), 1);

fprintf('Classifcation Results for Radial Basis Kernel (sigma^2 = 2) SVM... \n')
figure;
plot(Xsvm, Ysvm,'-')
xlabel('false positive rate');
ylabel('true positive rate');
axis([-0.01,1.01,0,1.01])
title('ROC Curve for Radial Basis Kernel (sigma^2 = 2) SVM')
fprintf('Classifcation Accuracy on Test Set: %f %% \n', accuracy);
fprintf('AUC : %f \n', AUCsvm);

% Sigma^2 = 20

% Train RBF SVM
K = 'rbf';
svmModel_rbf_20 = fitcsvm(x_train, y_train, 'Standardize', true, 'KernelFunction', K,'KernelScale',sqrt(20));
svmModel_rbf_20 = fitPosterior(svmModel_rbf_20);
[~, Yscores] = predict(svmModel_rbf_20, x_test);

y_predict = double(Yscores(:,2)>0.5);
index = (y_predict==0);
y_predict(index) = -1;
accuracy = (sum(abs(y_predict == y_test))/length(y_test))*100;
            
% Compute the standard ROC curve and the AUROC
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(y_test, Yscores(:, 2), 1);

fprintf('Classifcation Results for Radial Basis Kernel (sigma^2 = 20) SVM... \n')
figure;
plot(Xsvm, Ysvm,'-')
xlabel('false positive rate');
ylabel('true positive rate');
axis([-0.01,1.01,0,1.01])
title('ROC Curve for Radial Basis Kernel (sigma^2 = 20) SVM')
fprintf('Classifcation Accuracy on Test Set: %f %% \n', accuracy);
fprintf('AUC : %f \n', AUCsvm);
            