function [ y_pred ] = predict_svm( lambda,lambda_0,x_test )
%Predict_svm Predict the output of hard-margin SVM 

y_pred = (lambda*x_test' + lambda_0)';
index = y_pred>0;
y_pred(index) = 1;
index = y_pred<0;
y_pred(index) = -1;


end

