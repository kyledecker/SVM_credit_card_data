%% Define x-axis
chi = [-2:0.01:2];

%% Logistic Loss
logistic_loss = log(1+exp(-1*chi));

%% Hinge Loss
hinge_loss = (1-chi);
index = (hinge_loss<0);
hinge_loss(index) = 0;

%% Plot the 2 loss functions and compare
figure;
plot(chi,1/log(2)*logistic_loss,'LineWidth',2);
hold on
plot(chi,hinge_loss,'LineWidth',2);
xlabel('y*f(x)')
legend('Logistic Loss', 'Hinge Loss')
axis([-2,2,-0.1,3.5])