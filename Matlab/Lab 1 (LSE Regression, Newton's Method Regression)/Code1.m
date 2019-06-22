%% Title: Linear regression by LSE Method and Newton's Method
%% Author: Niraj Prakash Kini, (Multimedia Architecture and Processing Laboratory, EECS, NCTU)
%% Year 2019

%% Initial Setup
clc;
clear;
close all;
format long;
precision = 12;

%% Input
T = dlmread('testfile.txt');    % n x 2 matrix. n = number of observation
x = T(:, 1);
b = T(:, 2);
basis = 3;  % degree of polynomial
lambda = 0; %only for LSE

%% Linear Regression by LSE
%Fitting Line
A = MDesign(x, basis);
L = lambda .* MI(basis);

% Main Formula
X = MM(MM(invLU(MM(MT(A), A) + L), MT(A)), b);

disp('LSE : ');
display(X, basis);

Y = ones(length(b), 1);
Y = X(1, 1) .* Y;
for i = 1:basis-1
    for j = 1:length(b)
        Y(j, 1) = Y(j, 1) + X(i + 1, 1) * x(j, 1) .^ i;
    end
end
NewY=Y;

%Total Error
A = MDesign(x, basis);
E = MM(MM(MM(MT(X),MT(A)), A), X) - 2 .* MM(MM(MT(X), MT(A)), b) + MM(MT(b), b);
E = strcat("Total Error : ", num2str(E, precision));
disp(E);
disp(' ');

%% Newton's Method
A = MDesign(x, basis);
X0 = zeros(1, 1);

for i=1:basis
    X0(i, 1) = i;
end

% Main Formula
X1 = X0 - MM(MM(MM(invLU(MM(MT(A), A)), MT(A)), A), X0) + MM(MM(invLU(MM(MT(A), A)), MT(A)), b);

disp("Newton's Method : ");
display(X1, basis);

Y = ones(length(b), 1);
Y = X1(1, 1) .* Y;
for i = 1:basis-1
    for j = 1:length(b)
        Y(j, 1) = Y(j, 1) + X1(i+1, 1) * x(j, 1) .^ i;
    end
end
NewY = Y;

%Total Error
X = X1;
A = MDesign(x, basis);

E = MM(MM(MM(MT(X),MT(A)),A),X) - 2.*MM(MM(MT(X),MT(A)), b) + MM(MT(b),b);
E = strcat("Total Error : ", num2str(E, precision));
disp(E);

%% Visualization
figure;
subplot(2, 1, 1);
p = plot(x, b, 'o', x, NewY, 'b');
p(1).MarkerSize = 15;
title('LSE');
xlabel('X-Axis');
ylabel('Y-Axis');

subplot(2, 1, 2);
p = plot(x, b, 'o', x, NewY, 'b');
p(1).MarkerSize = 15;
title("Newton's Method");
xlabel('X-Axis');
ylabel('Y-Axis');


%% Command Window Display
function display(MatA, bases)
precision = 12;
A = "Fitting Line : ";
    for i = bases:-1:2
        A = strcat(A, num2str(abs(MatA(i, 1)), precision), "X^", num2str(i - 1));
        if MatA(i-1,1) >= 0
            A = strcat(A, " + ");
        else
            A = strcat(A, " - ");
        end
    end
    A = strcat(A, num2str(abs(MatA(1, 1)), precision));
    disp(A);
end

%% Design Matrix calculator
function MatX = MDesign(MatA, bases)
[m, n] = size(MatA);
    if bases < 1 || n ~= 1
        disp('Error in Design element calculator. Bases is less than 1 OR dimentions of matrix are incorrect.');
        return;
    elseif bases == 1
        MatX = ones(m ,1);
    end
    
    MatX = ones(m, bases);
    for i = 1:bases-1
        for j = 1:length(MatA)
            MatX(j, i + 1) = MatA(j, 1) .^ i;
        end
    end
end

%% Inverse by LU decomposition
function [MatInv] = invLU(MatA)
format short;
[m, n] = size(MatA);
    if m ~= n
        disp('Error: invLU. Matrix dimensions.');
        return;
    end

MatC = MI(m);
MatL = MI(m);
MatU = MatA;

    %Finding L and U Matrices----------------------------
    for i = 1:m-1
        for j = 1:m-i
            RowS = MatU(i, :);
            RowB = MatU(j + i, :);
            MatU(j + i, :) = RowB - (RowB(i) / RowS(i)) * RowS;
            MatL(j + i, i) = RowB(i) / RowS(i);
        end
    end
    
    %Checking if Determinent is zero
    [m, ~] = size(MatU);
    MatDet = 1;
    for i = 1:m
        for j = 1:m
            if i == j
                MatDet = MatDet * MatU(i, j);
            end
        end
    end
    
    if MatDet == 0
        str0 = strcat("Determinent = ", num2str(MatDet), ". Error: It is a singular and Non-Invertible Matrix.");
        disp(str0);
        return;
    end
    
    %Finding Inverse using L and U matrices---------------
    MatInv = zeros(m, n);
    for i = 1:m
        MatZ = forward_substitution(MatL, MatC(:, i));
        MatInv(:, i) = backward_substitution(MatU, MatZ);
    end
end

%% Forward Substitution
function [MatFS] = forward_substitution(MatA, MatY)
[m, n] = size(MatA);
[p, ~] = size(MatY);
    if m ~= n || p ~= m
        disp('Error: forward substitution. Matrix dimensions.');
        return;
    end
    
MatFS = zeros(m, 1);
MatFS(1, 1) = MatY(1, 1) / MatA(1, 1);
    for i = 2:n
        A = MatY(i, 1);
        for j = 1:n-1
            A = A - MatFS(j, 1) * MatA(i, j);
        end
        MatFS(i, 1) = A / MatA(i, i);
    end
end

%% Backward substitution
function [MatBS] = backward_substitution(MatA, MatY)
[m, n] = size(MatA);
[p, ~] = size(MatY);
    if m ~= n || p ~= m
        disp('Error: backward substitution. Matrix dimensions.');
        return;
    end
    
MatBS = zeros(m, 1);
MatBS(n, 1) = MatY(n, 1) / MatA(n, n);
    for i = n-1:-1:1
        A = MatY(i, 1);
        for j = 1 : n-i
            A = A - MatBS(n + 1 -j, 1) * MatA(i, n + 1 - j);
        end
        MatBS(i, 1) = A / MatA(i, i);
    end
end

%% Transpose of a matrix
function MatT = MT(MatA)
[m, n] = size(MatA);
MatT = zeros(n, 1);
    for i = 1:m
        for j = 1:n
            MatT(j, i) = MatA(i, j);
        end
    end
end

%% Matrix Multiplication
function MatM = MM(MatA, MatB)
[m, n] = size(MatA);
[p, q] = size(MatB);
MatM = zeros(m, q);

    if n ~= p
        disp('Error in Matrix Multiplication, with matrix dimesions.');
        return;
    end
    
    for i = 1:m
        for j = 1:q
            MatM(i, j) = 0;
            for k = 1:n
                MatM(i, j) = MatM(i, j) + MatA(i, k) * MatB(k, j);
            end
        end
    end
end

%% Identity Matrix Initialization
function MatI = MI(m)
MatI = zeros(m, m);
    for i = 1:m
        MatI(i, i) = 1;
    end
end