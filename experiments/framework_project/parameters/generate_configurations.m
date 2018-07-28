function [configuration] = generate_configurations()
%GENERATE_CONFIGURATIONS generate a set of configurations for the pricing
%   problem
%
%   Output:
%       configuration: structure array with:
%           - name: name of the configuration (acronym and description)
%           - arms: values for the considered arms
%           - pd  : threshold distribution for the considered arms
%
%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

%% Configuration 1
configuration(1).name = '(1) FLNE few arms, optimal arm with low value, second optimal arm near, easy problem';
configuration(1).arms = 1:5;
configuration(1).pd = makedist('Normal', 2, 1);

%% Configuration 2
configuration(2).name = '(2) FLNH few arms, optimal arm with low value, second optimal arm near, hard problem';
configuration(2).arms = 1:5;
configuration(2).pd = makedist('Normal', 0.5, 3);

%% Configuration 3
configuration(3).name = '(3) FLFE few arms, optimal arm with low value, second optimal arm far, easy problem';
configuration(3).arms = (1:5)';
mu = [2; 8];
sigma = cat(3, 0.5, 0.5);
p = [0.8,0.2];
configuration(3).pd = gmdistribution(mu,sigma,p);

%% Configuration 4
configuration(4).name = '(4) FLFH few arms, optimal arm with low value, second optimal arm far, hard problem';
configuration(4).arms = (1:5)';
mu = [2; 7];
sigma(1,1,:) = [1 1];
p = [0.75,0.25];
configuration(4).pd = gmdistribution(mu,sigma,p);

%% Configuration 5
configuration(5).name = '(5) FHNE few arms, optimal arm with high value, second optimal arm near, easy problem';
configuration(5).arms = 1:5;
configuration(5).pd = makedist('Normal', 5, 1);

%% Configuration 6
configuration(6).name = '(6) FHNH few arms, optimal arm with high value, second optimal arm near, hard problem';
configuration(6).arms = 1:5;
configuration(6).pd = makedist('Normal', 3, 5);

%% Configuration 7
configuration(7).name = '(7) FHFE few arms, optimal arm with high value, second optimal arm far, easy problem';
configuration(7).arms = (1:5)';
mu = [1; 5];
sigma(1,1,:) = [0.5 0.5];
p = [0.81,0.19];
configuration(7).pd = gmdistribution(mu,sigma,p);

%% Configuration 8
configuration(8).name = '(8) FHFH few arms, optimal arm with high value, second optimal arm far, hard problem';
configuration(8).arms = (1:5)';
mu = [1; 6];
sigma(1,1,:) = [0.5 0.5];
p = [0.87,0.13];
configuration(8).pd = gmdistribution(mu,sigma,p);

%% Configuration 9
configuration(9).name = '(9) MLNE many arm, optimal arm with low value, second optimal arm near, easy problem';
configuration(9).arms = 1:20;
configuration(9).pd = makedist('Normal', 3, 0.5);

%% Configuration 10
configuration(10).name = '(10) MLNH many arm, optimal arm with low value, second optimal arm near, hard problem';
configuration(10).arms = 1:20;
configuration(10).pd = makedist('Normal', 2, 6);

%% Configuration 11
configuration(11).name = '(11) MLFE many arm, optimal arm with low value, second optimal arm far, easy problem';
configuration(11).arms = (1:20)';
mu = [2; 16];
sigma(1,1,:) = [2 1];
p = [0.93,0.07];
configuration(11).pd = gmdistribution(mu,sigma,p);

%% Configuration 12
configuration(12).name = '(12) MLFH many arm, optimal arm with low value, second optimal arm far, hard problem';
configuration(12).arms = (1:20)';
mu = [2; 15];
sigma(1,1,:) = [2 1];
p = [0.92,0.08];
configuration(12).pd = gmdistribution(mu,sigma,p);

%% Configuration 13
configuration(13).name = '(13) MHNE many arm, optimal arm with high value, second optimal arm near, easy problem';
configuration(13).arms = 1:20;
configuration(13).pd = makedist('Normal', 20, 1);

%% Configuration 14
configuration(14).name = '(14) MHNH many arm, optimal arm with high value, second optimal arm near, hard problem';
configuration(14).arms = 1:20;
configuration(14).pd = makedist('Normal', 22, 5);

%% Configuration 15
configuration(15).name = '(15) MHFE many arm, optimal arm with high value, second optimal arm far, easy problem';
configuration(15).arms = (1:20)';
mu = [6.1; 18];
sigma(1,1,:) = [1 1];
p = [0.7,0.3];
configuration(15).pd = gmdistribution(mu,sigma,p);

%% Configuration 16
configuration(16).name = '(16) MHFH many arm, optimal arm with high value, second optimal arm far, hard problem';
configuration(16).arms = (1:20)';
mu = [8; 18];
sigma(1,1,:) = [1 1];
p = [0.59,0.41];
configuration(16).pd = gmdistribution(mu,sigma,p);

%% Configuration 17
configuration(16).name = '(17) MHFH many arm, optimal arm with high value, second optimal arm far, hard problem';
configuration(16).arms = (1:20)';
mu = [8; 18];
sigma(1,1,:) = [1 1];
p = [0.59,0.41];
configuration(16).pd = gmdistribution(mu,sigma,p);
