function configuration = generate_configurations_mu_max(mu_max)

mu = [15; -1000];
sigma(1,1,:) = [1 1];
p = [mu_max,1-mu_max];
pd = gmdistribution(mu,sigma,p);

% Configuration 1
configuration(1).name = 'Optimal high, 3 arms';
configuration(1).arms = (1:8:17)';
configuration(1).pd = pd;

% Configuration 2
configuration(2).name = 'Optimal high, 5 arms';
configuration(2).arms = (1:4:17)';
configuration(2).pd = pd;

% Configuration 3
configuration(3).name = 'Optimal high, 9 arms';
configuration(3).arms = (1:2:17)';
configuration(3).pd = pd;

% Configuration 4
configuration(4).name = 'Optimal high, 17 arms';
configuration(4).arms = (1:17)';
configuration(4).pd = pd;

%% Optimal arm with low value

mu = [5; -1000];
sigma(1,1,:) = [1 1];
p = [mu_max,1-mu_max];
pd = gmdistribution(mu,sigma,p);

% Configuration 5
configuration(5).name = 'Optimal low, 3 arms';
configuration(5).arms = (1:8:17)';
configuration(5).pd = pd;

% Configuration 6
configuration(6).name = 'Optimal low, 5 arms';
configuration(6).arms = (1:4:17)';
configuration(6).pd = pd;

% Configuration 7
configuration(7).name = 'Optimal low, 9 arms';
configuration(7).arms = (1:2:17)';
configuration(7).pd = pd;

% Configuration 8
configuration(8).name = 'Optimal low, 17 arms';
configuration(8).arms = (1:17)';
configuration(8).pd = pd;