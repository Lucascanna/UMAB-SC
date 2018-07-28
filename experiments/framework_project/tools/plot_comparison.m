function varargout = plot_comparison(mean_regret,std_regret,pol_names,varargin)
%PLOT_COMPARISON plot the results (regret) of multiple runs of different
% policies in different configuration 
%
%   Input:
%       mean_regret:    a cell matrix (number of configurations x number of
%           policies) containing the average regret
%       mean_regret:    a cell matrix (number of configurations x number of
%           policies) containing the average regret
%       pol_names:      cell arrat containing the policies names
%       varargin{1}:    configuration structure with field "name" to plot
%           configuration description as plot title
%
%   Output:
%       varargout: if required, handles array for the produced figures
%
%   Copyright ...

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano
n_pol = size(mean_regret,2);
n_conf = size(mean_regret,1);
n_iterations = size(mean_regret{1,1},2);

colors = distinguishable_colors(n_pol);
iterations = 1:n_iterations;
fig = [];
h = [];

for ii = 1:n_conf
    fig(ii) = figure();
    hold on;
    for jj = 1:n_pol
        h(jj) = plot(iterations,mean_regret{ii,jj},'Color',colors(jj,:));
        plot(iterations,mean_regret{ii,jj}+std_regret{ii,jj},'--', 'Color',colors(jj,:));
        plot(iterations,mean_regret{ii,jj}-std_regret{ii,jj},'--', 'Color',colors(jj,:));
    end
    hold off;
    ylabel('R_T');
    xlabel('t');
    if nargin == 4
        title(varargin{1}(ii).name);
    end
    legend(h,pol_names);
end

if ~isempty(fig)
    varargout{1} = fig;
end
