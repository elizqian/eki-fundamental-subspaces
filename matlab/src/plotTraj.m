function plotTraj(x,y,fignum,sty)
% if nargin == 2
%     figure; hold on
% else
%     figure(fignum); hold on
% end
x = squeeze(x);
y = squeeze(y);

cmap = colormap;

% Get the number of colors in the current colormap
numCmapColors = size(cmap, 1);

% Generate evenly spaced indices
indices = round(linspace(1, numCmapColors, length(x)));

% Get the colors from the colormap
colors = cmap(indices, :);
for i = 1:length(x)-1
    plot(x(i:i+1),y(i:i+1),'Color',colors(i,:),'LineStyle',sty); hold on
end