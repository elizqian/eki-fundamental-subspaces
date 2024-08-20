function S = add2struct(varargin)

S = varargin{1}; % first argument should be the struct

for k = 2:length(varargin)  %remaining arguments should be variables to add
    varName = inputname(k);
    S.(varName) = varargin{k};
end