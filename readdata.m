% najperw zmień katalog na ten, w którym zapisane są pliki funkcją:
% chdir folder

function [X, Y] = readdata(filename, delimiter)
    X = dlmread(filename, delimiter, 1, 0);
    
    Y = X(:,end);
    X = X(:,1:end-1);
end