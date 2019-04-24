clear;
path = 'D:\Ismail''s Folder\AUC\CS 585 - Assignment 2\Cars - Training\';
listing = dir(fullfile(path, '*.jpg'));
% sum =zeros(0, 0);


x = length(listing);
sumx = {};
% filenames = zeros(size(listing));
for i=1:numel(listing)
   filenames = listing(i).name;
%     sum = [sum; filenames];
sumx{i} = filenames;
end

sumx = transpose(sumx);

clear filenames;
% save listAll sum;

% if nargin < 3 
%     subDim = dim - 1; 
% end; 
 
 
disp(' ') 
 
% load listAll; 

listAll = sumx;
trainList = listAll;
 
% Constants 
numIm = length(trainList); 
subDim = 70;
 
 
% Memory allocation for DATA matrix 
fprintf('Creating DATA matrix\n') 
tmp = imread ( [path char(listAll(1))] ); 
% [m, n] = size (tmp);                    % image size - used later also!!! 
m = 64;
n = 64;
DATA = uint8 (zeros(m*n, numIm));       % Memory allocated 
clear str tmp; 
 
% Creating DATA matrix 
for i = 1 : numIm 
    im2 = imread ( [path char(listAll(i))] );
    im2 = rgb2gray(im2);
    im = imresize(im2, [m n]);
    DATA(:, i) = reshape (im, m*n, 1); 
end; 
save DATA DATA; 
clear im; 
 
% Creating training images space 
fprintf('Creating training images space\n') 
dim = length (trainList); 
imSpace = zeros (m*n, dim); 
for i = 1 : dim 
    index = strmatch (trainList(i), listAll); 
    imSpace(:, i) = DATA(:, index); 
end; 
save imSpace imSpace; 
clear DATA; 
 
% Calculating mean face from training images 
fprintf('Zero mean\n') 
psi = mean(double(imSpace'))'; 
save psi psi; 
 
% Zero mean 
zeroMeanSpace = zeros(size(imSpace)); 
for i = 1 : dim 
    zeroMeanSpace(:, i) = double(imSpace(:, i)) - psi; 
end; 
save zeroMeanSpace zeroMeanSpace; 
% clear imSpace; 
 
% PCA 
fprintf('PCA\n') 
L = zeroMeanSpace' * zeroMeanSpace;         % Turk-Pentland trick (part 1) 
[eigVecs, eigVals] = eig(L); 
 
diagonal = diag(eigVals); 
[diagonal, index] = sort(diagonal); 
index = flipud(index); 
  
pcaEigVals = zeros(size(eigVals)); 
for i = 1 : size(eigVals, 1) 
    pcaEigVals(i, i) = eigVals(index(i), index(i)); 
    pcaEigVecs(:, i) = eigVecs(:, index(i)); 
end; 
 
pcaEigVals = diag(pcaEigVals); 
pcaEigVals = pcaEigVals / (dim-1); 
pcaEigVals = pcaEigVals(1 : subDim);        % Retaining only the largest subDim ones 
 
pcaEigVecs = zeroMeanSpace * pcaEigVecs;    % Turk-Pentland trick (part 2) 
 
save pcaEigVals pcaEigVals; 
 
% Normalisation to unit length 
fprintf('Normalising\n') 
for i = 1 : dim 
    pcaEigVecs(:, i) = pcaEigVecs(:, i) / norm(pcaEigVecs(:, i)); 
end; 
 
% Dimensionality reduction.  
fprintf('Creating lower dimensional subspace\n') 
w = pcaEigVecs(:, 1:subDim); 
save w w; 
clear w; 
 
% Subtract mean face from all images 
load DATA; 
load psi; 
zeroMeanDATA = zeros(size(DATA)); 
for i = 1 : size(DATA, 2) 
    zeroMeanDATA(:, i) = double(DATA(:, i)) - psi; 
end; 
% clear psi; 
% clear DATA; 
 
% Project all images onto a new lower dimensional subspace (w) 
fprintf('Projecting all images onto a new lower dimensional subspace\n') 
load w; 
pcaProj = w' * zeroMeanDATA; 
% clear w; 
% clear zeroMeanDATA; 
save pcaProj pcaProj;

%%%% Projecting new image(s) onto subspace

% Positive Testing Set
path = 'D:\Ismail''s Folder\AUC\CS 585 - Assignment 2\Cars - Testing\';
% path = 'C:\Users\Public\Pictures\Sample Pictures\';

listing = dir(fullfile(path, '*.jpg'));

x = length(listing);
sumx = {};
% filenames = zeros(size(listing));
for i=1:numel(listing)
   filenamesnew = listing(i).name;
%     sum = [sum; filenames];
sumx{i} = filenamesnew;
end

sumx = transpose(sumx);

clear filenamesnew;
% save listAll sum;

% if nargin < 3 
%     subDim = dim - 1; 
% end; 
 
 
disp(' ') 
 
% load listAll; 

listAllnew = sumx;
testList = listAllnew;

clear sumx;
% clear listAllnew;
% Constants 
numIm = length(testList); 
%subDim = 200;
 
 
% Memory allocation for DATANEW matrix 
fprintf('Creating DATANEW matrix\n') 
tmp = imread ( [path char(listAllnew(1))] ); 
% [m, n] = size (tmp);                    % image size - used later also!!! 
m = 64;
n = 64;
DATANEW = uint8 (zeros(m*n, numIm));       % Memory allocated 
clear str tmp; 
 
% Creating DATANEW matrix 
for i = 1 : numIm 
    im2 = imread ( [path char(listAllnew(i))] ); 
    im2 = rgb2gray(im2);
    im = imresize(im2, [m n]);
    DATANEW(:, i) = reshape (im, m*n, 1); 
end; 
save DATANEW DATANEW; 
clear im; 

% Subtract mean face from new image(s) 
load DATANEW; 
load psi; 
zeroMeanDATAnew = zeros(size(DATANEW)); 
for i = 1 : size(DATANEW, 2) 
    zeroMeanDATAnew(:, i) = double(DATANEW(:, i)) - psi; 
end; 
% clear psi; 
% clear DATANEW; 

% Project all images onto a new lower dimensional subspace (w) 
fprintf('Projecting new image(s) onto a new lower dimensional subspace\n') 
load w; 
pcaProjnew = w' * zeroMeanDATAnew; 
% clear w; 
% clear zeroMeanDATAnew; 
save pcaProjnew pcaProjnew;

% Calculate Euclidean distance between new image(s) and current images

load pcaProj;
load pcaProjnew;
load w;


reconzero = zeros(size(imSpace)); 
recon = zeros(size(imSpace));
reconzero = w*pcaProj;
for i = 1 : dim 
    recon(:, i) = double(reconzero(:, i)) + psi; 
end; 

% reconzeronew = zeros(size(psi)); 
% reconnew = zeros(size(psi));
reconzeronew = w*pcaProjnew;

fprintf('Training Reconstruction complete\n');

for i = 1 : size(testList, 1)
reconnew(:, i) = double(reconzeronew(:,i)) + psi; 
end;

fprintf('Testing Reconstruction complete\n');

% Works - Produces undesirable results
diff = zeros(size(recon));
sumdiff2 = zeros(size(reconnew, 2), size(recon, 2));
sqrtdiff2 = zeros(size(reconnew, 2), size(recon, 2));
for k = 1: size(reconnew, 2) % 31
    for i = 1 : size(recon, 2) % 95
        for j = 1 : size(reconnew, 1)  % 4096
        diff(j, i) = (reconnew(j,k) - recon(j, i))^2;
        sumdiff2(k, i) = sumdiff2(k, i) + diff(j, i);
        end;
        sqrtdiff2(k, i) = sqrt(sumdiff2(k, i));

    end;

fprintf('Differences calculated \n')
end;

[possmallsumdiff, index] = min(sqrtdiff2, [],2);
display(possmallsumdiff);
posmaxsumdiff = max(possmallsumdiff);
posminsumdiff = min(possmallsumdiff);
posavgsumdiff = mean(possmallsumdiff);
display(posmaxsumdiff);
display(posminsumdiff);
display(posavgsumdiff);
display(index);
display(listAll(index));

clear diff;
clear sumdiff2;
clear sqrtdiff2;



% Negative Testing Set
% path = 'D:\Ismail''s Folder\AUC\CS 585 - Assignment 2\Cars - Testing\';
% path = 'C:\Users\Public\Pictures\Sample Pictures\';

path = 'D:\Ismail''s Folder\AUC\CS 485 - Assignment 3\CS 485 - Assignment 3 - Screenshots\';


listing = dir(fullfile(path, '*.jpg'));

x = length(listing);
sumx = {};
% filenames = zeros(size(listing));
for i=1:numel(listing)
   filenamesnew = listing(i).name;
%     sum = [sum; filenames];
sumx{i} = filenamesnew;
end

sumx = transpose(sumx);

clear filenamesnew;
% save listAll sum;

% if nargin < 3 
%     subDim = dim - 1; 
% end; 
 
 
disp(' ') 
 
% load listAll; 

listAllnew = sumx;
testList = listAllnew;

clear sumx;
% clear listAllnew;
% Constants 
numIm = length(testList); 
%subDim = 200;
 
 
% Memory allocation for DATANEW matrix 
fprintf('Creating DATANEW matrix\n') 
tmp = imread ( [path char(listAllnew(1))] ); 
% [m, n] = size (tmp);                    % image size - used later also!!! 
m = 64;
n = 64;
DATANEW = uint8 (zeros(m*n, numIm));       % Memory allocated 
clear str tmp; 
 
% Creating DATANEW matrix 
for i = 1 : numIm 
    im2 = imread ( [path char(listAllnew(i))] ); 
    im2 = rgb2gray(im2);
    im = imresize(im2, [m n]);
    DATANEW(:, i) = reshape (im, m*n, 1); 
end; 
save DATANEW DATANEW; 
clear im; 

% Subtract mean face from new image(s) 
load DATANEW; 
load psi; 
zeroMeanDATAnew = zeros(size(DATANEW)); 
for i = 1 : size(DATANEW, 2) 
    zeroMeanDATAnew(:, i) = double(DATANEW(:, i)) - psi; 
end; 
% clear psi; 
% clear DATANEW; 

% Project all images onto a new lower dimensional subspace (w) 
fprintf('Projecting new image(s) onto a new lower dimensional subspace\n') 
load w; 
pcaProjnew = w' * zeroMeanDATAnew; 
% clear w; 
% clear zeroMeanDATAnew; 
save pcaProjnew pcaProjnew;

% Calculate Euclidean distance between new image(s) and current images

load pcaProj;
load pcaProjnew;
load w;


reconzero = zeros(size(imSpace)); 
recon = zeros(size(imSpace));
reconzero = w*pcaProj;
for i = 1 : dim 
    recon(:, i) = double(reconzero(:, i)) + psi; 
end; 

% reconzeronew = zeros(size(psi)); 
% reconnew = zeros(size(psi));
reconzeronew = w*pcaProjnew;

fprintf('Training Reconstruction complete\n');

for i = 1 : size(testList, 1)
reconnew(:, i) = double(reconzeronew(:,i)) + psi; 
end;

fprintf('Testing Reconstruction complete\n');

% Works - Produces undesirable results
diff = zeros(size(recon));
sumdiff2 = zeros(size(reconnew, 2), size(recon, 2));
sqrtdiff2 = zeros(size(reconnew, 2), size(recon, 2));
for k = 1: size(reconnew, 2) % 31
    for i = 1 : size(recon, 2) % 95
        for j = 1 : size(reconnew, 1)  % 4096
        diff(j, i) = (reconnew(j,k) - recon(j, i))^2;
        sumdiff2(k, i) = sumdiff2(k, i) + diff(j, i);
        end;
        sqrtdiff2(k, i) = sqrt(sumdiff2(k, i));

    end;

fprintf('Differences calculated \n')
end;
[negsmallsumdiff, index] = min(sqrtdiff2, [],2);;
display(negsmallsumdiff);
negmaxsumdiff = max(negsmallsumdiff);
negminsumdiff = min(negsmallsumdiff);
negavgsumdiff = mean(negsmallsumdiff);
display(negmaxsumdiff);
display(negminsumdiff);
display(negavgsumdiff);
display(index);
display(listAll(index));

theta = (posavgsumdiff + negavgsumdiff)/2;
display(theta);

maxsize = numIm + dim;

% hits = zeros(size(possmallsumdiff, 2), size(possmallsumdiff, 1));           % Correct Detection
% misses = zeros(size(possmallsumdiff, 2), size(possmallsumdiff, 1));         % Misdetection
% falsehits = zeros(size(possmallsumdiff, 2), size(possmallsumdiff, 1));      % False Alarm
% truemisses = zeros(size(possmallsumdiff, 2), size(possmallsumdiff, 1));     % Correct Rejection
% confusion = [2, 2];

hits = 1;
misses = 1;
falsehits = 1;
truemisses = 1;

for theta2 = posminsumdiff: 0.1e+03: negmaxsumdiff
    for i = 1: size(possmallsumdiff, 1)
        if (possmallsumdiff(i) <= theta2)

                hits = hits + 1;


        elseif (possmallsumdiff(i) > theta2)

                misses = misses + 1;

        end

        if (negsmallsumdiff(i) <= theta2)

                falsehits = falsehits + 1;


        elseif (possmallsumdiff(i) > theta2)

                truemisses = truemisses + 1;

        end

    end;

%     confusion(1, 1) = truemisses;
%     confusion(1, 2) = falsehits;
%     confusion(2, 1) = misses;
%     confusion(2, 2) = hits;

    ROC(falsehits, hits) = hits;
end

% plot(falsehits, hits);
plot(ROC, 'x');                 % Due to the nature of this matrix, a curve of best fit was not drawn
title('ROC Curve');
xlabel('False Alarms');
ylabel('Hits');





