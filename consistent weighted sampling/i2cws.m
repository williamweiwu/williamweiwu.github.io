% ======================================================================= %
%           *** Canonical Consistent Weighted Sampling ***                %
%              Author: Wei WU (william.third.wu@gmail.com)                %
%              CAI, University of Technology Sydney (UTS)                 % 
% ----------------------------------------------------------------------- %                                 
% Citation: W. Wu, B. Li, L. Chen, C. Zhang, & P. S. Yu "Improved         %
%         Consistent Weighted Sampling Revisited", 	arXiv:1706.01172.     %
% ======================================================================= %

function [ fingerprintI, fingerprintT, runtime ] = i2cws( weightedSet, D)
% Input: 
%   weightedSet - a m*n matrix of weighted sets
%		rows 	- the number of features in the universal sets
%		columns - the number of weighted sets
%   D - the number of hash functions
% Output: 
%   fingerprintK - 'k' in the returned hash code '(k,y)'
%   fingerprintY - 'y' in the returned hash code '(k,y)'
%   runtime - total runtime in seconds

n = size(weightedSet, 2);% number of docs
fingerprintI=zeros(n,D);     % fingerprints with the length of D for n docs
fingerprintT=zeros(n,D);

m = size(weightedSet, 1);

beta1 = unifrnd(0,1, m, D);	
beta2 = unifrnd(0,1, m, D);	
r1 = gamrnd(2,1, m, D);
r2 = gamrnd(2,1, m, D);
c2 = gamrnd(2,1, m, D);

tic;
for j=1:n      
    [wordId,~] = find(weightedSet(:,j)>0);
    tMatrix = floor(repmat(log(weightedSet(wordId,j)),1,D)./ r2(wordId,:) + beta2(wordId,:));
    zMatrix = exp(r2(wordId,:) .* (tMatrix - beta2(wordId,:)+1));
    aMatrix = c2(wordId,:) ./ zMatrix;
         
	[~,Imin] = min(aMatrix,[],1);
	fingerprintI(j,:) = wordId(Imin);    % fingerprint for the n-th docs 
    
    gamma1 = -log(diag(r1(wordId(Imin),:)));
    b = diag(beta1(wordId(Imin),:));
    tMatrix = floor(log(weightedSet(wordId(Imin),j))./ gamma1 + b);
    fingerprintT(j,:) = exp(gamma1 .* (tMatrix - b));
end
runtime = toc;

end


