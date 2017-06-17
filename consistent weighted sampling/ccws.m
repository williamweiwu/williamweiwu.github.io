% ======================================================================= %
%           *** Canonical Consistent Weighted Sampling ***                %
%              Author: Wei WU (william.third.wu@gmail.com)                %
%              CAI, University of Technology, Sydney (UTS)                % 
% ----------------------------------------------------------------------- %                                 
% Citation: W. Wu, B. Li, L. Chen, & C. Zhang, "Canonical Consistent      %
%         Weighted Sampling for Real-Value Weighted Min-Hash", ICDM 2016. %
% ======================================================================= %

function [ fingerprintK,fingerprintY, runtime ] = ccws( C, weightedSet, D)
% Input: 
%	C - the scaling parameter
%   weightedSet - a m*n matrix of weighted sets
%		rows 	- the number of features in the universal sets
%		columns - the number of weighted sets
%   D - the number of hash functions
% Output: 
%   fingerprintK - 'k' in the returned hash code '(k,y)'
%   fingerprintY - 'y' in the returned hash code '(k,y)'
%   runtime - total runtime in seconds


n = size(weightedSet, 2);	% the number of weighted sets
fingerprintI=zeros(n,D);    % fingerprints with the length of D for n weighted sets
fingerprintT=zeros(n,D);

m = size(weightedSet, 1);	% the number of features
tic;
gamma = betarnd(2,1, m, D);
c = gamrnd(2,1, m, D);
beta = unifrnd(0,1, m, D);	

for j=1:n
    
    [wordId,~] = find(weightedSet(:,j)>0);
    tMatrix = floor(C*repmat(weightedSet(wordId,j),1,D)./ gamma(wordId,:) + beta(wordId,:));
    yMatrix = gamma(wordId,:) .* (tMatrix - beta(wordId,:));
    aMatrix = c(wordId,:)./yMatrix - 2*gamma(wordId,:).*c(wordId,:);    
      
	[~,Imin] = min(aMatrix,[],1);
	yWM1 = yMatrix(size(aMatrix,1).*(0:size(aMatrix,2)-1)+Imin);
	fingerprintK(j,:) = wordId(Imin);    % fingerprint for the n-th weighted sets 
    fingerprintY(j,:) = yWM1;
end

runtime = toc;

end


