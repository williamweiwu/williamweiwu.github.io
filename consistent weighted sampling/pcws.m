% ======================================================================= %
%           *** Practical Consistent Weighted Sampling ***                %
%              Author: Wei WU (william.third.wu@gmail.com)                %
%              CAI, University of Technology, Sydney (UTS)                % 
% ----------------------------------------------------------------------- %                                 
% Citation: W. Wu, B. Li, L. Chen, & C. Zhang, "Consistent Weighted       %
%           Sampling Made More Practical", WWW 2017.  					  %               %
% ======================================================================= %

function [ fingerprintK,fingerprintY, runtime ] = pcws( weightedSet, D )
% Input: 
%   weightedSet - a m*n matrix of weighted sets
%		rows 	- the number of features in the universal sets
%		columns - the number of weighted sets
%   D - the number of hash functions
% Output: 
%   fingerprintK - 'k' in the returned hash code '(k,y)'
%   fingerprintY - 'y' in the returned hash code '(k,y)'
%   runtime - total runtime in seconds

n = size(weightedSet, 2);	 % the number of weighted sets
fingerprintK=zeros(n,D);     % fingerprints with the length of D for n weighted sets
fingerprintY=zeros(n,D);

m = size(weightedSet, 1);	 % the number of features
tic;
beta = unifrnd(0,1, m, D);
x = unifrnd(0,1, m, D);	
u1 = unifrnd(0,1, m, D);
u2 = unifrnd(0,1, m, D);
	
for j=1:n
    
    [wordId,~] = find(weightedSet(:,j)>0);
    gamma = -log(u1(wordId,:) .* u2(wordId,:));
    tMatrix = floor(repmat(log(weightedSet(wordId,j)),1,D)./ gamma + beta(wordId,:));
    yMatrix = exp(gamma .* (tMatrix - beta(wordId,:)));
    aMatrix = -log(x(wordId,:)) ./ (yMatrix ./ (u1(wordId,:) ));
     
	[~,Imin] = min(aMatrix,[],1);
	yWM1 = yMatrix(size(aMatrix,1).*(0:size(aMatrix,2)-1)+Imin);
	fingerprintK(j,:) = wordId(Imin);    % fingerprint for the j-th weighted sets 
    fingerprintY(j,:) = yWM1;
end
runtime = toc;

end


