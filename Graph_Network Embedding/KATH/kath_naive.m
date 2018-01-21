% ======================================================================= %
%         *** $K$-Ary Tree Hashing for Fast Graph Classification ***      %
%              Author: Wei WU (william.third.wu@gmail.com)                %
%              CAI, University of Technology Sydney (UTS)                 % 
% ----------------------------------------------------------------------- %                                 
% Citation: W. Wu, B. Li, L. Chen, Xingquan Zhu, C. Zhang,        		  %
%         "$K$-Ary Tree Hashing for Fast Graph Classification", TKDE.  	  %
% ======================================================================= %


function [K,runtime] = kath_naive(Graphs,sizeOfTable, hashdims)

% Input: 
%   	Graphs 			- a graph set
%		sizeOfTable 	- the number of columns of the traversal table
%		hashdims 		- the number of the fingerprint for each graph
% Output: 
%   	K 				- the kernel matrix derived from fingerprints
%   	runtime 		- total runtime in seconds


N = size(Graphs,2); % number of graphs
R = size(hashdims,2); % number of iterations

primeRange = 700000000;
ps = primes(primeRange);
clear primeRange;

X = cell(1,R);
hfun = cell(1,R);
mPrime = ps(end-hashdims+1:end);
for r = 1:R
    X{r} = ones(N,hashdims(r))*1000000000;
    hfun{r} = ps(randi([1, length(ps)], hashdims(r),2));
end

t = cputime;
for n = 1:N
    V = length(Graphs(n).nl.values); 
    T = ones(V+1,sizeOfTable)*(V+1);
    for v = 1:V
        [~,I] = sort(Graphs(n).nl.values(Graphs(n).al{v}));
        deg = min(length(I),sizeOfTable);
        T(v,1:deg) = Graphs(n).al{v}(I(1:deg));
    end
    nodes = [Graphs(n).nl.values;0];
    S = (1:V)';
    Z = nodes(S);

    Perm = mod(Z*hfun{1}(:,1)'+ones(V,1)*hfun{1}(:,2)',repmat(mPrime, V, 1));
    X{1}(n,:) = min(Perm);
    for r = 2:R
        S = reshape(T(S,:)',1,[]);
        Z = str2hash(reshape(char(nodes(S)),[],V)',ps(end));
        Perm = mod(Z*hfun{r}(:,1)'+ones(V,1)*hfun{r}(:,2)',repmat(mPrime, V, 1));
        X{r}(n,:) = min(Perm);
    end

end
runtime = cputime-t;

K = zeros(N);
for r = 1:R
    K = K+(1-squareform(pdist(X{r},'hamming')));
end