function [coeff,latent,QNew,mNew,sqsNew]=IncrementalPCA(QOld,n,xNew,coeffOld,latentOld,mOld,sqsOld)
% INCREMENTALPCA an incremental computation of Principal Components Analysis
%
% [COEFF,LATENT,QNEW,MNEW,SQSNEW] = 
% IncrementalPCA(QOLD,N,XNEW,COEFFOLD,LATENTOLD,MOLD,SQSOLD)
% returns the coefficients COEFF and the vector of PCs LATENT. 
%
% QOLD is the covariance matrix at the previous step (set it to zero at the 
% first step)
% N is the current step counter
% XNEW is the new sample
% COEFFOLD and LATENTOLD can be passed in order to enforce continuity, set
% it to [] if this is not needed/wanted
% MOLD is the mean at the previous step
% SQSOLD is the sum of squares at the previous step
% QNEW,MNEW,SQSNEW are usend as inputs for the next steps respectively for
% QOLD,MOLD,SQSOLD
%
%
% [COEFF,LATENT,QNEW] =IncrementalPCA(QOLD,N,XNEW) 
% is an approximate version that assumes that the means of the data series
% are zero and the variance unitary. This assumption is obviously not
% veryfied in general but, when applying the algorithm on Z-score data the
% coefficients would converge to the ones obtained with the batch algorithm
% once all the samples have been presented.
%
%
% Vittorio Lippi* & Giacomo Ceccarelli^ 
% December 2015
% *Uniklinik Freiburg ^University of Pisa + INFN
%
%

if nargin<6
[QNew]= UpdateQ(QOld,xNew,n);
sqsOld=0;
elseif nargin<7
[QNew,mNew]= UpdateQ(QOld,xNew,n,mOld);
else
[QNew,mNew,sqsNew]= UpdateQ(QOld,xNew,n,mOld,sqsOld);
end

[~,latent,coeff] = svd(QNew/n); % the samples are now n+1
latent=diag(latent)';
m=length(latent);

if (nargin>3 && ~isempty(coeffOld))

    % fix discontinuities due to degenerate subspaces
    th=1/sqrt(m); %%%% THRESHOLD DEFINE HOW IT IS DECIDED
    latentNew=latent;
    coeffNew=coeff;
    moltOld=[];
    moltNew=[];
    i=1;
    while i<=m
        j=sum(abs(latentOld(i)-latentOld(i:end))<th);
        moltOld=[moltOld;j];
        i=i+j;
    end
    i=1;
    while i<=m
        j=sum(abs(latentNew(i)-latentNew(i:end))<th);
        moltNew=[moltNew;j];
        i=i+j;
    end 
    molt=[];
    i=1;
    j=1;
    k=1;
    while(i<=length(moltNew) && j<=length(moltOld))
        if(moltNew(i)==moltOld(j))
            molt(k)=moltNew(i);
            k=k+1;
            j=j+1;
            i=i+1;
        elseif(moltNew(i)>moltOld(j))
            moltOld(j+1)=moltOld(j+1)+moltOld(j);
            j=j+1;
        elseif(moltNew(i)<moltOld(j))
            moltNew(i+1)=moltNew(i+1)+moltNew(i);
            i=i+1;
        end
    end   
    k=1;
    Unew=[];
    for i=molt
        if i>1
            V=coeffOld(:,k:i+k-1);
            U=coeffNew(:,k:i+k-1);
            for j=1:i
                u=U*U'*V(:,j);
                if isempty(Unew)
                    Unew=u./norm(u);
                else
                    u=u-Unew*Unew'*u;
                    Unew=[Unew,u./norm(u)];
                end
            end
        else
            Unew=[Unew,coeffNew(:,k)];
        end
        k=k+i;
    end
    coeff=Unew;
    
    % fix discontinuities due to eigenvectors sign
    S=sign(diag(coeffOld'*coeff));
    coeff=coeff*diag(S);
end
end


function [QNew,mNew,SqSumNew]=UpdateQ(QOld,xNew,n,mOld,SqSumOld)
% UPDATEQ updates the non-normalized covariance matrix for the dataset X,
% with a new sample XNEW. The non normalized covariance matrix is the
% covariance matrix multiplied by N-1, where N is the number of samples.
%
% [QNEW]=UpdateQ(QOLD,MOLD,N,XNEW)
% 
% QNEW is the new non-normalized covariance matrix and MNEW the row vector
% of data series means. Both values are used for incremental update.
% XNEW and MOLD are row vectors.
%
% Vittorio Lippi* & Giacomo Ceccarelli^ 
% December 2015
% *Uniklinik Freiburg ^University of Pisa + INFN
%
%

% update the mean and updates the sum of squared samples
if nargin<4
  QNew=QOld+ xNew'*xNew;
  return
end

mNew=mOld*(n/(n+1))+xNew/(n+1);

if nargin<5
  size(xNew)
  size(mNew)
   x=(xNew - mNew);
  
  d= mOld-mNew;
  QNew=QOld+(n*(d'*d))+ x'*x;
  return
end

sigmaOld=sqrt((SqSumOld-(n)*(mOld.^2))/(n-1));
SqSumNew=SqSumOld+xNew.^2;
sigmaNew=sqrt((SqSumNew-(n+1)*(mNew.^2))/(n));
gamma=diag(1./sigmaNew);
xi=diag(sigmaOld);
R= gamma*xi;
x=(xNew - mNew)*gamma;

% subtract the new mean from the new sample and normalizes for the new STD
x=(xNew - mNew)*gamma;

% d is the row of the difference between the old and the new 
%zero-average sample matrix DELTA. 
d= mOld-mNew;

% Update Q
% QNew=Q+DELTA'DELTA+DELTA'*x+x'*DELTA+x'*x;
% since mean(x) = zeros(1,length(x)) and columns of delta are repetition of
% the same number d(i) then DELTA'*x and x'*DELTA are zeros(length(x)). 
QNew=R*QOld*R+ gamma*(n*(d'*d))*gamma+ x'*x;
end