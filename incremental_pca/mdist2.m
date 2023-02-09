function [d]=mdist2(U,V,lambdau,lambdav)
% pseudodistance
% invariant for parity
% pseudodist
N=length(lambdau);
d=N;
degu=zeros(N,1);
degv=zeros(N,1);
for i=1:N
     degu(i)=(find(abs(lambdau-lambdau(i))<1/sqrt(N),1,'first'));
     degv(i)=(find(abs(lambdav-lambdav(i))<1/sqrt(N),1,'first'));
%        if degu(i)~=degv(i)
%            d=NaN;
%            return
%       end
end

for i=1:N
    iu=degu==degu(i);
%    iv=degv==degv(i);
  p=((U(:,iu)'*V(:,iu)).^2)./sum(iu);
  d=d-sum(sum(p));
end
