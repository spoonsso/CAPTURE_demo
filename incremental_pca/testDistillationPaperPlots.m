% Plots figures from: 
close all
load('distillationtower');
tic
%incremental
T=10000;
W=10;
t=1:T;
data=distillationtower;

[s,N]=size(data);
T=s;
%data=data+0.1*rand(T,N);
dw=(data);
%dd=dw-repmat(mean(dw,1),size(dw,1),1);
dd=zscore(dw);
[COEFF, SCORE, LATENT, TSQUARED] = pca(dd);
COV=cov(dd);

WS=nan(T,N);
WCN=nan(T,2+N);
HLATENT=NaN(T,N);
deg=nan(1,N);

step=1; %keep it!!
fMEAN=mean(data);
fVAR=var(data);
toc
disp('init done');

tic
dw=data(1:W,:);
%dd=zscore(dw);
sqs=sum(dw.^2);
m=mean(dw);
%dd=(dw-repmat(mean(dw,1),size(dw,1),1));
dd=zscore(dw);
[wCOEFF, wSCORE, wLATENT] = pca(dd);
Q=dd'*dd;

n=W;

for k=W+1:step:T
    k
    test=data(1:k,:);
    
    %oCOEFF=wCOEFF; %toggle comment for continuity
    oCOEFF=[];      %toggle comment for continuity
    
    oLATENT=wLATENT;
    xNew=data(k-step+1:k,:);
    
    [wCOEFF,wLATENT,Q,m,sqs]=IncrementalPCA(Q,n,xNew,oCOEFF,oLATENT,m,sqs);%RealTimePCA(Q,m,sqs,n,xNew,oCOEFF);
    qt=cov(test);
    wCOV=Q/n; %n+1 samples
    %pause;
    
    wSigma=sqrt((sqs-(n+1)*(m.^2))/(n));
    wSCORE=((xNew-m)./wSigma)*wCOEFF;
    
    n=n+1;
    WS(k-step+1:k,:)=wSCORE;
    HLATENT(k-step+1:k,:)=repmat(wLATENT',step,1);
    
    WCN(k-1,1)=norm(wCOV-COV,'fro');%mdist(wCOEFF,COEFF);
    WCN(k-1,2)=mdist2(wCOEFF,COEFF,wLATENT,LATENT);%norm(LATENT-wLATENT);%mdist(wCOEFF,COEFF);
    
    for i=1:N
        WCN(k-1,2+i)=(sum(abs(wLATENT-wLATENT(i))<1/sqrt(N)));
    end
    
end
disp([num2str(k),' samples done']);
disp('done.');
toc

%black and white respectable plots

MyStyles = {'-';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.';'-';':';'--';'-.'};

figure(40)
subplot(4,1,1);
h=plot(SCORE);
%set(h,{'LineStyle'},MyStyles(1:27));
ylabel('Score')
axis([1 T min(min(SCORE))-1 max(max(SCORE))+5])
subplot(4,1,2);
h=plot(WS);
%set(h,{'LineStyle'},MyStyles(1:27));
axis([1 T min(min(SCORE))-1 max(max(SCORE))+5])
ylabel('Score')
subplot(4,1,3);
semilogy(WCN(:,2));
plot(WCN(:,1));
ylabel('||COV-wCOV||')
axis([1 T min(min(WCN(:,1))) max(max(WCN(:,1)))+1])
subplot(4,1,4);
plot(HLATENT);
axis([1 T min(min(HLATENT))-1 max(max(HLATENT))+1])

xlabel('samples');
WCNrt=WCN;

figure(50)
subplot(2,1,1);
h=plot(SCORE(:,1:2));
%set(h,{'LineStyle'},MyStyles(1:2));
ylabel('Score');
legend('PC 1','PC 2')
axis([1 T min(min(SCORE))-1 max(max(SCORE))+5])
subplot(2,1,2);
h=plot(WS(:,1:2));
legend('PC 1','PC 2')
%set(h,{'LineStyle'},MyStyles(1:2));
axis([1 T min(min(SCORE))-1 max(max(SCORE))+5])
ylabel('Score');

xlabel('Samples');

figure(60)
idx=[1:2]+8;
 subplot(2,1,1);
h=plot(100:T,WS(100:end,idx));
WSd=WS;
%set(h,{'LineStyle'},MyStyles(1:2));
legend('PC 9','PC 10')
axis([100 T min(min(SCORE(100:end,idx)))-1 max(max(SCORE(100:end,idx)))+1])
ylabel('Score');

 subplot(2,1,2);
 
for k=W+1:step:T
    test=data(1:k,:);
    
    oCOEFF=wCOEFF; %toggle comment for continuity
    %oCOEFF=[];      %toggle comment for continuity
    
    oLATENT=wLATENT;
    xNew=data(k-step+1:k,:);
    
    [wCOEFF,wLATENT,Q,m,sqs]=IncrementalPCA(Q,n,xNew,oCOEFF,oLATENT,m,sqs);%RealTimePCA(Q,m,sqs,n,xNew,oCOEFF);
    qt=cov(test);
    wCOV=Q/n; %n+1 samples
    %pause;
    
    wSigma=sqrt((sqs-(n+1)*(m.^2))/(n));
    wSCORE=((xNew-m)./wSigma)*wCOEFF;
    
    n=n+1;
    WS(k-step+1:k,:)=wSCORE;
    HLATENT(k-step+1:k,:)=repmat(wLATENT',step,1);
    
    WCN(k-1,1)=norm(wCOV-COV,'fro');%mdist(wCOEFF,COEFF);
    WCN(k-1,2)=mdist2(wCOEFF,COEFF,wLATENT,LATENT);%norm(LATENT-wLATENT);%mdist(wCOEFF,COEFF);
    
    for i=1:N
        WCN(k-1,2+i)=(sum(abs(wLATENT-wLATENT(i))<1/sqrt(N)));
    end
    
end
 
h=plot(100:T,WS(100:end,idx));
%set(h,{'LineStyle'},MyStyles(1:2));
legend('PC 9','PC 10')
axis([100 T min(min(SCORE(100:end,idx)))-1 max(max(SCORE(100:end,idx)))+1])
ylabel('Score');
xlabel('Samples');


 
figure(70)
%[bCOEFF, bSCORE, bLATENT, bTSQUARED] = princomp(WS);

hLATENT=HLATENT(end,:);

iLATENT=var(WS((W+1):end,:))';
%barcont=discLATENT/sum(iLATENT)*100;
barcont=iLATENT/sum(iLATENT)*100;
barbatch=LATENT/sum(LATENT)*100;
bar([barcont,barbatch],1)
hold on
 plot(cumsum(iLATENT/sum(iLATENT)*100),'b')
 plot(cumsum(LATENT/sum(LATENT)*100),'y')
legend('On-line PCA with continuity correction','Batch PCA','cumulative Var. On-line','cumulative Var. Batch')

ylabel('Var[%]');
xlabel('PC');