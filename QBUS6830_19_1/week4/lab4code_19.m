%%  Q1  Stress testing
% I copied and pasted the monthly data into
% Matlab into the variable names 'FFmat' and 'indmat'
% FFmat = [ insert copied data here ];
% indmat=[ insert copied data here ];
% I then saved  them as lab4Q1.mat:

% save lab4Q1.mat
% load lab4Q1


%% Q1(a) Fit  multi-factor CAPM to each industry portfolio excess return series.

% Create Excess Returns
rf=FFmat(:,5);   % risk-free rate
ex_mark_ret=FFmat(:,2);  % excess market return
cnsmr_ret=indmat(:,2); cnsmr_ex_ret=cnsmr_ret-rf;  % excess returns
manuf_ret=indmat(:,3);manuf_ex_ret=manuf_ret-rf; 
hitech_ret=indmat(:,4);hitech_ex_ret=hitech_ret-rf;
health_ret=indmat(:,5);health_ex_ret=health_ret-rf;
other_ret=indmat(:,6);other_ex_ret=other_ret-rf;
ymat=[cnsmr_ex_ret manuf_ex_ret hitech_ex_ret health_ex_ret other_ex_ret];  % rows are observations over time, columns are variables
size(ymat)   % size() command returns the the number of rows and columns of the matrix should be T by n

% FF factors
hml=FFmat(:,4);smb=FFmat(:,3);
xmat=[ones(length(ex_mark_ret),1) ex_mark_ret hml smb];   % creates X matrix for regression with three factors

% OLS regresssions
[Bc,BINTc,Rc,RINTc,STATSc] = regress(cnsmr_ex_ret,xmat);   % runs OLS regression
cnsmr_est=xmat*Bc;   % estimates of average return from SLR
[Bm,BINTm,Rm,RINTm,STATSm] = regress(manuf_ex_ret,xmat);   % runs OLS regression
manuf_est=xmat*Bm;
[Bhi,BINThi,Rhi,RINThi,STATShi] = regress(hitech_ex_ret,xmat);   % runs OLS regression
hitech_est=xmat*Bhi;
[Bhe,BINThe,Rhe,RINThe,STATShe] = regress(health_ex_ret,xmat);   % runs OLS regression
health_est=xmat*Bhe;
[Bo,BINTo,Ro,RINTo,STATSo] = regress(other_ex_ret,xmat);   % runs OLS regression
other_est=xmat*Bo;

% Coefficients and 95% Confidence intervals
[Bc(1) BINTc(1,:) Bc(2) BINTc(2,:) Bc(3) BINTc(3,:) Bc(4) BINTc(4,:);
 Bm(1) BINTm(1,:) Bm(2) BINTm(2,:) Bm(3) BINTm(3,:) Bm(4) BINTm(4,:);
 Bhi(1) BINThi(1,:) Bhi(2) BINThi(2,:) Bhi(3) BINThi(3,:) Bhi(4) BINThi(4,:);
 Bhe(1) BINThe(1,:) Bhe(2) BINThe(2,:) Bhe(3) BINThe(3,:) Bhe(4) BINThe(4,:);
 Bo(1) BINTo(1,:) Bo(2) BINTo(2,:) Bo(3) BINTo(3,:) Bo(4) BINTo(4,:)]

%% Q1(b) and (c) Estimate mean returns under different scenarios for each industry along with VaR and ES
% Note: The Function getstuff() calculates the quantities required for parts (b) and (c).
% See the file "getstuff.m" for details. 'est' is the regression estimate (predicted values) for each scenario.
% VaRG is the VaR assuming a Gaussian distribution, ESG is the Expected Shortfall assuming a Gaussian distribution,
% VaRN is the VaR using Non-parametric methods, ESN is the Expected Shortfall using Non-parametric methods.

% Consumer
[c_est500, c_VaRG500, c_ESG500, c_VaRN500, c_ESN500] = getstuff([1 -5 0 0],Bc,sqrt(STATSc(4)),Rc);
[c_est1000, c_VaRG1000, c_ESG1000, c_VaRN1000, c_ESN1000] = getstuff([1 -10 0 0],Bc,sqrt(STATSc(4)),Rc);
[c_est502, c_VaRG502, c_ESG502, c_VaRN502, c_ESN502] = getstuff([1 -5 0 -2],Bc,sqrt(STATSc(4)),Rc);
[c_est1002, c_VaRG1002, c_ESG1002, c_VaRN1002, c_ESN1002] = getstuff([1 -10 0 -2],Bc,sqrt(STATSc(4)),Rc);
[c_est520, c_VaRG520, c_ESG520, c_VaRN520, c_ESN520] = getstuff([1 -5 -2 0],Bc,sqrt(STATSc(4)),Rc);
[c_est1020, c_VaRG1020, c_ESG1020, c_VaRN1020, c_ESN1020] = getstuff([1 -10 -2 0],Bc,sqrt(STATSc(4)),Rc);
[c_est522, c_VaRG522, c_ESG522, c_VaRN522, c_ESN522] = getstuff([1 -5 -2 -2],Bc,sqrt(STATSc(4)),Rc);
[c_est1022, c_VaRG1022, c_ESG1022, c_VaRN1022, c_ESN1022] = getstuff([1 -10 -2 -2],Bc,sqrt(STATSc(4)),Rc);

% display results in matrix
[c_est500 c_VaRG500 c_ESG500 c_VaRN500 c_ESN500;
 c_est502, c_VaRG502, c_ESG502, c_VaRN502, c_ESN502;
 c_est520, c_VaRG520, c_ESG520, c_VaRN520, c_ESN520;
 c_est522, c_VaRG522, c_ESG522, c_VaRN522, c_ESN522;
 c_est1000, c_VaRG1000, c_ESG1000, c_VaRN1000, c_ESN1000;
 c_est1002, c_VaRG1002, c_ESG1002, c_VaRN1002, c_ESN1002;
 c_est1020, c_VaRG1020, c_ESG1020, c_VaRN1020, c_ESN1020;
 c_est1022, c_VaRG1022, c_ESG1022, c_VaRN1022, c_ESN1022]

% Histogram of residuals
figure;hist(Rc,25);title('Consumer residuals');
% Skewness and Kurtosis of residuals
[skewness(Rc) kurtosis(Rc)]
% JB test for normality.
[h,p]=jbtest(Rc)   

%Manufacturing
[ma_est500, ma_VaRG500, ma_ESG500, ma_VaRN500, ma_ESN500] = getstuff([1 -5 0 0],Bm,sqrt(STATSm(4)),Rm);
[ma_est1000, ma_VaRG1000, ma_ESG1000, ma_VaRN1000, ma_ESN1000] = getstuff([1 -10 0 0],Bm,sqrt(STATSm(4)),Rm);
[ma_est502, ma_VaRG502, ma_ESG502, ma_VaRN502, ma_ESN502] = getstuff([1 -5 0 -2],Bm,sqrt(STATSm(4)),Rm);
[ma_est1002, ma_VaRG1002, ma_ESG1002, ma_VaRN1002, ma_ESN1002] = getstuff([1 -10 0 -2],Bm,sqrt(STATSm(4)),Rm);
[ma_est520, ma_VaRG520, ma_ESG520, ma_VaRN520, ma_ESN520] = getstuff([1 -5 -2 0],Bm,sqrt(STATSm(4)),Rm);
[ma_est1020, ma_VaRG1020, ma_ESG1020, ma_VaRN1020, ma_ESN1020] = getstuff([1 -10 -2 0],Bm,sqrt(STATSm(4)),Rm);
[ma_est522, ma_VaRG522, ma_ESG522, ma_VaRN522, ma_ESN522] = getstuff([1 -5 -2 -2],Bm,sqrt(STATSm(4)),Rm);
[ma_est1022, ma_VaRG1022, ma_ESG1022, ma_VaRN1022, ma_ESN1022] = getstuff([1 -10 -2 -2],Bm,sqrt(STATSm(4)),Rm);
[ma_est500 ma_VaRG500 ma_ESG500 ma_VaRN500 ma_ESN500;
 ma_est502, ma_VaRG502, ma_ESG502, ma_VaRN502, ma_ESN502;
 ma_est520, ma_VaRG520, ma_ESG520, ma_VaRN520, ma_ESN520;
 ma_est522, ma_VaRG522, ma_ESG522, ma_VaRN522, ma_ESN522;
 ma_est1000, ma_VaRG1000, ma_ESG1000, ma_VaRN1000, ma_ESN1000;
 ma_est1002, ma_VaRG1002, ma_ESG1002, ma_VaRN1002, ma_ESN1002;
 ma_est1020, ma_VaRG1020, ma_ESG1020, ma_VaRN1020, ma_ESN1020;
 ma_est1022, ma_VaRG1022, ma_ESG1022, ma_VaRN1022, ma_ESN1022]

figure;hist(Rm,25);title('Manufacturing residuals');
[skewness(Rm) kurtosis(Rm)]
[h,p]=jbtest(Rm)   % JB test for normality.

%Hi_tech
[hi_est500, hi_VaRG500, hi_ESG500, hi_VaRN500, hi_ESN500] = getstuff([1 -5 0 0],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est1000, hi_VaRG1000, hi_ESG1000, hi_VaRN1000, hi_ESN1000] = getstuff([1 -10 0 0],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est502, hi_VaRG502, hi_ESG502, hi_VaRN502, hi_ESN502] = getstuff([1 -5 0 -2],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est1002, hi_VaRG1002, hi_ESG1002, hi_VaRN1002, hi_ESN1002] = getstuff([1 -10 0 -2],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est520, hi_VaRG520, hi_ESG520, hi_VaRN520, hi_ESN520] = getstuff([1 -5 -2 0],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est1020, hi_VaRG1020, hi_ESG1020, hi_VaRN1020, hi_ESN1020] = getstuff([1 -10 -2 0],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est522, hi_VaRG522, hi_ESG522, hi_VaRN522, hi_ESN522] = getstuff([1 -5 -2 -2],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est1022, hi_VaRG1022, hi_ESG1022, hi_VaRN1022, hi_ESN1022] = getstuff([1 -10 -2 -2],Bhi,sqrt(STATShi(4)),Rhi);
[hi_est500 hi_VaRG500 hi_ESG500 hi_VaRN500 hi_ESN500;
 hi_est502, hi_VaRG502, hi_ESG502, hi_VaRN502, hi_ESN502;
 hi_est520, hi_VaRG520, hi_ESG520, hi_VaRN520, hi_ESN520;
 hi_est522, hi_VaRG522, hi_ESG522, hi_VaRN522, hi_ESN522;
 hi_est1000, hi_VaRG1000, hi_ESG1000, hi_VaRN1000, hi_ESN1000;
 hi_est1002, hi_VaRG1002, hi_ESG1002, hi_VaRN1002, hi_ESN1002;
 hi_est1020, hi_VaRG1020, hi_ESG1020, hi_VaRN1020, hi_ESN1020;
 hi_est1022, hi_VaRG1022, hi_ESG1022, hi_VaRN1022, hi_ESN1022]

figure;hist(Rhi,25);title('Hitech residuals');
[skewness(Rhi) kurtosis(Rhi)]
[h,p]=jbtest(Rhi)   % JB test for normality.

%Health
[he_est500, he_VaRG500, he_ESG500, he_VaRN500, he_ESN500] = getstuff([1 -5 0 0],Bhe,sqrt(STATShe(4)),Rhe);
[he_est1000, he_VaRG1000, he_ESG1000, he_VaRN1000, he_ESN1000] = getstuff([1 -10 0 0],Bhe,sqrt(STATShe(4)),Rhe);
[he_est502, he_VaRG502, he_ESG502, he_VaRN502, he_ESN502] = getstuff([1 -5 0 -2],Bhe,sqrt(STATShe(4)),Rhe);
[he_est1002, he_VaRG1002, he_ESG1002, he_VaRN1002, he_ESN1002] = getstuff([1 -10 0 -2],Bhe,sqrt(STATShe(4)),Rhe);
[he_est520, he_VaRG520, he_ESG520, he_VaRN520, he_ESN520] = getstuff([1 -5 -2 0],Bhe,sqrt(STATShe(4)),Rhe);
[he_est1020, he_VaRG1020, he_ESG1020, he_VaRN1020, he_ESN1020] = getstuff([1 -10 -2 0],Bhe,sqrt(STATShe(4)),Rhe);
[he_est522, he_VaRG522, he_ESG522, he_VaRN522, he_ESN522] = getstuff([1 -5 -2 -2],Bhe,sqrt(STATShe(4)),Rhe);
[he_est1022, he_VaRG1022, he_ESG1022, he_VaRN1022, he_ESN1022] = getstuff([1 -10 -2 -2],Bhe,sqrt(STATShe(4)),Rhe);
[he_est500 he_VaRG500 he_ESG500 he_VaRN500 he_ESN500;
 he_est502, he_VaRG502, he_ESG502, he_VaRN502, he_ESN502;
 he_est520, he_VaRG520, he_ESG520, he_VaRN520, he_ESN520;
 he_est522, he_VaRG522, he_ESG522, he_VaRN522, he_ESN522;
 he_est1000, he_VaRG1000, he_ESG1000, he_VaRN1000, he_ESN1000;
 he_est1002, he_VaRG1002, he_ESG1002, he_VaRN1002, he_ESN1002;
 he_est1020, he_VaRG1020, he_ESG1020, he_VaRN1020, he_ESN1020;
 he_est1022, he_VaRG1022, he_ESG1022, he_VaRN1022, he_ESN1022]

figure;hist(Rhe,25);title('Health residuals');
[skewness(Rhe) kurtosis(Rhe)]
[h,p]=jbtest(Rhe)   % JB test for normality.

%Other
[o_est500, o_VaRG500, o_ESG500, o_VaRN500, o_ESN500] = getstuff([1 -5 0 0],Bo,sqrt(STATSo(4)),Ro);
[o_est1000, o_VaRG1000, o_ESG1000, o_VaRN1000, o_ESN1000] = getstuff([1 -10 0 0],Bo,sqrt(STATSo(4)),Ro);
[o_est502, o_VaRG502, o_ESG502, o_VaRN502, o_ESN502] = getstuff([1 -5 0 -2],Bo,sqrt(STATSo(4)),Ro);
[o_est1002, o_VaRG1002, o_ESG1002, o_VaRN1002, o_ESN1002] = getstuff([1 -10 0 -2],Bo,sqrt(STATSo(4)),Ro);
[o_est520, o_VaRG520, o_ESG520, o_VaRN520, o_ESN520] = getstuff([1 -5 -2 0],Bo,sqrt(STATSo(4)),Ro);
[o_est1020, o_VaRG1020, o_ESG1020, o_VaRN1020, o_ESN1020] = getstuff([1 -10 -2 0],Bo,sqrt(STATSo(4)),Ro);
[o_est522, o_VaRG522, o_ESG522, o_VaRN522, o_ESN522] = getstuff([1 -5 -2 -2],Bo,sqrt(STATSo(4)),Ro);
[o_est1022, o_VaRG1022, o_ESG1022, o_VaRN1022, o_ESN1022] = getstuff([1 -10 -2 -2],Bo,sqrt(STATSo(4)),Ro);
[o_est500 o_VaRG500 o_ESG500 o_VaRN500 o_ESN500;
 o_est502, o_VaRG502, o_ESG502, o_VaRN502, o_ESN502;
 o_est520, o_VaRG520, o_ESG520, o_VaRN520, o_ESN520;
 o_est522, o_VaRG522, o_ESG522, o_VaRN522, o_ESN522;
 o_est1000, o_VaRG1000, o_ESG1000, o_VaRN1000, o_ESN1000;
 o_est1002, o_VaRG1002, o_ESG1002, o_VaRN1002, o_ESN1002;
 o_est1020, o_VaRG1020, o_ESG1020, o_VaRN1020, o_ESN1020;
 o_est1022, o_VaRG1022, o_ESG1022, o_VaRN1022, o_ESN1022]

figure;hist(Ro,25);title('Other residuals');
[skewness(Ro) kurtosis(Ro)]
[h,p]=jbtest(Ro)   % JB test for normality.

% SERs for industry regressions for part (b)
[sqrt(STATSc(4)) sqrt(STATSm(4)) sqrt(STATShi(4)) sqrt(STATShe(4)) sqrt(STATSo(4))]

% mean estimates for part (b)
[c_est500 ma_est500 hi_est500 he_est500 o_est500;
 c_est502 ma_est502 hi_est502 he_est502 o_est502;
 c_est520 ma_est520 hi_est520 he_est520 o_est520;
 c_est522 ma_est522 hi_est522 he_est522 o_est522;
 c_est1000 ma_est1000 hi_est1000 he_est1000 o_est1000;
 c_est1002 ma_est1002 hi_est1002 he_est1002 o_est1002;
 c_est1020 ma_est1020 hi_est1020 he_est1020 o_est1020;
 c_est1022 ma_est1022 hi_est1022 he_est1022 o_est1022]

m_N500=length(ex_mark_ret(ex_mark_ret<-5));
m_pN500=length(ex_mark_ret(ex_mark_ret<-5))/length(ex_mark_ret);
m_N1000=length(ex_mark_ret(ex_mark_ret<-10));
m_pN1000=length(ex_mark_ret(ex_mark_ret<-10))/length(ex_mark_ret);
[m_N500 m_pN500;
 m_N1000 m_pN1000]

%% Q1(e) Count the number of times loses were greater than VaRs and market returns <-5,-10 in actual data series
c_N500=sum(cnsmr_ex_ret<c_VaRN500 & ex_mark_ret<-5);
c_pN500=sum(cnsmr_ex_ret<c_VaRN500 & ex_mark_ret<-5)/sum(ex_mark_ret<-5);
c_mN500=mean(cnsmr_ex_ret(cnsmr_ex_ret<c_VaRN500 & ex_mark_ret<-5));

% histograms of Excess returns lower than estimated VaR
figure;hist(cnsmr_ex_ret(cnsmr_ex_ret<c_VaRN500 & ex_mark_ret<-5));
title('Consumer Excess returns lower than 1% VaR - mkt drops 5%');

c_N1000=sum(cnsmr_ex_ret<c_VaRN1000 & ex_mark_ret<-10);
c_pN1000=sum(cnsmr_ex_ret<c_VaRN1000 & ex_mark_ret<-10)/sum(ex_mark_ret<-10);
c_mN1000=mean(cnsmr_ex_ret(cnsmr_ex_ret<c_VaRN1000 & ex_mark_ret<-10));

figure;hist(cnsmr_ex_ret(cnsmr_ex_ret<c_VaRN1000 & ex_mark_ret<-10));
title('Consumer Excess returns lower than 1% VaR - mkt drops 10%');

ma_N500=sum(manuf_ex_ret<ma_VaRN500 & ex_mark_ret<-5);
ma_pN500=sum(manuf_ex_ret<ma_VaRN500 & ex_mark_ret<-5)/sum(ex_mark_ret<-5);
ma_mN500=mean(manuf_ex_ret(manuf_ex_ret<ma_VaRN500 & ex_mark_ret<-5));
ma_N1000=sum(manuf_ex_ret<ma_VaRN1000 & ex_mark_ret<-10);
ma_pN1000=sum(manuf_ex_ret<ma_VaRN1000 & ex_mark_ret<-10)/sum(ex_mark_ret<-10);
ma_mN1000=mean(manuf_ex_ret(manuf_ex_ret<ma_VaRN1000 & ex_mark_ret<-10));
hi_N500=sum(hitech_ex_ret<hi_VaRN500 & ex_mark_ret<-5);
hi_pN500=sum(hitech_ex_ret<hi_VaRN500 & ex_mark_ret<-5)/sum(ex_mark_ret<-5);
hi_mN500=mean(hitech_ex_ret(hitech_ex_ret<hi_VaRN500 & ex_mark_ret<-5));
hi_N1000=sum(hitech_ex_ret<hi_VaRN1000 & ex_mark_ret<-10);
hi_pN1000=sum(hitech_ex_ret<hi_VaRN1000 & ex_mark_ret<-10)/sum(ex_mark_ret<-10);
hi_mN1000=mean(hitech_ex_ret(hitech_ex_ret<hi_VaRN1000 & ex_mark_ret<-10));
he_N500=sum(health_ex_ret<he_VaRN500 & ex_mark_ret<-5);
he_pN500=sum(health_ex_ret<he_VaRN500 & ex_mark_ret<-5)/sum(ex_mark_ret<-5);
he_mN500=mean(health_ex_ret(health_ex_ret<he_VaRN500 & ex_mark_ret<-5));
he_N1000=sum(health_ex_ret<he_VaRN1000 & ex_mark_ret<-10);
he_pN1000=sum(health_ex_ret<he_VaRN1000 & ex_mark_ret<-10)/sum(ex_mark_ret<-10);
he_mN1000=mean(health_ex_ret(health_ex_ret<he_VaRN1000 & ex_mark_ret<-10));
o_N500=sum(other_ex_ret<o_VaRN500 & ex_mark_ret<-5);
o_pN500=sum(other_ex_ret<o_VaRN500 & ex_mark_ret<-5)/sum(ex_mark_ret<-5);
o_mN500=mean(other_ex_ret(other_ex_ret<o_VaRN500 & ex_mark_ret<-5));
o_N1000=sum(other_ex_ret<o_VaRN1000 & ex_mark_ret<-10);
o_pN1000=sum(other_ex_ret<o_VaRN1000 & ex_mark_ret<-10)/sum(ex_mark_ret<-10);
o_mN1000=mean(other_ex_ret(other_ex_ret<o_VaRN1000 & ex_mark_ret<-10));

[c_N500 c_pN500 c_pN500*m_pN500 c_mN500 c_N1000 c_pN1000 c_pN1000*m_pN1000 c_mN1000;
 ma_N500 ma_pN500 ma_pN500*m_pN500 ma_mN500 ma_N1000 ma_pN1000 ma_pN1000*m_pN1000 ma_mN1000;
 hi_N500 hi_pN500 hi_pN500*m_pN500 hi_mN500 hi_N1000 hi_pN1000 hi_pN1000*m_pN1000 hi_mN1000;
 he_N500 he_pN500 he_pN500*m_pN500 he_mN500 he_N1000 he_pN1000 he_pN1000*m_pN1000 he_mN1000;
 o_N500 o_pN500 o_pN500*m_pN500 o_mN500 o_N1000 o_pN1000 o_pN1000*m_pN1000 o_mN1000]

%clear; % clear workspace 

%% Q2 PCA and Factor Modelling
% I copied the data from Tsay_FM_data.txt
% Tsay_data = [ <insert copied data here> ];
% save Tsay_data.mat
% load('Tsay_data.mat'); % Equivalent command to "load Tsay_data.mat" or simply "load Tsay_data" 

%% Q2(a) Correlation matrix

% plot each series on the one graph
figure;plot(Tsay_data);title('Stock returns');
legend('IBM','HPQ','INTC','JPM','BAC','Location','SouthWest');
xlim([ 1 length(Tsay_data)]);

% Correlation matrix
corr(Tsay_data)    % estimates the sample correlation matrix, showing all pairwise correlations.

% Plot of JPM vs BAC
figure;plot(Tsay_data(:,4),Tsay_data(:,5),'+');title('JPM vs BAC');
xlabel('BAC');ylabel('JPM');

%% Q2(b) PCA analysis
% Conduct a principle component analysis of the 5 stock returns in Tsay's
% data set 

[pc_ret,score_ret,latent_ret] = princomp(Tsay_data); % score are linearly transformed dataset
pc_ret

% Lambdas, i.e. % of total variance explained per componenet and cumulative variance explained per componenet 
[latent_ret latent_ret./sum(latent_ret) cumsum(latent_ret)./sum(latent_ret)]

% Plot 3 componenents below individual stock returns
figure;subplot(4,1,1);plot(Tsay_data);
title('Stock returns');xlim([1 length(Tsay_data)]);
subplot(4,1,2);plot(score_ret(:,1));
title('1st Principle component');xlim([1 length(score_ret)]);
subplot(4,1,3);plot(score_ret(:,2));
title('2nd Principle component');xlim([1 length(score_ret)]);
subplot(4,1,4);plot(score_ret(:,3));
title('3rd Principle component');xlim([1 length(score_ret)]);


% The following demonstrates that the components are uncorrelated
corr(score_ret)

% plot all stock returns in one plot and all components in a second plot
% below the stock returns
figure;subplot(2,1,1);plot(Tsay_data);
title('Stock returns');xlim([1 length(Tsay_data)]);
legend('IBM','HPQ','INTC','JPM','BAC','Location','SouthWest','Orientation','horizontal');
subplot(2,1,2);plot(score_ret);
title('Principle components');xlim([1 length(score_ret)]);
legend('PC1','PC2','PC3','PC4','PC5','Location','SouthWest','Orientation','horizontal');

% Plot average returns against first componenet
figure;plot(mean(Tsay_data'),score_ret(:,1),'+');title('Average return vs PC1');

% Plot average returns with first componenet
figure;plot(mean(Tsay_data'));
hold on;plot(score_ret(:,1),'r');title('Average return with PC1');
legend('Average rtn','PC1');xlim([1 length(Tsay_data)]);

% Calculate correlation of first componenet with average returns
corrcoef(mean(Tsay_data'),score_ret(:,1));

% Bi-plots for components
figure;biplot(pc_ret(:,1:2));
figure;biplot(pc_ret(:,1:3));