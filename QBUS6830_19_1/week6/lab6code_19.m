%% Lab Sheet 6: Forecasting and forecast accuracy 
% Import "Tsay_FM_data.txt" as column vectors using the names in the first row for
% the created vectors
% save lab6.mat
% load lab6.mat

%% Q1 Forecasting stock returns

%% Q1(a) Conduct exploratory data analysis (EDA)

Tsay_data=[IBM HPQ INTC JPM BAC];

% Summary Stats
[mean(Tsay_data); median(Tsay_data); std(Tsay_data); min(Tsay_data); max(Tsay_data); skewness(Tsay_data); kurtosis(Tsay_data)]

% Plot returns data
figure;plot(Tsay_data);title('Stock Returns for 5 assets');
legend('IBM','HPQ','INTC','JPM','BAC','Location','SouthWest');
xlim([0,length(Tsay_data)]);

% Plot histograms of returns
figure;subplot(3,2,1);hist(IBM,25);title('Histogram of IBM returns');
subplot(3,2,2);hist(HPQ,25);title('Histogram of HPQ returns');
subplot(3,2,3);hist(INTC,25);title('Histogram of INTC returns');
subplot(3,2,4);hist(JPM,25);title('Histogram of JPM returns');
subplot(3,2,5);hist(BAC,25);title('Histogram of BAC returns');

%% Q1(b) Forecast last 24 months of returns using a variety of methods

% Setup forecast and in-sample data
ret_f = Tsay_data(end-23:end,:); % forecast sample data is last 24 months.
ret_is = Tsay_data(1:end-24,:);  % in-sample is all days before the last 24 months.

% Non-parametrics forecasts (all based on averages)
ret_f1 = mean(ret_is);   % first forecast is long-run mean of in-sample period for each asset
ret_f2 = mean(ret_is(end-2:end,:));   % second forecast is mean of last 3 months of in-sample period
ret_f3 = mean(ret_is(end-11:end,:));   % third forecast is mean of last 12 months of in-sample period
ret_f4 = mean(ret_is(end-23:end,:));   % fourth forecast is mean of last 24 months of in-sample period
ret_f5 = ret_is(end,:);                % fifth forecast is return of the last month of in-sample period
ret_f6 = ones(1,5)*mean(mean(ret_is(end-5:end,:))); % sixth forecast is mean of last 6 month's returns over all 5 assets

% Parametric forecast (based on ARMA model)
ret_f7 = zeros(1,5);  %setting up space for ARMA model forecasts

% 7th method forecasts using an ARMA model chosen for each series.

% Plot IBM series and ACF
figure;subplot(2,1,1);plot(ret_is(:,1));title('IBM returns');xlim([0,length(ret_is)]);
subplot(2,1,2);autocorr(ret_is(:,1), 25);
% LB test for AR effects on IBM
[H5, pval5, Qs5, CV5] = lbqtest(ret_is(:,1), 5, 0.05);
[H10, pval10, Qs10, CV10] = lbqtest(ret_is(:,1), 10, 0.05);
[pval5 pval10]
% I choose AR(4) for IBM, now fit it and forecast
Mdl=arima(4,0,0);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is(:,1));
[ret_f7(1), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,1));% 1-period forecasts

% Plot HPQ series and ACF
figure; subplot(2,1,1);plot(ret_is(:,2));title('HPQ returns');xlim([0,length(ret_is)]);
subplot(2,1,2);autocorr(ret_is(:,2), 25);figure(gcf)
% LB test for AR effects on HPQ
[H5, pval5, Qs5, CV5] = lbqtest(ret_is(:,2), 5, 0.05);
[H10, pval10, Qs10, CV10] = lbqtest(ret_is(:,2), 10, 0.05);
[pval5 pval10]
% I choose AR(7) for HPQ, now fit it and forecast
Mdl=arima(7,0,0);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is(:,2));
[ret_f7(2), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,2));% 1-period forecasts

% Plot INTC series and ACF
figure;subplot(2,1,1);plot(ret_is(:,3));title('INTC returns');xlim([0,length(ret_is)]);
subplot(2,1,2);autocorr(ret_is(:,3), 25);figure(gcf)
% LB test for AR effects on INTC
[H5, pval5, Qs5, CV5] = lbqtest(ret_is(:,3), 5, 0.05);
[H10, pval10, Qs10, CV10] = lbqtest(ret_is(:,3), 10, 0.05);
[pval5 pval10]
% I choose ARMA(1,1) for INTC, now fit it and forecast
Mdl=arima(1,0,1);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is(:,3));
[ret_f7(3), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,3));% 1-period forecasts

% Plot JPM series and ACF
figure;subplot(2,1,1);plot(ret_is(:,4));title('JPM returns');xlim([0,length(ret_is)]);
subplot(2,1,2);autocorr(ret_is(:,4), 25);figure(gcf)
% LB test for AR effects on JPM
[H5, pval5, Qs5, CV5] = lbqtest(ret_is(:,4), 5, 0.05);
[H10, pval10, Qs10, CV10] = lbqtest(ret_is(:,4), 10, 0.05);
[pval5 pval10]
% I choose AR(4) for JPM, now fit it and forecast
Mdl=arima(4,0,0);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is(:,4));
[ret_f7(4), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,4));% 1-period forecasts

% Plot BAC series and ACF
figure; subplot(2,1,1);plot(ret_is(:,5));title('BAC returns');xlim([0,length(ret_is)]);
subplot(2,1,2);autocorr(ret_is(:,5), 25);figure(gcf)
% LB test for AR effects on BAC
[H5, pval5, Qs5, CV5] = lbqtest(ret_is(:,5), 5, 0.05);
[H10, pval10, Qs10, CV10] = lbqtest(ret_is(:,5), 10, 0.05);
[pval5 pval10]
% I choose AR(9) for BAC, now fit it and forecast
Mdl=arima(9,0,0);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl,ret_is(:,5));
[ret_f7(5), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,5));% 1-period forecasts

% Note that these are all vectors, representing the forecasts for all 5 series.
% These forecasts should be compared with ret_f(1,:), ie actual return on day 1 of forecast period 

% Create a loop to create forecasts for 24 days in forecast sample 
% (Loop starts at 2nd day of forecast sample as day one already calculated
% above)
for t=2:24
  ret_is = Tsay_data(t:end-25+t,:); % you need to update in-sample data to continuously perform forecast
  ret_f1(t,:) = mean(ret_is);
  ret_f2(t,:) = mean(ret_is(end-2:end,:));
  ret_f3(t,:) = mean(ret_is(end-11:end,:)); 
  ret_f4(t,:) = mean(ret_is(end-23:end,:)); 
  ret_f5(t,:) = ret_is(end,:);   
  ret_f6(t,:) = ones(1,5)*mean(mean(ret_is(end-5:end,:))); % mean of last 6 months across all 5 assets
  
  %7th method forecasts are from an ARMA model chosen for each series.
  Mdl=arima(4,0,0);
  EstMdl = estimate(Mdl,ret_is(:,1),'Display','Off'); % Note: I suppress the display of estimation results here
  [ret_f7(t,1), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,1)); % 1-period forecasts

  Mdl=arima(7,0,0);
  EstMdl = estimate(Mdl,ret_is(:,2),'Display','Off');
  [ret_f7(t,2), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,2)); % 1-period forecasts

  Mdl=arima(1,0,1);
  EstMdl = estimate(Mdl,ret_is(:,3),'Display','Off');
  [ret_f7(t,3), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,3)); % 1-period forecasts

  Mdl=arima(4,0,0);
  EstMdl = estimate(Mdl,ret_is(:,4),'Display','Off');
  [ret_f7(t,4), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,4)); % 1-period forecasts

  Mdl=arima(9,0,0);
  EstMdl = estimate(Mdl,ret_is(:,5),'Display','Off');
  [ret_f7(t,5), FMSE] = forecast(EstMdl,1,'Y0',ret_is(:,5)); % 1-period forecasts
end    
% we now have 24 sets of forecasts one-step-ahead for all five series using each of 7 different methods.

%% Q1(c) Assess accuracy of forecasts
% Plot forecasts and measure forecast accuracy for each stock separately
% and calculate forecast accuracy measures
% Note that MAPE is undefined if return is zero (can't divide by zero) so 
% won't be used here (getfa()attempts to calculate it and gets an error) 

% IBM
% Plot forecasts from all 7 methods along with actual returns
figure;plot(ret_f(:,1),'g');
hold on;plot(ret_f1(:,1),'k*');plot(ret_f2(:,1),'rd');plot(ret_f3(:,1),'m^');
plot(ret_f4(:,1),'ko');plot(ret_f5(:,1),'m+');plot(ret_f6(:,1),'bp');plot(ret_f7(:,1),'cd')
title('IBM Stock Returns and Forecasts');
legend('Actual return','LR mean','3mth mean','12mth mean','24mth mean','Naive','6mth All','ARMA','Location','SouthWest');

% Calculate RMSE, MAD, and MAP for all forecast methods using getfa()
[rmse1, mad1, map1]=getfa(ret_f(:,1),ret_f1(:,1));
[rmse2, mad2, map2]=getfa(ret_f(:,1),ret_f2(:,1));
[rmse3, mad3, map3]=getfa(ret_f(:,1),ret_f3(:,1));
[rmse4, mad4, map4]=getfa(ret_f(:,1),ret_f4(:,1));
[rmse5, mad5, map5]=getfa(ret_f(:,1),ret_f5(:,1));
[rmse6, mad6, map6]=getfa(ret_f(:,1),ret_f6(:,1));
[rmse7, mad7, map7]=getfa(ret_f(:,1),ret_f7(:,1));

% Display accuracy measures
[rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7;mad1 mad2 mad3 mad4 mad5 mad6 mad7]

% HPQ
% Plot forecasts from all 7 methods along with actual returns
figure;plot(ret_f(:,2),'g')
hold on;plot(ret_f1(:,2),'k*');plot(ret_f2(:,2),'rd');plot(ret_f3(:,2),'m^');
plot(ret_f4(:,2),'ko');plot(ret_f5(:,2),'m+');plot(ret_f6(:,2),'bp');plot(ret_f7(:,2),'cd')
title('HPQ Stock Returns and Forecasts');
legend('Actual return','LR mean','3mth mean','12mth mean','24mth mean','Naive','6mth All','ARMA','Location','SouthWest');

% Calculate RMSE, MAD, and MAP for all forecast methods using getfa()
[rmse1, mad1, map1]=getfa(ret_f(:,2),ret_f1(:,2));
[rmse2, mad2, map2]=getfa(ret_f(:,2),ret_f2(:,2));
[rmse3, mad3, map3]=getfa(ret_f(:,2),ret_f3(:,2));
[rmse4, mad4, map4]=getfa(ret_f(:,2),ret_f4(:,2));
[rmse5, mad5, map5]=getfa(ret_f(:,2),ret_f5(:,2));
[rmse6, mad6, map6]=getfa(ret_f(:,2),ret_f6(:,2));
[rmse7, mad7, map7]=getfa(ret_f(:,2),ret_f7(:,2));

% Display accuracy measures
[rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7;mad1 mad2 mad3 mad4 mad5 mad6 mad7]

% INTC
% Plot forecasts from all 7 methods along with actual returns
figure;plot(ret_f(:,3),'g')
hold on;plot(ret_f1(:,3),'k*');plot(ret_f2(:,3),'rd');plot(ret_f3(:,3),'m^');
plot(ret_f4(:,3),'ko');plot(ret_f5(:,3),'m+');plot(ret_f6(:,3),'bp');plot(ret_f7(:,3),'cd')
title('INTC Stock Returns and Forecasts');
legend('Actual return','LR mean','3mth mean','12mth mean','24mth mean','Naive','6mth All','ARMA','Location','SouthWest');

% Calculate RMSE, MAD, and MAP for all forecast methods using getfa()
[rmse1, mad1, map1]=getfa(ret_f(:,3),ret_f1(:,3));
[rmse2, mad2, map2]=getfa(ret_f(:,3),ret_f2(:,3));
[rmse3, mad3, map3]=getfa(ret_f(:,3),ret_f3(:,3));
[rmse4, mad4, map4]=getfa(ret_f(:,3),ret_f4(:,3));
[rmse5, mad5, map5]=getfa(ret_f(:,3),ret_f5(:,3));
[rmse6, mad6, map6]=getfa(ret_f(:,3),ret_f6(:,3));
[rmse7, mad7, map7]=getfa(ret_f(:,3),ret_f7(:,3));

% Display accuracy measures
[rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7;mad1 mad2 mad3 mad4 mad5 mad6 mad7]

% JPM
% Plot forecasts from all 7 methods along with actual returns
figure;plot(ret_f(:,4),'g')
hold on;plot(ret_f1(:,4),'k*');plot(ret_f2(:,4),'rd');plot(ret_f3(:,4),'m^');
plot(ret_f4(:,4),'ko');plot(ret_f5(:,4),'m+');plot(ret_f6(:,4),'bp');plot(ret_f7(:,4),'cd')
title('JPM Stock Returns and Forecasts');
legend('Actual return','LR mean','3mth mean','12mth mean','24mth mean','Naive','6mth All','ARMA','Location','SouthWest');

% Calculate RMSE, MAD, and MAP for all forecast methods using getfa()
[rmse1, mad1, map1]=getfa(ret_f(:,4),ret_f1(:,4));
[rmse2, mad2, map2]=getfa(ret_f(:,4),ret_f2(:,4));
[rmse3, mad3, map3]=getfa(ret_f(:,4),ret_f3(:,4));
[rmse4, mad4, map4]=getfa(ret_f(:,4),ret_f4(:,4));
[rmse5, mad5, map5]=getfa(ret_f(:,4),ret_f5(:,4));
[rmse6, mad6, map6]=getfa(ret_f(:,4),ret_f6(:,4));
[rmse7, mad7, map7]=getfa(ret_f(:,4),ret_f7(:,4));

% Display accuracy measures
[rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7;mad1 mad2 mad3 mad4 mad5 mad6 mad7]

% BAC
% Plot forecasts from all 7 methods along with actual returns
figure;plot(ret_f(:,5),'g')
hold on;plot(ret_f1(:,5),'k*');plot(ret_f2(:,5),'rd');plot(ret_f3(:,5),'m^');
plot(ret_f4(:,5),'ko');plot(ret_f5(:,5),'m+');plot(ret_f6(:,5),'bp');plot(ret_f7(:,5),'cd')
title('BAC Stock Returns and Forecasts');
legend('Actual return','LR mean','3mth mean','12mth mean','24mth mean','Naive','6mth All','ARMA','Location','SouthWest');

% Calculate RMSE, MAD, and MAP for all forecast methods using getfa()
[rmse1, mad1, map1]=getfa(ret_f(:,5),ret_f1(:,5));
[rmse2, mad2, map2]=getfa(ret_f(:,5),ret_f2(:,5));
[rmse3, mad3, map3]=getfa(ret_f(:,5),ret_f3(:,5));
[rmse4, mad4, map4]=getfa(ret_f(:,5),ret_f4(:,5));
[rmse5, mad5, map5]=getfa(ret_f(:,5),ret_f5(:,5));
[rmse6, mad6, map6]=getfa(ret_f(:,5),ret_f6(:,5));
[rmse7, mad7, map7]=getfa(ret_f(:,5),ret_f7(:,5));

% Display accuracy measures
[rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7;mad1 mad2 mad3 mad4 mad5 mad6 mad7]

%% Q1(d) Calculate returns for investment strategies using forecast methods

% performance of equally weighted portfolio
ewret=mean(ret_f')'; 

% Other methods that vary with time
for t=1:24
    % choose max predicted return asset
    [rf1,orf1]=sort(ret_f1(t,:)); % sort forecasts from mthd 1 in period t   
    ret_inv1(t) =ret_f(t,orf1(5)); % choose actual return in period t for asset with highest forecast
    [rf2,orf2]=sort(ret_f2(t,:));
    ret_inv2(t) =ret_f(t,orf2(5));
    [rf3,orf3]=sort(ret_f3(t,:));
    ret_inv3(t) =ret_f(t,orf3(5));
    [rf4,orf4]=sort(ret_f4(t,:));
    ret_inv4(t) =ret_f(t,orf4(5));
    [rf5,orf5]=sort(ret_f5(t,:));
    ret_inv5(t) =ret_f(t,orf5(5));
    [rf6,orf6]=sort(ret_f6(t,:));
    ret_inv6(t) =ret_f(t,orf6(5));
    [rf7,orf7]=sort(ret_f7(t,:));
    ret_inv7(t) =ret_f(t,orf7(5));
    
    % weight by predicted returns
    pmwt_r1(t) = sum(ret_f1(t,:).*ret_f(t,:))/sum(ret_f1(t,:));
    pmwt_r2(t) = sum(ret_f2(t,:).*ret_f(t,:))/sum(ret_f2(t,:));
    pmwt_r3(t) = sum(ret_f3(t,:).*ret_f(t,:))/sum(ret_f3(t,:));
    pmwt_r4(t) = sum(ret_f4(t,:).*ret_f(t,:))/sum(ret_f4(t,:));
    pmwt_r5(t) = sum(ret_f5(t,:).*ret_f(t,:))/sum(ret_f5(t,:));
    pmwt_r6(t) = sum(ret_f6(t,:).*ret_f(t,:))/sum(ret_f6(t,:));
    pmwt_r7(t) = sum(ret_f7(t,:).*ret_f(t,:))/sum(ret_f7(t,:));
    
    % weight by inverse absolute predicted returns
    piamwt_r1(t) = sum(ret_f(t,:)./abs(ret_f1(t,:)))/sum(1./abs(ret_f1(t,:)));
    piamwt_r2(t) = sum(ret_f(t,:)./abs(ret_f2(t,:)))/sum(1./abs(ret_f2(t,:)));
    piamwt_r3(t) = sum(ret_f(t,:)./abs(ret_f3(t,:)))/sum(1./abs(ret_f3(t,:)));
    piamwt_r4(t) = sum(ret_f(t,:)./abs(ret_f4(t,:)))/sum(1./abs(ret_f4(t,:)));
    piamwt_r5(t) = sum(ret_f(t,:)./abs(ret_f5(t,:)))/sum(1./abs(ret_f5(t,:)));
    piamwt_r6(t) = sum(ret_f(t,:)./abs(ret_f6(t,:)))/sum(1./abs(ret_f6(t,:)));
    piamwt_r7(t) = sum(ret_f(t,:)./abs(ret_f7(t,:)))/sum(1./abs(ret_f7(t,:)));
end    

% Mean and standard deviation of portfolio returns over forecast period
% using each forecast method and weighting strategy
   [mean(ewret) std(ewret)
    mean(ret_inv1) std(ret_inv1);
    mean(ret_inv2) std(ret_inv2);
    mean(ret_inv3) std(ret_inv3);
    mean(ret_inv4) std(ret_inv4);
    mean(ret_inv5) std(ret_inv5); 
    mean(ret_inv6) std(ret_inv6); 
    mean(ret_inv7) std(ret_inv7);
    mean(pmwt_r1) std(pmwt_r1);
    mean(pmwt_r2) std(pmwt_r2);
    mean(pmwt_r3) std(pmwt_r3); 
    mean(pmwt_r4) std(pmwt_r4);
    mean(pmwt_r5) std(pmwt_r5);
    mean(pmwt_r6) std(pmwt_r6);
    mean(pmwt_r7) std(pmwt_r7);
    mean(piamwt_r1) std(piamwt_r1);
    mean(piamwt_r2) std(piamwt_r2);
    mean(piamwt_r3) std(piamwt_r3);
    mean(piamwt_r4) std(piamwt_r4);
    mean(piamwt_r5) std(piamwt_r5);
    mean(piamwt_r6) std(piamwt_r6);
    mean(piamwt_r7) std(piamwt_r7);]

% Mean and standard deviation of each stock over forecast period
[mean(ret_f); std(ret_f)]