% Home-->Import data "CBA_Daily_Jan1999_Jan2018_yahoo.csv" 
% as a Numeric Matrix and name the data 'CBAdata'

%% (i) Plot the CBA price series
prices=CBAdata(:,6);
figure; plot(prices)

%% (ii)	Convert the price series to percentage log returns and calculate descriptive stats
CBAr = 100*diff(log(prices));

[mean(CBAr) median(CBAr) std(CBAr) skewness(CBAr) kurtosis(CBAr)]

%% (iii) Plot a histogram of the percentage log returns.
figure; hist(CBAr,50)

%% (iv) Calculate percentiles
[prctile(CBAr,0.5) prctile(CBAr,1) prctile(CBAr,10) prctile(CBAr,25) prctile(CBAr,75) prctile(CBAr,90) prctile(CBAr,99) prctile(CBAr,99.5) ]

%% (v) 0.1% percentile
prctile(CBAr,0.1)

%% (vi) Plot log returns
figure; plot(CBAr)