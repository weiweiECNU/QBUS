%% Q1 PCA and Factor Modelling
% I imported the data from the file 'Tsay_FM_data.txt' as column vectors.
% The names of the columns are in the first row of the file and become the
% names of the 5 separate vectors. I then saved the workspace with command:
% save lab5.mat
load lab5.mat
Tsay_data = [BAC HPQ IBM INTC JPM]; % combine all columns into one matrix

%% Q1(a) Estimate the sample correlation matrix
% Estimate the sample correlation matrix showing all pairwise correlations.
corr(Tsay_data)    

% Scatterplot of BAC vs JPM
figure;plot(BAC,JPM,'+');title('Plot of JPM vs BAC')
;xlabel('BAC');ylabel('JPM');
% relationship seems roughly linear.

% Calculate (and test) correlation between BAC and INTC (least significant)
[pc,p]=corr(BAC,INTC)

%% Q1(b) PCA analysis

% Plot the 5 data series
plot(Tsay_data);title('5 US return series');
legend('BAC','HPQ','IBM','INTC','JPM','location','southwest');
xlim([0 length(Tsay_data)]);

% Perform PCA on Tsay_data
[pc_ret,score_ret,latent_ret] = princomp(Tsay_data)

% Percentage variance explained per component
latent_ret./sum(latent_ret)

% Cumulative percentage variance explained
cumsum(latent_ret)./sum(latent_ret)

% create vector/cell array of labels for biplot
vbls = {'BAC','HPQ','IBM','INTC','JPM'};

% Biplot of components 1 and 2
figure;biplot(pc_ret(:,1:2),'varlabels',vbls);

% Biplot of components 1, 2 and 3
figure;biplot(pc_ret(:,1:3),'varlabels',vbls);

%% Q1(c) Describe PCs

% sample covariance matrix
cov(Tsay_data)

% Plot of 5 return series together along with first 3 components
figure;subplot(4,1,1);plot(Tsay_data);xlim([0 length(Tsay_data)]);
title('Stock returns');xlim([1 length(Tsay_data)]);
subplot(4,1,2);plot(score_ret(:,1));
title('1st Principle component');xlim([1 length(score_ret)]);
subplot(4,1,3);plot(score_ret(:,2));xlim([0 length(Tsay_data)]);
title('2nd Principle component');xlim([1 length(score_ret)]);
subplot(4,1,4);plot(score_ret(:,3));xlim([0 length(Tsay_data)]);
title('3rd Principle component');xlim([1 length(score_ret)]);

% Correlation matrix of all 5 components
corr(score_ret)   %  note the zero correlations of the PCs

% Plot of 5 return series together along with first 5 components together 
figure;subplot(2,1,1);plot(Tsay_data);title('5 US return series');
xlim([1 length(Tsay_data)]);
legend('BAC','HPQ','IBM','INTC','JPM','location','south','Orientation','horizontal');
subplot(2,1,2);plot(score_ret);title('5 Principle Components');
legend('PC1','PC2','PC3','PC4','PC5','location','south','Orientation','horizontal');
xlim([1 length(Tsay_data)]);

% Correlation between the average of the returns along with the first component
corrcoef(mean(Tsay_data'),score_ret(:,1));

% Scatter plot of the average of the returns and the first component
figure;plot(mean(Tsay_data'),score_ret(:,1),'+');
title('PC1 vs Average of the returns');
xlabel('Average return');ylabel('PC1');

% Plot of the daily average of the returns and the first component
figure;plot(mean(Tsay_data')); %transpose to take average of 5 returns each period
hold on;plot(score_ret(:,1),'r');xlim([0 length(Tsay_data)]);
legend('Average returns','PC1','location','southeast');
title('Average Returns and PC1');

%% Q1(d) Perform Factor Analysis m=1

ymat=Tsay_data;  % rows are observations over time, columns are variables
size(ymat)   % should be T by n

[lam1_rets,psi1_rets,T1_ret,stats1_ret,F1_ret] = factoran(ymat,1);  % fit 1 factor model

% show standardised factor loadings and specific error variances
[lam1_rets psi1_rets]

% factor loadings (regression coefficients)
lam1_ret=lam1_rets.*(std(ymat))'; % std(ymat) gives stds, transpose give 5x1 vector, dot multiply another 5x1

%  specific error variances (SER^2)
psi1_ret=(psi1_rets.*(var(ymat))');

[lam1_ret psi1_ret sqrt(psi1_ret) (1-psi1_ret'./var(ymat))']
%  above code displays actual factor loadings (regression coefficients),
%  specific error variances (SER^2), SER and adjusted R-squared for each industry series
%  psi1 are specific variances, so var(ymat)-psi1_ret'  gives communalities

% overall amount of variance captured by the factor model
(trace(cov(ymat))-sum(psi1_ret))/trace(cov(ymat))

% Stats from the factor analysis
stats1_ret

% Estimated error variances, and sample variances, for each asset
[psi1_ret';var(ymat)]

% Sample mean and variance of single factor 
[mean(F1_ret) var(F1_ret)]

% SER, R-squared and Standard Deviations for each asset
[sqrt(psi1_ret'); (1-psi1_ret'./var(ymat));  std(ymat)]


%% Q1(e)	Describe the factor loadings and factor found
% Plot 5 US returns series and single factor
figure;subplot(2,1,1);plot(ymat);xlim([1 length(Tsay_data)]);
title('5 Asset Returns');
legend('IBM','HPQ','INTC','JPM','BAC','Location','South','Orientation','horizontal');
subplot(2,1,2);plot(F1_ret);xlim([1 length(Tsay_data)]);
title('1st Factor');

%% Q1(f) Assess whether 1 factor model is appropriate 

% Correlation between first principle component and single factor
corrcoef(score_ret(:,1),F1_ret) 

% Correlations of asset return and single factor
corr([ymat F1_ret])

%% Q1(g) Perform a Factor Analysis with m=2 factors

% Estimate 2-factor model, m=2
[lam2_rets,psi2_rets,T2_ret,stats2_ret,F2_ret]=factoran(ymat,2,'maxit',500);  % fit 2 factor model
% NB the maxit option specifies how many iterations to use in the search
% procedure for estimates. the default is 250 which is not enough for this
% model and data, so I set it higher.

% show standardised factor loadings and specific error variances
[lam2_rets psi2_rets]
%  Display the two columns of factor loadings, then specific error variances, SER and adjusted R
%  squared for each industry series from the single factor model.
lam2_ret=lam2_rets;
lam2_ret(:,1)=lam2_rets(:,1).*(std(ymat))';
lam2_ret(:,2)=lam2_rets(:,2).*(std(ymat))';
psi2_ret=psi2_rets.*(var(ymat))';

% Display the two columns of factor loadings, specific error variances, 
% SER and adj. R-squared for each industry for the single factor model.
[lam2_ret psi2_ret sqrt(psi2_ret) (1-psi2_ret'./var(ymat))']

% overall amount of variance captured by the factor model
(trace(cov(ymat))-sum(psi2_ret))/trace(cov(ymat))

% Stats from the factor analysis
stats2_ret

% Estimated error variances, and sample variances, for each asset
[psi2_ret';var(ymat)]

% Sample means and variance of 2 factors 
mean(F2_ret) 
% Covariance matrix of 2 factors 
var(F2_ret)

% SER, R-squared and Standard Deviations for each asset
[sqrt(psi2_ret'); (1-psi2_ret'./var(ymat));std(ymat)]

figure;subplot(3,1,1);plot(ymat);xlim([1 length(Tsay_data)]);
title('5 Asset Returns');
%legend('IBM','HPQ','INTC','JPM','BAC','location','South','Orientation','horizontal');
subplot(3,1,2);plot(F2_ret(:,1));xlim([1 length(Tsay_data)]);
title('1st Factor');
subplot(3,1,3);plot(F2_ret(:,2));xlim([1 length(Tsay_data)]);
title('2nd Factor');

% Correlations between asset return and factors
corr([ymat F2_ret])

% create vector/cell array of labels for biplot
vbls = {'BAC','HPQ','IBM','INTC','JPM'};

% Create biplot of two factors
figure;biplot(lam2_ret,'varlabels',vbls,...  % Add variable labels to plot
   'LineWidth',2,...                         % Set linewidth for biplot
   'MarkerSize',20)                          % Set marker size for biplot

%% Q1(h) Unrotated factors
% Calculate and display unrotated loadings.
unrot_lam = lam2_ret*inv(T2_ret);   % Often makes 1st factor similar to single factor model :)
unrot_lam'

% Create biplot of two unrotated factors
figure;biplot(unrot_lam,'varlabels',vbls,...    % Add variable labels to plot
       'LineWidth',2,...                        % Set linewidth for biplot
       'MarkerSize',20)                         % Set marker size for biplot

%% Fit 3 factor model
[lam3_ret,psi3_ret,T3_ret,stats3_ret,F3_ret]=factoran(ymat,3,'maxit',500);  

% This code returns the following error message: 
%   " Error using factoran (line 139)
%     The number of factors requested, M, is too large for the number of the observed variables. "
% Matlab is telling us that three factors give too many unknowns to estimate for this data.
