function [est, VaRG, ESG, VaRN, ESN] = getstuff(x,B,ser,res)

est=x*B;                        % Predicted excess returns for value scenario given by values in x
VaRG=est+norminv(0.01)*ser;     % VaR for scenario assuming Gaussian errors
ESG=est+norminv(0.0038)*ser;    % ES for scenario assuming Gaussian errors
VaRN=est+quantile(res,0.01);    % Non-parametric VaR 
ESN=est+mean(res(res<quantile(res,0.01))); % Non-parametric ES