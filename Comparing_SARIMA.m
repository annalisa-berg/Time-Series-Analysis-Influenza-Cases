 %% Refined SARIMA Model Comparison
yT = yLog;   % log-transformed series
n  = length(yT);
s  = 12;     % seasonality

% Store refined models
refinedModels = {
    % SARIMA(1,1,4)(0,1,2)[12] refined → MA(4), SMA(12), SMA(24)
    arima('ARLags',[], 'D',1, 'MALags',4, ...
          'Seasonality',s, 'SMALags',[12 24]), ...
    'SARIMA(1,1,[4])(0,1,[1,2])[12]';

    % SARIMA(4,1,1)(0,1,1)[12] refined → AR(1,2,4), SMA(12)
    arima('ARLags',[1 2 4], 'D',1, 'MALags',[], ...
          'Seasonality',s, 'SMALags',12), ...
    'SARIMA([1,2,4],1,0)(0,1,[1])[12]';

    % SARIMA(4,1,1)(0,1,2)[12] refined → AR(1,2,4), SMA(12), SMA(24)
    arima('ARLags',[1 2 4], 'D',1, 'MALags',[], ...
          'Seasonality',s, 'SMALags',[12 24]), ...
    'SARIMA([1,2,4],1,0)(0,1,[1,2])[12]'
};

nModels = size(refinedModels,1);

% Initialize result storage
AIC    = nan(nModels,1);
AICc   = nan(nModels,1);
BIC    = nan(nModels,1);
Sigma2 = nan(nModels,1);
K      = nan(nModels,1);

%% Loop through refined models
for i = 1:nModels
    mdl = refinedModels{i,1};
    name = refinedModels{i,2};

    try
        [Est, EstCov, logL] = estimate(mdl, yT, 'Display','off');

        % Parameter count
        K(i) = size(EstCov,1);

        % Innovation variance
        Sigma2(i) = Est.Variance;

        % Information criteria
        AIC(i)  = -2*logL + 2*K(i);
        AICc(i) = AIC(i) + (2*K(i)*(K(i)+1))/(n-K(i)-1);
        BIC(i)  = -2*logL + K(i)*log(n);

        fprintf('%s: AIC=%.2f, AICc=%.2f, BIC=%.2f, K=%d\n', ...
                 name, AIC(i), AICc(i), BIC(i), K(i));

    catch ME
        fprintf('%s FAILED: %s\n', name, ME.message);
    end
end
%% Build results table
Results = table(refinedModels(:,2), K, AIC, AICc, BIC, Sigma2, ...
    'VariableNames', {'Model','K','AIC','AICc','BIC','Sigma2'});

% Final chosen model: SARIMA([1,2,4],1,0)(0,1,[1,2])[12]

s  = 12;                 % seasonality
yT = yLog;               % your log-transformed series

% Define the SARIMA model structure
mdl_final = arima('ARLags',[1 2 4], 'D',1, ...
                  'MALags',[], ...
                  'Seasonality',s, ...
                  'SARLags',[], ...
                  'SMALags',[12 24]);

% Estimate the model parameters using the log-transformed series
[Est_final, ~, ~] = estimate(mdl_final, yT, 'Display','off');

% Infer residuals (innovations) and their conditional variances from the estimated model
[res, resVar] = infer(Est_final, yT);

% Standardize residuals (recommended for diagnostics)
resStd = res ./ sqrt(resVar);

% Drop any NaNs that may appear at the beginning due to differencing/seasonality
resStd_valid = resStd(~isnan(resStd));

figure;
plot(resStd_valid, 'LineWidth', 1);
yline(0, '--');
title('Standardized Residuals - SARIMA');
xlabel('Months From Jan 2015');
ylabel('Std. residual');
grid on;
figure;
histogram(resStd_valid, 20, 'Normalization','pdf');
hold on;
% Overlay standard normal (since we standardized)
x = linspace(min(resStd_valid), max(resStd_valid), 200);
y = normpdf(x, 0, 1);
plot(x, y, 'LineWidth', 1.5);
hold off;
title('Histogram of Standardized Residuals with N(0,1) Overlay');
xlabel('Std. residual');
ylabel('Density');
grid on;

%ACF and PACF of Standardized Residuals
figure; autocorr(resStd, 'NumLags', 40); title('ACF of Standardized Residuals');
figure; parcorr(resStd, 'NumLags', 40); title('PACF of Standardized Residuals');

%Ljung-Box
[h12,p12] = lbqtest(resStd, 'Lags', 12);
[h20,p20] = lbqtest(resStd, 'Lags', 20);
[h24,p24] = lbqtest(resStd, 'Lags', 24);

fprintf('Ljung-Box: h12=%d p12=%.4f | h20=%d p20=%.4f | h24=%d p24=%.4f\n', ...
% McLeod-Li     
rs2 = resStd.^2;
[hML12,pML12] = lbqtest(rs2, 'Lags', 12);
[hML20,pML20] = lbqtest(rs2, 'Lags', 20);
[hML24,pML24] = lbqtest(rs2, 'Lags', 24);
fprintf('McLeod–Li (squared res): h12=%d p12=%.4f | h20=%d p20=%.4f | h24=%d p24=%.4f\n', ...
        hML12,pML12,hML20,pML20,hML24,pML24);
%Jarque–Bera normality test on standardized residuals
r = resStd(:);
r = r(~isnan(r) & isfinite(r));   % clean

[hJB, pJB, JBstat] = jbtest(r);   % h=1 -> reject normality at default alpha=0.05
fprintf('Jarque–Bera: h=%d (1=reject), p=%.4f, JB=%.2f\n', hJB, pJB, JBstat);

% Removing Outliers
mask = isbetween(dates, datetime(2020,4,1), datetime(2020,9,1));
RemovedRows = fludataset(mask, {'Day','inf_allcombined'});   % subset rows/cols
RemovedRows.Index = find(mask);                               % add index
RemovedRows.Properties.VariableNames = {'Date','RawCases','Index'};
RemovedRows = RemovedRows(:, {'Index','Date','RawCases'});    % reorder
disp(RemovedRows)
yT_clean = yLog;
yT_clean(mask) = NaN;

% 2) Refit your chosen model
mdl_final = arima('ARLags',[1 2 4], 'D',1, ...
                  'Seasonality',12, 'SMALags',[12 24]);

[Est_clean, EstCov_clean, logL_clean] = estimate(mdl_final, yT_clean, 'Display','off');

% 3) Information criteria 
n   = numel(yT_clean);
K   = size(EstCov_clean,1);
AIC  = -2*logL_clean + 2*K;
AICc = AIC + (2*K*(K+1))/(n - K - 1);
BIC  = -2*logL_clean + K*log(n);
fprintf('ICs (cleaned): AIC=%.2f, AICc=%.2f, BIC=%.2f, K=%d\n', AIC, AICc, BIC, K);
ICs (cleaned): AIC = 299.27, AICc = 300.21, BIC = 319.18, K = 7
[resC, varC] = infer(Est_clean, yT_clean);
resStdC = resC ./ sqrt(varC);
resStdC = resStdC(~isnan(resStdC));

% Plots
figure; plot(resStdC,'LineWidth',1); yline(0,'--'); grid on
title('Standardized Residuals (after removing COVID months)');
figure; histogram(resStdC,20,'Normalization','pdf'); hold on
x = linspace(min(resStdC),max(resStdC),200); plot(x, normpdf(x,0,1),'LineWidth',1.5);
hold off; grid on; title('Histogram of Std Residuals with N(0,1) Overlay');

% Tests
[h12,p12] = lbqtest(resStdC,'Lags',12);
[h20,p20] = lbqtest(resStdC,'Lags',20);
[h24,p24] = lbqtest(resStdC,'Lags',24);
fprintf('LBQ (cleaned): p12=%.4f p20=%.4f p24=%.4f\n', p12,p20,p24);

[hML12,pML12] = lbqtest(resStdC.^2,'Lags',12);
[hML20,pML20] = lbqtest(resStdC.^2,'Lags',20);
[hML24,pML24] = lbqtest(resStdC.^2,'Lags',24);
fprintf('McLeod–Li (cleaned): p12=%.4f p20=%.4f p24=%.4f\n', pML12,pML20,pML24);

%% Align and build step regressor (Apr 2020 .. Sep 2021)
yUse   = yLog(:);                 % T×1 double
datesU = dates(:);                % T×1 datetime
assert(numel(yUse)==numel(datesU), 'yLog and dates must have same length');

T = numel(yUse);
t1 = datetime(2020,4,1); 
t2 = datetime(2021,9,1);

step = double((datesU >= t1) & (datesU <= t2));
step = reshape(step, [], 1);      % force T×1 column
M = regARIMA('ARLags',[1 2 4], 'D',1, ...
             'Seasonality',12, 'SMALags',[12 24], ...
             'Intercept',0, ...          % <-- FIX: intercept not estimated
             'Beta',NaN);                % 1 exogenous regressor (the step)

[EstR, ParamCovR, logLR] = estimate(M, yUse, 'X', step, 'Display','off');

% Residuals
[resR, varR] = infer(EstR, yUse, 'X', step);
rStdR = resR ./ sqrt(varR);
rStdR = rStdR(~isnan(rStdR));

% Diagnostics 
[h12,p12] = lbqtest(rStdR,'Lags',12);
[h20,p20] = lbqtest(rStdR,'Lags',20);
[h24,p24] = lbqtest(rStdR,'Lags',24);
[hML12,pML12] = lbqtest(rStdR.^2,'Lags',12);
[hJB,pJB,JB] = jbtest(rStdR);

fprintf('LBQ p12=%.4f p20=%.4f p24=%.4f | McLeod–Li p12=%.4f | JB p=%.4f (h=%d)\n', ...
        p12,p20,p24, pML12, pJB, hJB);

% ICs
T = numel(yUse); K = size(ParamCovR,1);
AIC  = -2*logLR + 2*K;
AICc = AIC + (2*K*(K+1))/(T - K - 1);
BIC  = -2*logLR + K*log(T);
fprintf('ICs (step): AIC=%.2f, AICc=%.2f, BIC=%.2f (K=%d)\n', AIC, AICc, BIC, K);

% You already have: yUse (T×1), datesU (T×1), step (T×1), and EstR from regARIMA with step.
T = numel(yUse);

% 1) Find the single largest standardized residual *with index mapping*
[resR, varR] = infer(EstR, yUse, 'X', step);
zFull = resR ./ sqrt(varR);                 % T×1, may have NaNs at start
valid = ~isnan(zFull);
idxMap = find(valid);                        % map from valid z to original time index
[~, iLocal] = max(abs(zFull(valid)));        % location among valid points
idxPulse = idxMap(iLocal);                   % index in 1..T of worst month

% 2) Build pulse dummy and fit regARIMA with 2 regressors: [step pulse]
pulse = zeros(T,1); pulse(idxPulse) = 1;
X2 = [step pulse];

M2 = regARIMA('ARLags',[1 2 4], 'D',1, ...
              'Seasonality',12, 'SMALags',[12 24], ...
              'Intercept',0, 'Beta',NaN(2,1));

[EstR2, ParamCovR2, logLR2] = estimate(M2, yUse, 'X', X2, 'Display','off');

% 3) Diagnostics
[res2, v2] = infer(EstR2, yUse, 'X', X2);
r2 = res2 ./ sqrt(v2); r2 = r2(~isnan(r2));

[h12,p12] = lbqtest(r2,'Lags',12);
[h20,p20] = lbqtest(r2,'Lags',20);
[h24,p24] = lbqtest(r2,'Lags',24);
[hJB,pJB,JB] = jbtest(r2);

% 4) Information criteria
K2   = size(ParamCovR2,1);
T    = numel(yUse);
AIC2  = -2*logLR2 + 2*K2;
AICc2 = AIC2 + (2*K2*(K2+1))/(T - K2 - 1);
BIC2  = -2*logLR2 + K2*log(T);

fprintf(['With 1 pulse at t=%d: LBQ p12=%.4f p20=%.4f p24=%.4f | ' ...
         'JB p=%.4f (h=%d) | AIC=%.2f AICc=%.2f BIC=%.2f (K=%d)\n'], ...
         idxPulse, p12,p20,p24, pJB,hJB, AIC2,AICc2,BIC2,K2);