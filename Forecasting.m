%%Forecasting + 12-mo holdout (uses existing y, dates, yLog) ===
assert(isequal(size(y),size(yLog),size(dates)), 'y, yLog, dates must align');
assert(all(ismember(class(dates), "datetime")), 'dates must be datetime');

back = @(z) exp(z);

% Step regressor (Apr 2020 .. Sep 2021)
t1 = datetime(2020,4,1);  t2 = datetime(2021,9,1);
step = double((dates >= t1) & (dates <= t2));  % 127×1

% Fit regARIMA with step, find 1 pulse, refit
M = regARIMA('ARLags',[1 2 4], 'D',1, 'Seasonality',12, 'SMALags',[12 24], ...
             'Intercept',0, 'Beta',NaN);
EstR = estimate(M, yLog, 'X', step, 'Display','off');

[resR, vR] = infer(EstR, yLog, 'X', step);
z = resR ./ sqrt(vR); valid = ~isnan(z);
[~,iLoc] = max(abs(z(valid)));
idxMap = find(valid); idxPulse = idxMap(iLoc);

pulse = zeros(size(yLog)); pulse(idxPulse)=1;
X2 = [step pulse];

M2 = regARIMA('ARLags',[1 2 4], 'D',1, 'Seasonality',12, 'SMALags',[12 24], ...
              'Intercept',0, 'Beta',NaN(2,1));
EstR2 = estimate(M2, yLog, 'X', X2, 'Display','off');

% ---------- 12-month ahead forecast ----------
h = 12;
XF = zeros(h,2);                               % future step & pulse = 0
[yF, yMSE] = forecast(EstR2, h, 'Y0', yLog, 'X0', X2, 'XF', XF);

fDates = (dates(end) + calmonths(1:h))';
yPred = back(yF);
yU    = back(yF + 1.96*sqrt(yMSE));
yL    = back(yF - 1.96*sqrt(yMSE));

figure; hold on; grid on
plot(dates, y, 'k-', 'LineWidth', 1.4);
plot(fDates, yPred, 'r-', 'LineWidth', 1.6);
plot(fDates, yU, 'g--', fDates, yL, 'g--');
legend('Observed','Forecast','95% Upper','95% Lower','Location','best');
title('SARIMA Forecast (12-month horizon)');
xlabel('Date'); ylabel('Influenza Cases');

% ---------- 12-month HOLDOUT (no leakage) ----------
yTr = yLog(1:end-h); yTe = yLog(end-h+1:end);
dTr = dates(1:end-h); dTe = dates(end-h+1:end);

stepTr = double((dTr >= t1) & (dTr <= t2));
stepTe = double((dTe >= t1) & (dTe <= t2));

% pulse picked on TRAIN only
M_tr = regARIMA('ARLags',[1 2 4], 'D',1, 'Seasonality',12, 'SMALags',[12 24], ...
                'Intercept',0, 'Beta',NaN);
Est_tr = estimate(M_tr, yTr, 'X', stepTr, 'Display','off');
[res_tr, v_tr] = infer(Est_tr, yTr, 'X', stepTr);
z_tr = res_tr ./ sqrt(v_tr); valid_tr = ~isnan(z_tr);
[~,iLocTr] = max(abs(z_tr(valid_tr)));
idxMapTr = find(valid_tr); idxPulseTr = idxMapTr(iLocTr);

pulseTr = zeros(size(yTr)); pulseTr(idxPulseTr)=1;
pulseTe = zeros(size(yTe));
XTr = [stepTr pulseTr];  XTe = [stepTe pulseTe];

M2_tr = regARIMA('ARLags',[1 2 4], 'D',1, 'Seasonality',12, 'SMALags',[12 24], ...
                 'Intercept',0, 'Beta',NaN(2,1));
Est_hold = estimate(M2_tr, yTr, 'X', XTr, 'Display','off');
[yF_log, yMSE_h] = forecast(Est_hold, h, 'Y0', yTr, 'X0', XTr, 'XF', XTe);

yPredHO = back(yF_log);
yTrueHO = back(yTe);

rmse = sqrt(mean((yTrueHO - yPredHO).^2));
mae  = mean(abs(yTrueHO - yPredHO));
mape = mean(abs((yTrueHO - yPredHO)./max(yTrueHO,1e-9)))*100;
disp(table(rmse,mae,mape,'VariableNames',{'RMSE','MAE','MAPE'}));

figure; hold on; grid on
plot(dates, y, 'k-', 'LineWidth', 1.2);
plot(dTe, yPredHO, 'r-', 'LineWidth', 1.6);
scatter(dTe, yTrueHO, 36, 'filled');
legend('Observed (all)','Holdout Forecast','Holdout Actual','Location','best');
title('12-Month Holdout Comparison');
xlabel('Date'); ylabel('Influenza Cases');
disp(Accuracy)
