%% Detrending & Deseasonalizing: Applying difference
% Applying first differencing

%First difference: Δy_t = y_t - y_{t-1}
y_diff1 = diff(yLog, 1);

%aligning dates: drop first date
dates_diff1 = dates(2:end);

% Applying Seasonal Differencing
%lag = 12 months
S = 12; %12 months
y_sdiff = yLog(S+1:end) - yLog(1:end-S); 
%align dates --> series is now S shorter
dates_sdiff1 = dates(S+1:end);

%Applying Combined Difference (First+Seasonal)
%Seasonal first then first difference
tempVar = yLog(S+1:end) - yLog(1:end-S);
combined = diff(tempVar, 1);
dates_comb = dates(S+2:end);

figure;
plot(dates_comb, combined, 'm-', 'LineWidth', 1.5);
xlabel('Date');
ylabel('Combined Differenced log(Cases)');
title('Combined Differenced Log-Transformed Flu Cases');
grid on;

% ----- ACF -----
maxLag = 40;
figure;
autocorr(combined, NumLags=maxLag);
title('ACF of Stationary Flu Series');

% ----- PACF -----
figure;
parcorr(combined, NumLags=maxLag);
title('PACF of Stationary Flu Series');