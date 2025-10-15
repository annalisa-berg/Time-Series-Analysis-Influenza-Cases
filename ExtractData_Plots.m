% Extract the flu counts
y = fludataset.inf_allcombined;

% Extract the dates
dates = fludataset.Day;

% If dates are text, convert them to datetime:
dates = datetime(dates, 'InputFormat','yyyy-MM-dd');

% Create month index (0,1,2,...)
months = (0:length(y)-1)';
% Plot data
figure;
plot(months, y, 'b-', 'LineWidth', 1.5);
xlabel('Months since Jan 2015');
ylabel('Influenza Cases (All reported strains)');
title('Monthly Flu Cases (2015–2025)');
grid on;
% Histogram of raw data
figure;
histogram(y, 20);
xlabel('Influenza Cases');
ylabel('Frequency');
title('Distribution of Flu Cases');
% Overall mean and variance
mean_y = mean(y);
var_y = var(y);

fprintf('Overall mean of flu cases: %.2f\n', mean_y);
fprintf('Overall variance of flu cases: %.2f\n', var_y);
% Rolling mean and variance (12-month window)
window = 12; % yearly window
y_movavg = movmean(y, window);
y_movvar = movvar(y, window);

% Print some summary values (first and last window results)
fprintf('Rolling mean (first window): %.2f\n', y_movavg(window));
fprintf('Rolling mean (last window): %.2f\n', y_movavg(end));
fprintf('Rolling variance (first window): %.2f\n', y_movvar(window));
fprintf('Rolling variance (last window): %.2f\n', y_movvar(end));

%Table of yearly mean and yearly variance
% Extract years
years = year(datetime(fludataset.Day, 'InputFormat','yyyy-MM-dd'));

% Add years as a new column in the table
fludataset.Year = years;

% Mean and variance by year
T = groupsummary(fludataset, "Year", {"mean","var"}, "inf_allcombined");

% Rename variables for clarity
T.Properties.VariableNames = {'Year','GroupCount','MeanCases','VarianceCases'};

% Display table
disp(T)

%Variance Stabilization
%Log Transformation
yLog = log(y);

% Plot log-transformed series
figure;
plot(months, yLog, 'r-', 'LineWidth', 1.5);
xlabel('Months since Jan 2015');
ylabel('Log Flu Cases');
title('Log-Transformed Monthly Flu Cases');
grid on;

% Histogram of log-transformed data
figure;
histogram(yLog, 20);
xlabel('Log Flu Cases');
ylabel('Frequency');
title('Distribution after Log Transform');

%Box-Cox Transformation
%Estimate Lambda value using ARIMA
[transformedY, lambda] = boxcox(y);

fprintf('Estimated Box-Cox λ = %.3f\n', lambda);

yBC = (y.^lambda - 1) / lambda;

% Plot Box-Cox transformed data
figure;
plot(months, yBC, 'g-', 'LineWidth', 1.5);
xlabel('Months since Jan 2015');
ylabel('Box-Cox Transformed Flu Cases');
title('Box-Cox Transformed Monthly Flu Cases');
grid on;
%Histogram Box-Cox transformed data
figure;
histogram(yBC, 20);
xlabel('Box–Cox Transformed Flu Cases');
ylabel('Number of Months');
title(sprintf('Histogram of Box–Cox Transformed Data (λ = %.3f)', lambda));
grid on;
