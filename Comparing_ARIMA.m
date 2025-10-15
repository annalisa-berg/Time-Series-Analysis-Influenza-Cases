% ==========================================
% Compare non-seasonal ARIMA orders by AIC
% ==========================================
yT = yLog;   % log-transformed series

models = { ...
    arima('D',1,'MALags',1), ...            % ARIMA(0,1,1)
    arima('ARLags',1,'D',1,'MALags',1), ... % ARIMA(1,1,1)
    arima('ARLags',1:4,'D',1,'MALags',1) ...% ARIMA(4,1,1)
    arima('ARLags',1,'D',1,'MALags',1:4) ...% ARIMA(1,1,4)
};
names = {'ARIMA(0,1,1)','ARIMA(1,1,1)','ARIMA(4,1,1)', 'ARIMA(1,1,4)'};

AIC = nan(numel(models),1);

for i = 1:numel(models)
    try
        [Est,~,logL] = estimate(models{i}, yT, 'Display','off');

        % Count estimated parameters
        k = 0;
        f = fieldnames(Est);
        for j = 1:numel(f)
            v = Est.(f{j});
            if isnumeric(v)
                k = k + sum(~isnan(v(:)));
            end
        end

        % Compute AIC
        AIC(i) = -2*logL + 2*k;
        fprintf('%s  AIC = %.2f  (k=%d)\n', names{i}, AIC(i), k);
    catch ME
        fprintf('%s  FAILED: %s\n', names{i}, ME.message);
    end
end

% Show results table
Results = table(names', AIC, 'VariableNames', {'Model','AIC'});
disp(Results);

%% SARIMA Model Comparison with AIC, AICc, and BIC
% Data: yLog (log-transformed flu counts)

yT = yLog;                % series
n  = length(yT);          % sample size
s  = 12;                  % seasonal period (monthly data)

%% Define candidate SARIMA orders
% Each row = [p d q P D Q]
orders = [
    1 1 1 0 1 1;   % SARIMA(1,1,1)(0,1,1)[12]
    1 1 4 0 1 2;   % SARIMA(1,1,4)(0,1,2)[12]
    1 1 1 0 1 2;   % SARIMA(1,1,1)(0,1,2)[12]
    4 1 1 0 1 2;   % SARIMA(4,1,1)(0,1,2)[12]
    1 1 4 0 1 1;   % SARIMA(1,1,4)(0,1,1)[12]
    4 1 1 0 1 1    % SARIMA(4,1,1)(0,1,1)[12]
];

modelNames = {
    'SARIMA(1,1,1)(0,1,1)[12]'
    'SARIMA(1,1,4)(0,1,2)[12]'
    'SARIMA(1,1,1)(0,1,2)[12]'
    'SARIMA(4,1,1)(0,1,2)[12]'
    'SARIMA(1,1,4)(0,1,1)[12]'
    'SARIMA(4,1,1)(0,1,1)[12]'
};

nModels = size(orders,1);

%% Initialize results
AIC    = nan(nModels,1);
AICc   = nan(nModels,1);
BIC    = nan(nModels,1);
Sigma2 = nan(nModels,1);
K      = nan(nModels,1);

%% Loop through models
for i = 1:nModels
    p = orders(i,1); d = orders(i,2); q = orders(i,3);
    P = orders(i,4); D = orders(i,5); Q = orders(i,6);

    % Build ARIMA model
    mdl = arima('ARLags',1:p, 'D',d, 'MALags',1:q, ...
                'Seasonality',s, 'SARLags',s*(1:P), ...
                'SMALags',s*(1:Q));

    try
        [Est,~,logL] = estimate(mdl, yT, 'Display','off');

        % Number of parameters: p+q+P+Q + variance
        % (+1 more if constant is estimated, but usually not after differencing)
        K(i) = p + q + P + Q + 1;

        % Innovation variance
        Sigma2(i) = Est.Variance;

        % Information criteria
        AIC(i)  = -2*logL + 2*K(i);
        AICc(i) = AIC(i) + (2*K(i)*(K(i)+1))/(n-K(i)-1);
        BIC(i)  = -2*logL + K(i)*log(n);

    catch ME
        fprintf('%s FAILED: %s\n', modelNames{i}, ME.message);
    end
end

%% Force everything into columns
modelNames = modelNames(:);
K      = K(:);
AIC    = AIC(:);
AICc   = AICc(:);
BIC    = BIC(:);
Sigma2 = Sigma2(:);

%% Build results table
Results = table(modelNames, K, AIC, AICc, BIC, Sigma2, ...
    'VariableNames', {'Model','K','AIC','AICc','BIC','Sigma2'});

%% Display
disp(Results)

%MLE For 6 proposed SARIMA models
 for i = 1:size(orders,1)
    p = orders(i,1); d = orders(i,2); q = orders(i,3);
    P = orders(i,4); D = orders(i,5); Q = orders(i,6);

    mdl = arima('ARLags',1:p, 'D',d, 'MALags',1:q, ...
                'Seasonality',s, 'SARLags',s*(1:P), ...
                'SMALags',s*(1:Q));

    fprintf('\n=== %s ===\n', modelNames{i});

    try
        % Estimate parameters
        [Est,~,~] = estimate(mdl, yT, 'Display','off');

        % Get summary
        S = summarize(Est);

        % Extract coefficient table only
        T = S.Table;

        % Add significance flag (p < 0.05)
        T.Significant = T.PValue < 0.05;

        disp(T)

    catch err
        fprintf('%s FAILED: %s\n', modelNames{i}, err.message);
    end
 end