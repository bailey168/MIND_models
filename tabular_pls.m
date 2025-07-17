tic

fprintf('Tabular PLS\n');

rng(42, 'combRecursive');

addpath('/Users/baileyng/MIND_models/circularGraph');

% Load the CSV data with preserved column headers
data_path = '/Users/baileyng/MIND_data/ukb_master_allcols_no_outliers.csv';
data = readtable(data_path, 'VariableNamingRule', 'preserve');
fprintf('Data loaded successfully. Size: %d rows x %d columns\n', height(data), width(data));


% Set X
% Read region names from file
regions_file = '/Users/baileyng/MIND_models/region_names/CT_regions.txt';
regions = readlines(regions_file);
regions = regions(regions ~= ""); % Remove empty lines

X = data(:, regions);
fprintf('X matrix created with %d rows x %d columns\n', height(X), width(X));

% Set Y
Y_columns = {'trailmaking_score', '20016-2.0', '20197-2.0', '23324-2.0'};
Y = data(:, Y_columns);
fprintf('Y matrix created with %d rows x %d columns\n', height(Y), width(Y));

% Convert tables to arrays for zscore
Y = table2array(Y);    % Convert Y table to numeric array
X = table2array(X);    % Convert X table to numeric array

Y = zscore(Y);    % mean-0, SD-1 each column
X = zscore(X);
Y(:,1)=Y(:,1)*-1;
ncomp = 3;        % number of PLS components

% Regress out age, sex and other covariates
% Extract covariate columns from data
age = data.('21003-2.0');
sex = data.('31-0.0');
assessment_center = data.('54-2.0');
head_motion = data.('25741-2.0');

% Regress out covariates from Y columns - VECTORIZED VERSION
Y_covariates = [age, sex, assessment_center];

% Create design matrix for Y regression
Y_design_matrix = [ones(size(Y_covariates,1),1), Y_covariates]; % Add intercept

% Vectorized regression for all Y columns at once
Y_beta_coeffs = Y_design_matrix \ Y; % Solve for all columns simultaneously
Y_predicted = Y_design_matrix * Y_beta_coeffs;
Y = Y - Y_predicted; % Residuals

% Regress out covariates from X columns - VECTORIZED VERSION
% X_covariates = [age, sex, assessment_center, head_motion];
X_covariates = [age, sex, assessment_center];

% Create design matrix once
X_design_matrix = [ones(size(X_covariates,1),1), X_covariates]; % Add intercept

% Vectorized regression for all X columns at once
X_beta_coeffs = X_design_matrix \ X; % Solve for all columns simultaneously
X_predicted = X_design_matrix * X_beta_coeffs;
X = X - X_predicted; % Residuals

clear Y_design_matrix Y_beta_coeffs Y_predicted X_design_matrix X_beta_coeffs X_predicted;

[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X, Y, ncomp); PCTVAR

% % Visualize correlations and scatter plots
figure(1);
corr_values = corr(XS(:,1),Y);
imagesc(corr_values); 
colormap bone;
colorbar; % Add colorbar to show correlation values
title('Correlation between XS component 1 and Y variables');

% Add axis labels for better interpretation
set(gca, 'XTick', 1:length(Y_columns), 'XTickLabel', Y_columns);
set(gca, 'YTick', 1, 'YTickLabel', 'XS Component 1');
xlabel('Y Variables');
ylabel('XS Component');

% Display the actual correlation values on the plot
for i = 1:length(corr_values)
    text(i, 1, sprintf('%.3f', corr_values(i)), ...
         'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
end

% Set colorbar label
cb = colorbar;
cb.Label.String = 'Correlation Coefficient';
cb.Label.FontSize = 12;

% Display correlations between XS components and Y variables
corr_XS_Y = corr(XS, Y);
fprintf('Correlations between XS components and Y variables:\n');
disp(corr_XS_Y);

% Create scatter plots for all combinations of XS components (1:3) and Y variables (1:4)
% Generate all combinations
[comp_idx, y_idx] = meshgrid(1:3, 1:4);
combs = [comp_idx(:), y_idx(:)];

figure(13);
for i = 1:length(combs)
    ix1 = combs(i,1);  % XS component index
    ix2 = combs(i,2);  % Y variable index
    
    subplot(4,3,i);
    scatter(XS(:,ix1), Y(:,ix2), 5, 'b', 'filled');
    
    % Fit linear model and add regression line with confidence intervals
    mdl = fitlm(XS(:,ix1), Y(:,ix2));
    [ypred, yci] = predict(mdl, XS(:,ix1), 'Alpha', 0.001);
    
    hold on;
    plot(XS(:,ix1), ypred, 'k', 'LineWidth', 2);
    plot(XS(:,ix1), yci(:,1), 'k', 'LineWidth', 0.5);
    plot(XS(:,ix1), yci(:,2), 'k', 'LineWidth', 0.5);
    
    xlabel(['XS Component ', num2str(ix1)]);
    ylabel(['Y Variable ', num2str(ix2)]);
    title(['XS', num2str(ix1), ' vs Y', num2str(ix2)]);
    hold off;
end

% Clean up variables
clear mdl ypred yci ix1 ix2 combs comp_idx y_idx;

% Check correlation between age and XS components
age_XS_corr = corr(age, XS);
fprintf('Correlations between age and XS components:\n');
disp(age_XS_corr);

% Find maximum correlation between X and Y
max_corr_XY = max(corr(X, Y));
fprintf('Maximum correlation between X and Y: %.4f\n', max_corr_XY);

%% Permutation testing
fprintf('Starting permutation testing...\n');
permutations = 5000;   
allobservations = Y; 

% Wrap large arrays in parallel.pool.Constant
X_const = parallel.pool.Constant(X);
allobservations_const = parallel.pool.Constant(allobservations);

for ncomp_test = 1:3
    fprintf('Testing %d components...\n', ncomp_test);
    
    % Initialize arrays for this component count
    Rsq = zeros(permutations, 1);
    Rsq1 = zeros(permutations, 1);
    
    parfor n = 1:permutations
        % selecting random permutation
        permutation_index = randperm(length(allobservations_const.Value));
        % creating random sample based on permutation index
        randomSample = allobservations_const.Value(permutation_index,:);
        % running the PLS for this permutation
        [~,~,~,~,~,PCTVAR_perm,~,~] = plsregress(X_const.Value,randomSample,ncomp_test);
        Rsq(n) = sum(PCTVAR_perm(2,:));
        Rsq1(n) = sum(PCTVAR_perm(1,:));
    end
    
    % Run actual PLS for comparison (use original variables since this is outside parfor)
    [XL_actual,YL_actual,XS_actual,YS_actual,BETA_actual,PCTVAR_actual,MSE_actual,stats_actual] = plsregress(X,Y,ncomp_test);
    
    % Calculate p-values
    p(ncomp_test) = sum(sum(PCTVAR_actual(2,:)) < Rsq) / permutations;
    p_1(ncomp_test) = sum(sum(PCTVAR_actual(1,:)) < Rsq1) / permutations;
    
    fprintf('Component %d: p-value (Y) = %.4f, p-value (X) = %.4f\n', ncomp_test, p(ncomp_test), p_1(ncomp_test));
end


% Figure to show the permutation distribution vs actual distribution
figure(2); 
histogram(Rsq); 
xlim([min(Rsq)-0.01, max(max(Rsq), sum(PCTVAR(2,:)))+0.01]); 
xlabel('R-squared (Y variance explained)');
ylabel('Frequency');
title('Permutation Test Distribution');
hold on;
actual_rsq = sum(PCTVAR(2,:));
line([actual_rsq actual_rsq], ylim, 'Color', 'r', 'LineWidth', 2);
legend('Permutation Distribution', 'Actual R-squared');
fprintf('Actual R-squared: %.4f\n', actual_rsq);


%% Bootstrapping to get the func connectivity weights for PLS1, 2 and 3
fprintf('Starting bootstrapping analysis...\n');
dim = 3;

PLS1w = stats.W(:,1);
PLS2w = stats.W(:,2);
PLS3w = stats.W(:,3);

bootnum = 5000;
PLS1weights = zeros(size(X,2), bootnum);
PLS2weights = zeros(size(X,2), bootnum);
PLS3weights = zeros(size(X,2), bootnum);

% Wrap large arrays in parallel.pool.Constant for bootstrapping
X_const = parallel.pool.Constant(X);
Y_const = parallel.pool.Constant(Y);
PLS1w_const = parallel.pool.Constant(PLS1w);
PLS2w_const = parallel.pool.Constant(PLS2w);
PLS3w_const = parallel.pool.Constant(PLS3w);

fprintf('Running %d bootstrap iterations...\n', bootnum);
parfor i = 1:bootnum
    myresample = randsample(size(X_const.Value,1), size(X_const.Value,1), 1);
    Xr = X_const.Value(myresample,:); % define X for resampled subjects
    Yr = Y_const.Value(myresample,:); % define Y for resampled subjects
    [~,~,~,~,~,~,~,stats_boot] = plsregress(Xr,Yr,dim);
      
    newW = stats_boot.W(:,1);
    if corr(PLS1w_const.Value,newW) < 0
        newW = -1*newW;
    end
    PLS1weights(:,i) = newW; % Instead of concatenation
    
    newW = stats_boot.W(:,2);
    if corr(PLS2w_const.Value,newW) < 0
        newW = -1*newW;
    end
    PLS2weights(:,i) = newW; % Instead of concatenation
    
    newW = stats_boot.W(:,3);
    if corr(PLS3w_const.Value,newW) < 0
        newW = -1*newW;
    end
    PLS3weights(:,i) = newW; % Instead of concatenation
end

fprintf('Calculating bootstrap statistics...\n');
PLS1sw = std(PLS1weights');
PLS2sw = std(PLS2weights');
PLS3sw = std(PLS3weights');

plsweights1 = PLS1w ./ PLS1sw';
plsweights2 = PLS2w ./ PLS2sw'; 
plsweights3 = PLS3w ./ PLS3sw';

fprintf('PLS1 weights > 3: %d\n', sum(plsweights1 > 3));
fprintf('PLS1 weights < -3: %d\n', sum(plsweights1 < -3));
fprintf('PLS2 weights > 3: %d\n', sum(plsweights2 > 3));
fprintf('PLS2 weights < -3: %d\n', sum(plsweights2 < -3));
fprintf('PLS3 weights > 3: %d\n', sum(plsweights3 > 3));
fprintf('PLS3 weights < -3: %d\n', sum(plsweights3 < -3));

% Clean up bootstrap variables
clear XL_boot YL_boot XS_boot YS_boot BETA_boot PCTVAR_boot MSE_boot stats_boot;
clear newW myresample Xr Yr;

%% Visuals
fprintf('Creating circular graph visualization...\n');

% Since your CSV already has the 210 unique column names, we can use them directly
% Get the FC region names that were used to create X
fc_region_names = regions; % These are the column names from FC_regions.txt

% Create the network labels from your actual column names
% Assuming your column names follow a pattern like 'NetworkA_NetworkB' or similar
% Extract unique network names
unique_networks = {};
for i = 1:length(fc_region_names)
    % Split the region name to get individual networks
    % Adjust this parsing based on your actual column naming convention
    region_name = char(fc_region_names(i));
    
    % Example: if your names are like 'IC1-IC7', 'IC1-IC9', etc.
    if contains(region_name, '-')
        parts = split(region_name, '-');
        for j = 1:length(parts)
            if ~ismember(parts{j}, unique_networks)
                unique_networks{end+1} = parts{j};
            end
        end
    end
end

% If your regions don't follow the IC naming, create generic labels
if isempty(unique_networks)
    D = 21; % Assuming 21 networks based on your original code
    unique_networks = {};
    for i = 1:D
        unique_networks{i} = sprintf('Network%d', i);
    end
else
    D = length(unique_networks);
end

% Create the connectivity matrix structure
fprintf('Number of unique networks detected: %d\n', D);

% Set visualization parameters
upperlim = 4; 
lowerlim = -4;

% Create weights matrix
Weights = plsweights1; % Use the bootstrap z-scores directly
Weights(Weights < upperlim & Weights > lowerlim) = 0; % Threshold weights

% Create figure
figure;

% Plot positive weights (red)
Weights_pos = Weights;
Weights_pos(Weights_pos < 0) = 0;
if any(Weights_pos > 0)
    % Create square matrix for positive weights
    Weights_square_pos = zeros(D);
    % Map the weights to upper triangle (adjust indexing based on your data structure)
    if length(Weights_pos) == D*(D-1)/2
        Weights_square_pos(triu(ones(D),1) > 0) = Weights_pos;
    end
    
    myColorMap_pos = zeros(D, 3);
    myColorMap_pos(:, 1) = 1; % Red for positive
    circularGraph(Weights_square_pos, 'Colormap', myColorMap_pos, 'Label', unique_networks);
end

hold on;

% Plot negative weights (blue)
Weights_neg = Weights;
Weights_neg(Weights_neg > 0) = 0;
if any(Weights_neg < 0)
    % Create square matrix for negative weights
    Weights_square_neg = zeros(D);
    % Map the absolute values of negative weights to upper triangle
    if length(abs(Weights_neg)) == D*(D-1)/2
        Weights_square_neg(triu(ones(D),1) > 0) = abs(Weights_neg);
    end
    
    myColorMap_neg = zeros(D, 3);
    myColorMap_neg(:, 3) = 1; % Blue for negative
    circularGraph(Weights_square_neg, 'Colormap', myColorMap_neg, 'Label', unique_networks);
end

title('PLS Component 1 - Functional Connectivity Weights');
fprintf('Visualization complete. Positive weights in red, negative weights in blue.\n');

% Display summary statistics
fprintf('Significant positive weights (>%d): %d\n', upperlim, sum(plsweights1 > upperlim));
fprintf('Significant negative weights (<%d): %d\n', lowerlim, sum(plsweights1 < lowerlim));

toc