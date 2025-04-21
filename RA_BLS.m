function [best_accuracy, best_rmse, best_iteration, pseudoinverse_times,accuracy_history,rmse_history,residual_history] = RA_BLS(train_x, train_y, test_x, test_y, scaling_param, lambda, features_per_window, num_windows, num_enhancements, max_iterations,try_gpu)
    % [Function declaration and comments remain the same]

    % Initialize variables
    enhancement_nodes = {};
    enhancement_weights = {};
    residual_history = [];
    accuracy_history = [];
    rmse_history = [];
    pseudoinverse_times = [];
    best_accuracy = 0;
    best_rmse = Inf;
    best_iteration = 0;

    %% Training Phase
    tic;
    %% Feature Node Generation
    % Concatenate input data with bias term
    input_with_bias = [train_x .1 * ones(size(train_x, 1), 1)];
    feature_nodes = zeros(size(train_x, 1), num_windows * features_per_window);
    % Clean up memory by clearing unused variables
    clear train_x;

    % Generate feature nodes for each window
    for i = 1:num_windows
        % Initialize random weights for feature mapping
        feature_weights_random = 2 * rand(size(input_with_bias, 2), features_per_window) - 1;
        
        % Linear transformation
        linear_features = input_with_bias * feature_weights_random;
        % Clean up memory by clearing unused variables
        clear feature_weights_random;
        
        linear_features = mapminmax(linear_features);
        
        % Calculate sparse weights
        sparse_weights = sparse_bls(linear_features, input_with_bias, 1e-3, 50)';
        % Clean up memory by clearing unused variables
        clear linear_features; 
        
        sparse_weights_matrix{i} = sparse_weights;
        
        % Extract and normalize window features
        window_feature_nodes = input_with_bias * sparse_weights;
        % Clean up memory by clearing unused variables
        clear sparse_weights;
        
        [window_feature_nodes, feature_params] = mapminmax(window_feature_nodes', 0, 1);
        window_feature_nodes = window_feature_nodes';
        feature_scaling_params(i) = feature_params;
        
        % Store features for current window
        feature_nodes(:, features_per_window * (i - 1) + 1:features_per_window * i) = window_feature_nodes;
        % Clean up memory by clearing unused variables
        clear window_feature_nodes feature_params;
    end

    %% Enhancement Node Generation
    % Concatenate Feature Node with bias term
    enhancement_input_with_bias = [feature_nodes .1 * ones(size(feature_nodes, 1), 1)];

    % Initialize random weights for enhancement mapping
    if num_windows * features_per_window >= num_enhancements
        enhancement_weights_random = orth(2 * rand(num_windows * features_per_window + 1, num_enhancements) - 1);
    else
        enhancement_weights_random = orth(2 * rand(num_windows * features_per_window + 1, num_enhancements)' - 1)';
    end

    % Calculate raw enhancement nodes
    enhancement_nodes_inactive = enhancement_input_with_bias * enhancement_weights_random;
    enhancement_weights{end + 1} = enhancement_weights_random;

    % Scale enhancement nodes
    scaling_factor = scaling_param / max(max(enhancement_nodes_inactive));
    enhancement_nodes_activated = tansig(enhancement_nodes_inactive * scaling_factor);

    training_time_phase1 = toc;

    % Store enhancement nodes and combine with feature nodes
    enhancement_nodes{end + 1} = enhancement_nodes_activated;
    hidden_nodes = [feature_nodes enhancement_nodes_activated];

    %% Calculate Output
    tic;
    % Calculate output weights pseudoinverse
    output_weights = (hidden_nodes' * hidden_nodes + eye(size(hidden_nodes', 1)) * lambda) \ (hidden_nodes' * train_y);
    pseudoinverse_time = toc;
    pseudoinverse_times = [pseudoinverse_times pseudoinverse_time];

    % Calculate training accuracy
    training_output = hidden_nodes * output_weights;
    predicted_labels = result(training_output);
    true_labels = result(train_y);
    training_accuracy = length(find(predicted_labels == true_labels)) / size(true_labels, 1);
    training_rmse = sqrt(mse(training_output - train_y));
    total_training_time = training_time_phase1 + pseudoinverse_time;
    
    % Display training results
    disp('----------------------------------------------------');
    disp(['Training Time: ', num2str(total_training_time), ' seconds']);
    disp(['Initial Training Accuracy: ', num2str(training_accuracy * 100), ' %']);
    disp(['Initial Training RMSE: ', num2str(training_rmse)]);
    fprintf('\n');
    
    % Clean up memory by clearing unused variables
    clear predicted_labels;

    %% Testing Phase
    tic;
    test_input_bias = [test_x .1 * ones(size(test_x, 1), 1)];
    test_feature_nodes = zeros(size(test_x, 1), num_windows * features_per_window);

    % Clean up memory by clearing unused variables
    clear test_x;

    % Generate test feature nodes
    for i = 1:num_windows
        sparse_weights = sparse_weights_matrix{i};
        feature_params = feature_scaling_params(i);
        window_test_feature_nodes = test_input_bias * sparse_weights;
        window_test_feature_nodes = mapminmax('apply', window_test_feature_nodes', feature_params)';
        test_feature_nodes(:, features_per_window * (i - 1) + 1:features_per_window * i) = window_test_feature_nodes;
        clear window_test_feature_nodes; % Clear after use
    end
    
    clear sparse_weights_matrix feature_scaling_params; % Clear after all iterations

    test_enhancement_input_with_bias = [test_feature_nodes .1 * ones(size(test_feature_nodes, 1), 1)];
    testing_time_phase1 = toc;
    tic;

    % Calculate test enhancement nodes
    test_enhancement_nodes_inactive = test_enhancement_input_with_bias * enhancement_weights_random;
    test_enhancement_nodes = tansig(test_enhancement_nodes_inactive * scaling_factor);
    test_hidden_nodes = [test_feature_nodes test_enhancement_nodes];
    test_output = test_hidden_nodes * output_weights;
    testing_time_phase2 = toc;

    % Calculate testing accuracy
    test_predicted_labels = result(test_output);
    test_true_labels = result(test_y);
    testing_accuracy = length(find(test_predicted_labels == test_true_labels)) / size(test_true_labels, 1);
    testing_rmse = sqrt(mse(test_output - test_y));

    % Store history
    accuracy_history = [accuracy_history [training_accuracy * 100; testing_accuracy * 100]];
    
    % Display test results
    disp(['Testing Time: ', num2str(testing_time_phase1 + testing_time_phase2), ' seconds']);
    disp(['Initial Testing Accuracy: ', num2str(testing_accuracy * 100), ' %']);
    disp(['Initial Testing RMSE: ', num2str(testing_rmse)]);
    fprintf('\n');

    %% Residual Attention Mechanism
    tic;

    if ~try_gpu
        % CPU computation path
        dim_attention = sqrt(features_per_window * num_windows);
        
        % Calculate attention matrices on CPU
        relevance_attention = enhancement_input_with_bias * enhancement_input_with_bias';
        attention_values = (relevance_attention * (1 / dim_attention))';
        clear relevance_attention; % Clear after use
        
        attention_exp = exp(attention_values);
        clear attention_values; % Clear after use
        
        attention_matrix = (attention_exp ./ sum(attention_exp, 1))';
        clear attention_exp; % Clear after use
        
        test_relevance_attention = test_enhancement_input_with_bias * enhancement_input_with_bias';
        test_attention_values = (test_relevance_attention * (1 / dim_attention))';
        clear test_relevance_attention; % Clear after use
        
        test_attention_exp = exp(test_attention_values);
        clear test_attention_values; % Clear after use
        
        test_attention_matrix = (test_attention_exp ./ sum(test_attention_exp, 1))';
        clear test_attention_exp; % Clear after use
    end
    
    % Clean up memory by clearing unused variables
    clear enhancement_weights_random enhancement_nodes;
    
    % Iterative with Residual Attention Mechanism
    t = 1;
    while t <= max_iterations
        disp('----------------------------------------------------');
        fprintf('Progress: %.1f%%\n', ceil((t / max_iterations) * 1000) / 10);
        
        % Separate the weights corresponding to the nodes
        Weight_enhancement_nodes = output_weights(features_per_window * num_windows + 1:end, :);

        %% Calculate residual attention
        gradient = (1 - enhancement_nodes_activated .^ 2);
        residual = (training_output - train_y);
        residual_history = [residual_history residual];
        residual_enhancement_nodes = (residual * Weight_enhancement_nodes') .* gradient;
        % Clean up memory by clearing unused variables
        clear residual gradient;
        
        % Update nodes
        residual_attention = attention_matrix * residual_enhancement_nodes;        
        enhancement_nodes_inactive = enhancement_nodes_inactive - residual_attention;

        % Clean up memory by clearing unused variables
        clear residual_attention;
        
        % Ativate nodes
        scaling_factor = scaling_param / max(max(enhancement_nodes_inactive));
        enhancement_nodes_activated = tansig(enhancement_nodes_inactive * scaling_factor);
        
        % Update combined nodes and output weights
        hidden_nodes = [feature_nodes enhancement_nodes_activated];
        tic;
        output_weights = (hidden_nodes' * hidden_nodes + eye(size(hidden_nodes', 1)) * lambda) \ (hidden_nodes' * train_y);
        pseudoinverse_times = [pseudoinverse_times toc];

        % Calculate training accuracy
        training_output = hidden_nodes * output_weights;
        predicted_labels = result(training_output);
        training_accuracy = length(find(predicted_labels == true_labels)) / size(true_labels, 1);
        training_rmse = sqrt(mse(training_output - train_y));
        clear predicted_labels; % Clear after use

        % Display training results
        disp(['The ', num2str(t), '-th Training Accuracy is : ', num2str(training_accuracy * 100), ' %' ]);

        %% Testing Phase
        test_residual_attention = test_attention_matrix * residual_enhancement_nodes;
        clear residual_enhancement_nodes; % Clear after use
        
        test_enhancement_nodes_inactive = test_enhancement_nodes_inactive - test_residual_attention;
        clear test_residual_attention; % Clear after use
        
        test_enhancement_nodes = tansig(test_enhancement_nodes_inactive * scaling_factor);
        test_hidden_nodes = [test_feature_nodes test_enhancement_nodes];
        test_output = test_hidden_nodes * output_weights;
        test_predicted_labels = result(test_output);
        testing_accuracy = length(find(test_predicted_labels == test_true_labels)) / size(test_true_labels, 1);
        testing_rmse = sqrt(mse(test_output - test_y));
        clear test_predicted_labels; % Clear after use
        
        % Track best performance
        if t == 1
            best_accuracy = testing_accuracy;
            best_rmse = testing_rmse;
            best_iteration = t;
        elseif testing_accuracy > best_accuracy
            best_accuracy = testing_accuracy;
            best_iteration = t;
        elseif testing_rmse < best_rmse
            best_rmse = testing_rmse;
        end
        
        disp(['The ', num2str(t), '-th Testing Accuracy is : ', num2str(testing_accuracy * 100), ' %  ' ]);
        disp(['The ', num2str(t), '-th Testing RMSE is : ', num2str(testing_rmse)]);
        disp(['The Best Accuracy is : ', num2str(best_accuracy * 100), ' % in ', num2str(best_iteration), '-th ']);
        disp(['The Best RMSE is : ', num2str(best_rmse), ' in ', num2str(best_iteration), '-th ']);

        % Store history
        accuracy_history = [accuracy_history [training_accuracy * 100; testing_accuracy * 100]];
        rmse_history = [rmse_history [training_rmse; testing_rmse]];
        
        t = t + 1;
    end
    
    % Display final results
    total_training_time = training_time_phase1 + sum(pseudoinverse_times);
    disp('---------------------------------------------------- ');
    disp('Training has been finished!');
    disp(['Total Training Time is : ', num2str(total_training_time), ' seconds' ]);
    disp(['Final Training Accuracy is : ', num2str(training_accuracy * 100), ' %' ]);
    disp(['Final Training RMSE is : ', num2str(training_rmse) ]);
    disp(['Best Testing Accuracy is : ', num2str(best_accuracy * 100), ' %']);
    disp(['Best Testing RMSE is : ', num2str(best_rmse)]);
    disp(['Best iteration is : ', num2str(best_iteration)]);
    fprintf('\n');
end
