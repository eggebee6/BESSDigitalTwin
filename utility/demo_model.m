function [fig] = demo_model(model, testing_data, correct_action, predict_full_sequence)
  arguments
    model = [];
    testing_data = [];
    correct_action = [];
    predict_full_sequence = false;
  end
  gp = global_params();
  
  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(testing_data, dim_C);
  size_T = size(testing_data, dim_T);

  % Only use first sample from batch
  testing_data = testing_data(:, 1, :);
  size_B = 1;

  % Scale data
  testing_data(1:3, :) = testing_data(1:3, :) ./ gp.iLf_err_scale;
  testing_data(4:6, :) = testing_data(4:6, :) ./ gp.vCf_err_scale;
  testing_data(7:9, :) = testing_data(7:9, :) ./ gp.iLo_err_scale;

  if (predict_full_sequence)
    % Pass entire sequence through model
    encoder_output = predict(model.encoder, testing_data);
    latent_sample = predict(model.latent_sampler, encoder_output);
    decoder_output = predict(model.decoder, latent_sample);
    action_output = predict(model.action_recommender, latent_sample);

    % Get reconstruction from decoder output
    recon_data = decoder_output;

    [~, action_data] = max(action_output);
  else
    % Split data into sequences, getting prediction for each one
    num_sequences = ceil(size_T / gp.min_sequence_len) - 1;
    if (num_sequences < 1)
      error('Input sequence length is too short');
    end
  
    start_index = 1;
    recon_data = dlarray(zeros(9, size_B, size_T));
    action_data = dlarray(zeros(1, size_B, size_T));
    for i = 1:num_sequences
      end_index = start_index + gp.min_sequence_len - 1;
  
      % Forward data through model
      encoder_output = predict(model.encoder, testing_data(:, :, start_index:end_index));
      %latent_sample = predict(model.latent_sampler, encoder_output);
      encoder_means = encoder_output(1:size(encoder_output, 1)/2, :, :);
      decoder_output = predict(model.decoder, encoder_means);
      action_output = predict(model.action_recommender, encoder_means);
  
      % Add decoder output to overall reconstruction
      recon_data(:, :, start_index:end_index) = decoder_output;

      % Repeat action for plot purposes
      [~, max_action] = max(action_output);
      action_data(:, :, start_index:end_index) = max_action;
  
      start_index = start_index + gp.min_sequence_len;
    end
  end

  % Reshape for convenience
  testing_data = reshape(testing_data, [size_C size_T]);
  recon_data = reshape(recon_data, [size(recon_data, 1) size(recon_data, 3)]);
  action_data = reshape(action_data, [size(action_data, 1) size(action_data, 3)]);

  % Truncate data ranges to shorter length
  max_len = min([size(testing_data, 2), size(recon_data, 2)]);
  testing_data = testing_data(:, 1:max_len);
  recon_data = recon_data(:, 1:max_len);

  % Rescale data
  recon_data(1:3, :) = recon_data(1:3, :) .* gp.iLf_err_scale;
  recon_data(4:6, :) = recon_data(4:6, :) .* gp.vCf_err_scale;
  recon_data(7:9, :) = recon_data(7:9, :) .* gp.iLo_err_scale;

  testing_data(1:3, :) = testing_data(1:3, :) .* gp.iLf_err_scale;
  testing_data(4:6, :) = testing_data(4:6, :) .* gp.vCf_err_scale;
  testing_data(7:9, :) = testing_data(7:9, :) .* gp.iLo_err_scale;

  testing_err_vec = testing_data(1:9, :);

  % Set plot parameters
  x_range = (1:max_len) ./ gp.Fs;
  start_skip = 2;

  y_lim_ilf = max([...
    max(testing_err_vec(1:3, start_skip:end), [], 'all'), ...
    max(recon_data(1:3, start_skip:end), [], 'all'), ...
    10
  ]);
  y_lim_ilf = [0 extractdata(y_lim_ilf)];

  y_lim_vcf = max([...
    max(testing_err_vec(4:6, start_skip:end), [], 'all'), ...
    max(recon_data(4:6, start_skip:end), [], 'all'), ...
    4
  ]);
  y_lim_vcf = [0 extractdata(y_lim_vcf)];

  y_lim_ilo = max([...
    max(testing_err_vec(7:9, start_skip:end), [], 'all'), ...
    max(recon_data(7:9, start_skip:end), [], 'all'), ...
    4
  ]);
  y_lim_ilo = [0 extractdata(y_lim_ilo)];

  y_lim_action = [0 model.label_count + 1];

  % Create plots
  fig = tiledlayout(4, 2);

  % Plot action recommendation histograms
  [~, max_action] = max(correct_action);
  action_data_len = size(action_data, 2);
  action_x_range = (1:action_data_len) ./ gp.Fs;

  rec_hist = histcounts(extractdata(action_data), 1:model.label_count+1) ./ action_data_len;

  nexttile;
  bar([rec_hist', correct_action]);
  yline(max(rec_hist), ':');
  ylim([0 1]);
  title('Recommendations');
  xlabel('Recommendation');
  ylabel('Action');
  legend('Recommended', 'Correct', ...
    'Location', 'northeastoutside');

  % Plot action recommendations over time
  nexttile;
  plot(...
    action_x_range, action_data(1, :)', '.', ...
    action_x_range, repmat(max_action, size_B, action_data_len)', '-');
  xlim('tight');
  ylim(y_lim_action);
  title('Recommendations over time');
  xlabel('Time (s)');
  ylabel('Action')

  % Plot iLf error
  nexttile([1 2]);
  hold on;
  data_range = 1:3;
  plot(...
    x_range, testing_err_vec(data_range(1), :)', 'r-', ...
    x_range, testing_err_vec(data_range(2), :)', 'g-', ...
    x_range, testing_err_vec(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');
  xlim('tight');
  ylim(y_lim_ilf);
  title('Lf error');
  xlabel('Time (s)');
  ylabel('Current (A)');

  % Plot vCf error
  nexttile([1 2]);
  hold on;
  data_range = 4:6;
  plot(...
    x_range, testing_err_vec(data_range(1), :)', 'r-', ...
    x_range, testing_err_vec(data_range(2), :)', 'g-', ...
    x_range, testing_err_vec(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');
  xlim('tight');
  ylim(y_lim_vcf);
  title('Cf error');
  xlabel('Time (s)');
  ylabel('Voltage (V)');

  % Plot iLo error
  nexttile([1 2]);
  hold on;
  data_range = 7:9;
  plot(...
    x_range, testing_err_vec(data_range(1), :)', 'r-', ...
    x_range, testing_err_vec(data_range(2), :)', 'g-', ...
    x_range, testing_err_vec(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');
  xlim('tight');
  ylim(y_lim_ilo);
  title('Lo error');
  xlabel('Time (s)');
  ylabel('Current (A)');

end
