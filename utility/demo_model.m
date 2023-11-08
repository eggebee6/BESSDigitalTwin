function [fig] = demo_model(model, error_vectors, action, predict_full_sequence)
  arguments
    model = [];
    error_vectors = [];
    action = [];
    predict_full_sequence = false;
  end
  gp = global_params();
  
  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(error_vectors, dim_C);
  size_T = size(error_vectors, dim_T);

  % Only use first sample from batch
  error_vectors = error_vectors(:, 1, :);
  size_B = 1;

  % Scale data
  error_vectors(1:3, :) = error_vectors(1:3, :) ./ gp.iLf_err_scale;
  error_vectors(4:6, :) = error_vectors(4:6, :) ./ gp.vCf_err_scale;
  error_vectors(7:9, :) = error_vectors(7:9, :) ./ gp.iLo_err_scale;

  if (predict_full_sequence)
    % Pass entire sequence through model
    encoder_output = predict(model.encoder, error_vectors);
    latent_sample = predict(model.latent_sampler, encoder_output);
    decoder_output = predict(model.decoder, latent_sample);
    action_output = predict(model.action_recommender, latent_sample);

    % Get reconstruction from decoder output
    recon_data = decoder_output;

    % Repeat action for plot purposes
    [~, action_data] = max(action_output);
  else
    % Split data into sequences, getting prediction for each one
    num_sequences = floor(size_T / gp.min_sequence_len) - 1;
    if (num_sequences < 1)
      error('Input sequence length is too short');
    end
  
    start_index = 1;
    recon_data = dlarray(zeros(size_C, size_B, size_T));
    action_data = dlarray(zeros(1, size_B, size_T));
    for i = 1:num_sequences
      end_index = start_index + gp.min_sequence_len - 1;
  
      % Forward data through model
      encoder_output = predict(model.encoder, error_vectors(:, :, start_index:end_index));
      latent_sample = predict(model.latent_sampler, encoder_output);
      decoder_output = predict(model.decoder, latent_sample);
      action_output = predict(model.action_recommender, latent_sample);
  
      % Add decoder output to overall reconstruction
      recon_data(:, :, start_index:end_index) = decoder_output;

      % Repeat action for plot purposes
      [~, max_action] = max(action_output);
      action_data(:, :, start_index:end_index) = max_action;
  
      start_index = start_index + gp.min_sequence_len;
    end
  end

  % Reshape for convenience
  error_vectors = reshape(error_vectors, [size_C size_T]);
  recon_data = reshape(recon_data, [size(recon_data, 1) size(recon_data, 3)]);
  action_data = reshape(action_data, [size(action_data, 1) size(action_data, 3)]);

  [~, max_action] = max(action);
  action = repmat(max_action, size_B, size_T);

  % Truncate data ranges to shorter length
  max_len = min([size(error_vectors, 2), size(recon_data, 2), size(action_data, 2)]);
  error_vectors = error_vectors(:, 1:max_len);
  recon_data = recon_data(:, 1:max_len);
  action_data = action_data(:, 1:max_len);

  % Rescale data
  recon_data(1:3, :) = recon_data(1:3, :) .* gp.iLf_err_scale;
  recon_data(4:6, :) = recon_data(4:6, :) .* gp.vCf_err_scale;
  recon_data(7:9, :) = recon_data(7:9, :) .* gp.iLo_err_scale;

  error_vectors(1:3, :) = error_vectors(1:3, :) .* gp.iLf_err_scale;
  error_vectors(4:6, :) = error_vectors(4:6, :) .* gp.vCf_err_scale;
  error_vectors(7:9, :) = error_vectors(7:9, :) .* gp.iLo_err_scale;

  % Set plot parameters
  x_range = (1:max_len) ./ gp.Fs;
  start_skip = 2;

  y_lim_ilf = max([...
    max(error_vectors(1:3, start_skip:end), [], 'all'), ...
    max(recon_data(1:3, start_skip:end), [], 'all'), ...
    10
  ]);
  y_lim_ilf = [0 extractdata(y_lim_ilf)];

  y_lim_vcf = max([...
    max(error_vectors(4:6, start_skip:end), [], 'all'), ...
    max(recon_data(4:6, start_skip:end), [], 'all'), ...
    4
  ]);
  y_lim_vcf = [0 extractdata(y_lim_vcf)];

  y_lim_ilo = max([...
    max(error_vectors(7:9, start_skip:end), [], 'all'), ...
    max(recon_data(7:9, start_skip:end), [], 'all'), ...
    4
  ]);
  y_lim_ilo = [0 extractdata(y_lim_ilo)];

  y_lim_action = [0 10];    % TODO: [0 model.label_count]

  % Create plots
  fig = tiledlayout(4, 1);

  % Plot iLf error
  nexttile;
  hold on;
  data_range = 1:3;
  plot(...
    x_range, error_vectors(data_range(1), :)', 'r-', ...
    x_range, error_vectors(data_range(2), :)', 'g-', ...
    x_range, error_vectors(data_range(3), :)', 'b-');
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
  nexttile;
  hold on;
  data_range = 4:6;
  plot(...
    x_range, error_vectors(data_range(1), :)', 'r-', ...
    x_range, error_vectors(data_range(2), :)', 'g-', ...
    x_range, error_vectors(data_range(3), :)', 'b-');
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
  nexttile;
  hold on;
  data_range = 7:9;
  plot(...
    x_range, error_vectors(data_range(1), :)', 'r-', ...
    x_range, error_vectors(data_range(2), :)', 'g-', ...
    x_range, error_vectors(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');
  xlim('tight');
  ylim(y_lim_ilo);
  title('Lo error');
  xlabel('Time (s)');
  ylabel('Current (A)');

  % Plot actions
  nexttile;
  hold on;
  plot(...
    x_range, action_data(1, :)', 'b.', ...
    x_range, action(1, :)', 'g-');
  xlim('tight');
  ylim(y_lim_action);
  title('Recommended action');
  xlabel('Time (s)');
  ylabel('Action')

end
