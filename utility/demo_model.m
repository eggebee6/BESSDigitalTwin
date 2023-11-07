function demo_model(model, data, predict_full_sequence)
  arguments
    model = [];
    data = [];
    predict_full_sequence = false;
  end
  gp = global_params();
  
  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(data, dim_C);
  size_T = size(data, dim_T);

  % Only use first sample from batch
  data = data(:, 1, :);
  size_B = 1;

  % Scale data
  data(1:3) = data(1:3) ./ gp.iLf_err_scale;
  data(4:6) = data(4:6) ./ gp.vCf_err_scale;
  data(7:9) = data(7:9) ./ gp.iLo_err_scale;

  if (predict_full_sequence)
    % Pass entire sequence through model
    encoder_output = predict(model.encoder, data);
    latent_sample = predict(model.latent_sampler, encoder_output);
    decoder_output = predict(model.decoder, latent_sample);

    recon_data = decoder_output;
  else
    % Split data into sequences, getting prediction for each one
    num_sequences = floor(size_T / gp.min_sequence_len);
    if (num_sequences < 1)
      error('Input sequence length is too short');
    end
  
    start_index = 1;
    recon_data = dlarray(zeros(size_C, size_B, size_T));
    for i = 1:num_sequences
      end_index = start_index + gp.min_sequence_len - 1;
  
      % Forward data through model
      encoder_output = predict(model.encoder, data(:, :, start_index:end_index));
      latent_sample = predict(model.latent_sampler, encoder_output);
      decoder_output = predict(model.decoder, latent_sample);
  
      % Add predictions to overall reconstruction
      recon_data(:, :, start_index:end_index) = decoder_output;
  
      start_index = start_index + gp.min_sequence_len;
    end
  end

  % Reshape for convenience
  data = reshape(data, [size_C size_T]);
  recon_data = reshape(recon_data, [size(recon_data, 1) size(recon_data, 3)]);

  % Truncate data ranges to shorter length
  max_len = min([size(data, 2), size(recon_data, 2)]);
  data = data(:, 1:max_len);
  recon_data = recon_data(:, 1:max_len);

  % Rescale output
  recon_data(1:3) = recon_data(1:3) .* gp.iLf_err_scale;
  recon_data(4:6) = recon_data(4:6) .* gp.vCf_err_scale;
  recon_data(7:9) = recon_data(7:9) .* gp.iLo_err_scale;

  % Set plot parameters
  x_range = (1:max_len) ./ gp.Fs;
  start_skip = 2;

  y_lim_ilf = max([...
    max(data(1:3, start_skip:end), [], 'all'), ...
    max(recon_data(1:3, start_skip:end), [], 'all'), ...
    10
  ]);
  y_lim_ilf = [0 extractdata(y_lim_ilf)];

  y_lim_vcf = max([...
    max(data(4:6, start_skip:end), [], 'all'), ...
    max(recon_data(4:6, start_skip:end), [], 'all'), ...
    4
  ]);
  y_lim_vcf = [0 extractdata(y_lim_vcf)];

  y_lim_ilo = max([...
    max(data(7:9, start_skip:end), [], 'all'), ...
    max(recon_data(7:9, start_skip:end), [], 'all'), ...
    4
  ]);
  y_lim_ilo = [0 extractdata(y_lim_ilo)];

  % Create plots
  fig = tiledlayout(3, 1);

  % Plot iLf error
  nexttile;
  hold on;
  data_range = 1:3;
  plot(...
    x_range, data(data_range(1), :)', 'r-', ...
    x_range, data(data_range(2), :)', 'g-', ...
    x_range, data(data_range(3), :)', 'b-');
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
    x_range, data(data_range(1), :)', 'r-', ...
    x_range, data(data_range(2), :)', 'g-', ...
    x_range, data(data_range(3), :)', 'b-');
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
    x_range, data(data_range(1), :)', 'r-', ...
    x_range, data(data_range(2), :)', 'g-', ...
    x_range, data(data_range(3), :)', 'b-');
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
