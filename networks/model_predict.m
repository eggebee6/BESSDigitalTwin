function [actions, error_vector_recon] = model_predict(model, dt_info, max_len)
  arguments
    model = [];
    dt_info = [];
    max_len = 0;
  end
  gp = global_params();

  encoder_output = [];
  error_vector_recon = [];
  actions = [];

  if (max_len < 1)
    max_len = gp.samples_per_cycle * 10;
  end

try
  %% Initialize
  gp = global_params();

  % Get model input data
  data = DTInfo.get_model_input(dt_info);
  data_len = min([size(data, 3), max_len]);
  data = data(:, :, 1:data_len);

  % Scale data
  data(1:3, :) = data(1:3, :) ./ gp.iLf_err_scale;
  data(4:6, :) = data(4:6, :) ./ gp.vCf_err_scale;
  data(7:9, :) = data(7:9, :) ./ gp.iLo_err_scale;

  %% Forward data through model
  encoder_output = forward(model.encoder, data);

  % Mean is the max-likelihood estimator, no sampling for prediction
  encoder_means = encoder_output(1:model.latent_dims, :, :);

  % Reconstruct error vectors
  error_vector_recon = forward(model.decoder, encoder_means);

  % Get action recommendations
  actions = forward(model.action_recommender, encoder_means);

  % Rescale reconstructed data
  error_vector_recon(1:3, :) = error_vector_recon(1:3, :) .* gp.iLf_err_scale;
  error_vector_recon(4:6, :) = error_vector_recon(4:6, :) .* gp.vCf_err_scale;
  error_vector_recon(7:9, :) = error_vector_recon(7:9, :) .* gp.iLo_err_scale;

catch ex
  rethrow(ex);
end

end
