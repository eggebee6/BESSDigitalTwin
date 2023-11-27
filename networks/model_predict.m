function [actions, error_vector_recon, latent_codes] = model_predict(model, dt_info, max_len)
  arguments
    model = [];
    dt_info = [];
    max_len = 0;
  end
  gp = global_params();

  encoder_output = [];
  error_vector_recon = [];
  actions = [];
  latent_codes = [];

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

  data(19:21, :) = data(19:21, :) ./ gp.voltage_pu;

  %% Forward data through model
  encoder_output = predict(model.encoder, data);

  % Mean is the max-likelihood estimator, no sampling for prediction
  latent_codes = encoder_output(1:model.latent_dims, :, :);

  % Reconstruct error vectors
  error_vector_recon = predict(model.decoder, latent_codes);

  % Get action recommendations
  actions = predict(model.action_recommender, latent_codes);

  % Rescale reconstructed data
  error_vector_recon(1:3, :) = error_vector_recon(1:3, :) .* gp.iLf_err_scale;
  error_vector_recon(4:6, :) = error_vector_recon(4:6, :) .* gp.vCf_err_scale;
  error_vector_recon(7:9, :) = error_vector_recon(7:9, :) .* gp.iLo_err_scale;

catch ex
  rethrow(ex);
end

end
