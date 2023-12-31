function [losses, grads, training_params] = evaluate_resnet_recon(model, training_data, error_vectors, training_params, grads)
  losses = [];

  encoder_output = [];
  latent_sample = [];
  decoder_output = [];

  % Debug stuff
  debug_max_loss = 1e20;
  
try
%% Initialize parameters
  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(training_data, dim_C);
  size_B = size(training_data, dim_B);
  size_T = size(training_data, dim_T);

  monte_carlo_reps = training_params.monte_carlo_reps;

  latent_dims = model.latent_dims;

%% Encode input
  encoder_output = forward(model.encoder, training_data);

  % Debug stuff
  %if (any(~isfinite(encoder_output), 'all'))
  %  error('Bad value in network output');
  %end

  % First half of encoder output is means, second half is logvars
  encoder_means = encoder_output(1:latent_dims, :, :);
  encoder_vars = exp(encoder_output(latent_dims+1:latent_dims*2, :, :));

  % Calculate KL loss
  kl_loss = 0.5 * (...
    sum(encoder_vars, 1) + ...
    -latent_dims + ...
    sum(encoder_means.^2, 1) + ...
    -sum(log(encoder_vars), 1) ...
  );

  %if any(~isfinite(kl_loss), 'all')
  %  error('Bad value in KL loss calculation');
  %elseif any(kl_loss < 0, 'all')
  %  error('Negative in KL loss calculation');
  %end

  % Average KL losses across time and across entire batch
  kl_loss = mean(kl_loss, dim_T);
  kl_loss = mean(kl_loss, dim_B);

%% Sample latent space and reconstruct input
  recon_loss = dlarray(0);

  for i = 1:monte_carlo_reps
    % Get latent sample
    latent_sample = forward(model.latent_sampler, encoder_output);

    % Reconstruct output
    decoder_output = forward(model.decoder, latent_sample);
 
    % Debug stuff
    %if (any(~isfinite(latent_sample), 'all') || ...
    %    any(~isfinite(decoder_output), 'all'))
    %  error('Bad value in network outputs');
    %end
  
    % Calculate reconstruction loss
    recon_loss = recon_loss + mse(decoder_output, error_vectors);

    % Debug stuff
    %if ~isfinite(recon_loss)
    %  error('Bad value in reconstruction loss');
    %elseif (recon_loss < 0)
    %  error('Negative in reconstruction loss');
    %end
  end

  recon_loss = recon_loss ./ monte_carlo_reps;

%% Calculate loss and gradients

  % Get reconstruction loss
  losses.recon_loss = recon_loss * training_params.recon_loss_factor;

  % Adjust KL loss factor and get KL loss
  kl_scaling_loss = losses.recon_loss;
  if (kl_scaling_loss > 0)
    training_params.kl_loss_factor = min([training_params.min_kl_scaling_loss / kl_scaling_loss, 1]);
    if (kl_scaling_loss < training_params.min_kl_scaling_loss)
      training_params.min_kl_scaling_loss = kl_scaling_loss;
    end
  end

  losses.kl_loss = kl_loss * training_params.kl_loss_factor;
  
  % Calculate total loss
  losses.total_recon_loss = ...
    losses.recon_loss + ...
    losses.kl_loss;

  % Get gradients
  [grads.decoder, grads.encoder_recon] = ...
    dlgradient(losses.total_recon_loss, ...
      model.decoder.Learnables, ...
      model.encoder.Learnables);

  % Debug stuff
  %if (any(~isfinite(grads.decoder{1, 3}{1}), 'all') || ...
  %    any(~isfinite(grads.encoder{1, 3}{1}), 'all'))
  %  error('Bad gradient');
  %end
  if (losses.total_recon_loss > 1e6)
    error('Loss is too high');
  end

catch ex
  save('eval_recon_debug.mat', 'model', 'training_params', ...
    'encoder_output', 'latent_sample', 'decoder_output', ...
    'losses', 'grads');
  rethrow(ex);
end

end

%% References
%
% [1] Kingma, Diederik P, Max, Welling. "Auto-encoding variational bayes". https://arxiv.org/abs/1312.6114
%
