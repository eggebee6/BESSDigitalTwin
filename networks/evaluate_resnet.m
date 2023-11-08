function [losses, grads, training_params] = evaluate_resnet(model, error_vectors, labels, training_params)
  losses = [];
  grads = [];

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
  size_C = size(error_vectors, dim_C);
  size_B = size(error_vectors, dim_B);
  size_T = size(error_vectors, dim_T);

  monte_carlo_reps = training_params.monte_carlo_reps;

  latent_dims = model.latent_dims;

  labels = dlarray(repmat(labels, 1, 1, size_T), 'CBT');

%% Encode input
  encoder_output = forward(model.encoder, error_vectors);

  % Debug stuff
  if (any(~isfinite(encoder_output), 'all'))
    error('Bad value in network output');
  end

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

  if any(~isfinite(kl_loss), 'all')
    error('Bad value in KL loss calculation');
  elseif any(kl_loss < 0, 'all')
    error('Negative in KL loss calculation');
  end

  % Average KL losses across time and across entire batch
  kl_loss = mean(kl_loss, dim_T);
  kl_loss = mean(kl_loss, dim_B);

%% Sample latent space, reconstruct input, and predict action
  recon_loss = dlarray(0);
  action_loss = dlarray(0);

  for i = 1:monte_carlo_reps
    % Get latent sample
    latent_sample = forward(model.latent_sampler, encoder_output);

    % Reconstruct output
    decoder_output = forward(model.decoder, latent_sample);

    % Get action
    action_output = forward(model.action_recommender, latent_sample);
    model.action_recommender = resetState(model.action_recommender);    % TODO: Is this necessary?

    % Debug stuff
    if (any(~isfinite(latent_sample), 'all') || ...
        any(~isfinite(decoder_output), 'all'))
      error('Bad value in network outputs');
    end
  
    % Calculate reconstruction loss
    recon_loss = recon_loss + mse(decoder_output, error_vectors);

    % Debug stuff
    if ~isfinite(recon_loss)
      error('Bad value in reconstruction loss');
    elseif (recon_loss < 0)
      error('Negative in reconstruction loss');
    end

    % Calculate action loss and reset state
    action_loss = action_loss + crossentropy(action_output, labels);

    % Debug stuff
    if ~isfinite(action_loss)
      error('Bad value in action loss');
    elseif (action_loss < 0)
      error('Negative in action loss');
    end
  end

  recon_loss = recon_loss ./ monte_carlo_reps;
  action_loss = action_loss ./ monte_carlo_reps;

%% Calculate loss and gradients
% Total loss is the sum of reconstruction loss and KL divergence loss

  % Get reconstruction loss and update KL loss factor
  losses.recon_loss = recon_loss * training_params.recon_loss_factor;
  if (losses.recon_loss > 0)
    training_params.kl_loss_factor = min([training_params.min_recon_loss / losses.recon_loss, 1]);
    if (losses.recon_loss < training_params.min_recon_loss)
      training_params.min_recon_loss = losses.recon_loss;
    end
  end

  % Get KL loss
  losses.kl_loss = kl_loss * training_params.kl_loss_factor;

  % Get action loss
  losses.action_loss = action_loss * training_params.action_loss_factor;
  
  % Calculate total loss
  losses.total_loss = ...
    losses.recon_loss + ...
    losses.kl_loss + ...
    losses.action_loss;

  % Get gradients
  [grads.decoder, grads.encoder, grads.action_recommender] = ...
    dlgradient(losses.total_loss, ...
      model.decoder.Learnables, ...
      model.encoder.Learnables, ...
      model.action_recommender.Learnables);

  % Debug stuff
  if (any(~isfinite(grads.decoder{1, 3}{1}), 'all') || ...
      any(~isfinite(grads.encoder{1, 3}{1}), 'all') || ...
      any(~isfinite(grads.action_recommender{1, 3}{1}), 'all'))
    error('Bad gradient');
  end
  if (losses.total_loss > 1e6)
    error('Loss is too high');
  end

catch ex
  save('eval_debug.mat', 'model', 'training_params', ...
    'encoder_output', 'latent_sample', 'decoder_output', 'action_output', ...
    'losses', 'grads');
  rethrow(ex);
end

end

%% References
%
% [1] Kingma, Diederik P, Max, Welling. "Auto-encoding variational bayes". https://arxiv.org/abs/1312.6114
%
