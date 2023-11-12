function [losses, grads, training_params] = evaluate_resnet_action(model, training_data, labels, training_params, grads)
  losses = [];

  encoder_output = [];
  latent_sample = [];
  action_output = [];

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
  
  %labels = mean(labels, dim_T);

%% Encode input
  %encoder_output = forward(model.encoder, training_data);

  % Debug stuff
  %if (any(~isfinite(encoder_output), 'all'))
  %  error('Bad value in network output');
  %end

%% Sample latent space, reconstruct input, and predict action
  % Get latent sample
  %latent_sample = forward(model.latent_sampler, encoder_output);

  action_loss = dlarray(0);
  for i = 1:monte_carlo_reps
    % Get action
    %action_output = forward(model.action_recommender, latent_sample);
    action_output = forward(model.action_recommender, training_data);
    
    % Calculate action loss
    %action_output = mean(action_output, dim_T);
    action_loss = action_loss + crossentropy(action_output, labels);
  
    % Debug stuff
    %if ~isfinite(action_loss)
    %  error('Bad value in action loss');
    %elseif (action_loss < 0)
    %  error('Negative in action loss');
    %end
  end

  action_loss = action_loss ./ monte_carlo_reps;

%% Calculate loss and gradients

  % Get action loss
  losses.action_loss = action_loss * training_params.action_loss_factor;

  % Calculate total loss
  losses.total_action_loss = ...
    losses.action_loss;

  % Get gradients
  %[grads.encoder_action, grads.action_recommender] = ...
  %  dlgradient(losses.total_action_loss, ...
  %    model.encoder.Learnables, ...
  %    model.action_recommender.Learnables);
  grads.action_recommender = dlgradient(losses.total_action_loss, model.action_recommender.Learnables);

  % Debug stuff
  %if (any(~isfinite(grads.encoder{1, 3}{1}), 'all') || ...
  %    any(~isfinite(grads.action_recommender{1, 3}{1}), 'all'))
  %  error('Bad gradient');
  %end
  if (losses.total_action_loss > 1e6)
    error('Loss is too high');
  end

catch ex
  save('eval_action_debug.mat', 'model', 'training_params', ...
    'encoder_output', 'latent_sample', 'action_output', ...
    'losses', 'grads');
  rethrow(ex);
end

end
