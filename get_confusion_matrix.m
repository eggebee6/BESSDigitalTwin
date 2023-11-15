function matrix = get_confusion_matrix(model, batch)
  matrix = [];

try
%% Initialize
  % Get validation data from batch
  [data, labels] = next(batch);

  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(data, dim_C);
  size_B = size(data, dim_B);
  size_T = size(data, dim_T);

  gp = global_params();

%% Forward data through model
  % Scale data
  data(1:3, :, :) = data(1:3, :, :) ./ gp.iLf_err_scale;
  data(4:6, :, :) = data(4:6, :, :) ./ gp.vCf_err_scale;
  data(7:9, :, :) = data(7:9, :, :) ./ gp.iLo_err_scale;

  % Encoder mean is the max-likelihood latent value, no need to sample
  encoder_output = predict(model.encoder, data);
  actions = predict(model.action_recommender, encoder_output(1:model.latent_dims, :, :));

  labels_len = size(labels, 3);
  actions = actions(:, :, 1:labels_len);

  actions = reshape(actions, size(actions, 1), []);
  labels = reshape(labels, size(labels, 1), []);

  [~, actions] = max(actions);
  [~, labels] = max(labels);

  %% Evaluate results
  matrix = confusionmat(extractdata(labels), extractdata(actions), 'Order', 1:model.label_count);

catch ex
  rethrow(ex);
end

end
