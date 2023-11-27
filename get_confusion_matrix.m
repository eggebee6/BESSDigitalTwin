function matrix = get_confusion_matrix(model, batch)
  matrix = [];

try
%% Initialize
  % Get validation data from batch
  [data, labels, event_times] = next(batch);

  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(data, dim_C);
  size_B = size(data, dim_B);
  size_T = size(data, dim_T);

  event_times = floor(event_times ./ 16);
  event_times(event_times == 0) = 1;

  gp = global_params();

%% Forward data through model
  % Scale data
  data(1:3, :, :) = data(1:3, :, :) ./ gp.iLf_err_scale;
  data(4:6, :, :) = data(4:6, :, :) ./ gp.vCf_err_scale;
  data(7:9, :, :) = data(7:9, :, :) ./ gp.iLo_err_scale;

  data(19:21, :) = data(19:21, :) ./ gp.voltage_pu;

  % Encoder mean is the max-likelihood latent value, no need to sample
  encoder_output = predict(model.encoder, data);
  actions = predict(model.action_recommender, encoder_output(1:model.latent_dims, :, :));

  [~, max_actions] = max(actions);
  [~, max_labels] = max(labels);

  max_len = min([size(labels, 3), size(actions, 3)]);
  max_actions = max_actions(:, :, 1:max_len);
  max_labels = max_labels(:, :, 1:max_len);

  max_actions = reshape(max_actions, [size(max_actions, 2) size(max_actions, 3)]);
  max_labels = reshape(max_labels, [size(max_labels, 2) size(max_labels, 3)]);

  max_actions = extractdata(max_actions);
  max_labels = extractdata(max_labels);

  matrix = zeros(model.label_count);
  for i = 1:size_B
    matrix = matrix + confusionmat(max_actions(i, :), max_labels(i, :), 'Order', 1:model.label_count);
  end

catch ex
  save('debug_confusion.mat', 'max_actions', 'max_labels', 'i');
  rethrow(ex);
end

end
