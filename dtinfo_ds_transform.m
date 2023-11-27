function [ds_out] = dtinfo_ds_transform(ds_in)
  persistent gp;
  if isempty(gp)
    gp = global_params();
  end
  
  % Scale data
  ds_in{1}(1:3, :) = ds_in{1}(1:3, :) ./ gp.iLf_err_scale;
  ds_in{1}(4:6, :) = ds_in{1}(4:6, :) ./ gp.vCf_err_scale;
  ds_in{1}(7:9, :) = ds_in{1}(7:9, :) ./ gp.iLo_err_scale;

  ds_in{1}(19:21, :) = ds_in{1}(19:21, :) ./ gp.voltage_pu;

  % Initialize output data
  training_sequence_len = 2 * gp.min_sequence_len;
  data_len = size(ds_in{1}, 2);
  num_cells = 2 * gp.strides_per_sequence * floor(data_len / training_sequence_len - 1);

  ds_out = cell(num_cells, 3);

  downsampled_event_time = floor(ds_in{3} / 16);
  onehot_label = repmat(DTInfo.get_scenario_label(ds_in{2}), [1 floor(training_sequence_len/16)]);
  onehot_label(:, 1:downsampled_event_time) = repmat(DTInfo.get_scenario_label("No events"), [1 downsampled_event_time]);

  % Fill training samples with input data
  start_index = 1;
  window_stride = floor(gp.min_sequence_len / gp.strides_per_sequence);
  for i = 1:num_cells
    % Get a training sample starting from a random offset into the sample window
    %rand_start = start_index + floor(0.5 * (rand() * gp.min_sequence_len) / gp.strides_per_sequence);
    rand_start = start_index;
    end_index = rand_start + (training_sequence_len - 1);

    % Copy measurements and errors into training sample
    ds_out{i, 1} = ds_in{1}(:, rand_start:end_index);
    ds_out{i, 2} = ds_in{1}(1:9, rand_start:end_index);

    ds_out{i, 3} = onehot_label;

    % Increment start index to next sample window
    start_index = start_index + window_stride;
  end
end
