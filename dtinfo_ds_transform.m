function [ds_out] = dtinfo_ds_transform(ds_in)
  persistent gp;
  if isempty(gp)
    gp = global_params();
  end
  
  % Scale data
  ds_in{1}(1:3, :) = ds_in{1}(1:3, :) ./ gp.iLf_err_scale;
  ds_in{1}(4:6, :) = ds_in{1}(4:6, :) ./ gp.vCf_err_scale;
  ds_in{1}(7:9, :) = ds_in{1}(7:9, :) ./ gp.iLo_err_scale;

  % Initialize output data
  data_len = size(ds_in{1}, 2);
  num_cells = gp.strides_per_sequence * floor(data_len / gp.min_sequence_len - 1);

  ds_out = cell(num_cells, 3);

  % Fill training samples with input data
  start_index = 1;
  window_stride = floor(gp.min_sequence_len / gp.strides_per_sequence);
  for i = 1:num_cells
    % Get a training sample starting from a random offset into the sample window
    rand_start = start_index + floor(0.5 * (rand() * gp.min_sequence_len) / gp.strides_per_sequence);
    end_index = rand_start + (gp.min_sequence_len - 1);

    % Copy measurements and errors into training sample
    ds_out{i, 1} = ds_in{1}(:, rand_start:end_index);
    ds_out{i, 2} = ds_in{1}(1:9, rand_start:end_index);

    ds_out{i, 3} = repmat(DTInfo.get_scenario_label(ds_in{2}), [1 gp.min_sequence_len]);
    ds_out{i, 3}(:, 1:ds_in{3}) = repmat(DTInfo.get_scenario_label("No events"), [1 ds_in{3}]);

    % Increment start index to next sample window
    start_index = start_index + window_stride;
  end
end
