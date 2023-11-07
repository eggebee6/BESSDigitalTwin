function [ds_out] = dtinfo_ds_transform(ds_in)
  persistent gp;
  if isempty(gp)
    gp = global_params();
  end
  
  ds_err = ds_in{1};
  ds_meas = ds_in{2};
  ds_vgrid = ds_in{3} ./ gp.voltage_pu;   % TODO: This should be done in the DT reader
  ds_label = DTInfo.get_scenario_label(ds_in{4});

  % Scale data
  ds_err(1:3, :) = ds_err(1:3, :) ./ gp.iLf_err_scale;
  ds_err(4:6, :) = ds_err(4:6, :) ./ gp.vCf_err_scale;
  ds_err(7:9, :) = ds_err(7:9, :) ./ gp.iLo_err_scale;

  % Initialize training data output
  data_len = size(ds_err, 2);
  start_index = 1;
  num_cells = 1 + gp.strides_per_sequence * floor((data_len - start_index) / gp.min_sequence_len - 1);
  ds_out = cell(num_cells, 4);

  % Fill training samples with input data
  for i = 1:num_cells
    % Get a training sample starting from a random offset into the sample window
    rand_start = start_index + floor((rand() * gp.min_sequence_len) / gp.strides_per_sequence);
    end_index = rand_start + (gp.min_sequence_len - 1);

    % Copy measurements and errors into training sample
    ds_out{i, 1} = ds_err(:, rand_start:end_index);
    ds_out{i, 2} = ds_meas(:, rand_start:end_index);
    ds_out{i, 3} = ds_vgrid(:, rand_start:end_index);
    ds_out{i, 4} = ds_label;

    % Increment start index to next sample window
    start_index = start_index + floor(gp.min_sequence_len / gp.strides_per_sequence);
  end
end
