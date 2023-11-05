function [ds_out] = dtinfo_ds_transform(ds_in)
  persistent gp;
  if isempty(gp)
    gp = global_params();
  end
  
  % Scale data
  ds_in(1:3) = ds_in(1:3) ./ gp.iLf_err_scale;
  ds_in(4:6) = ds_in(4:6) ./ gp.vCf_err_scale;
  ds_in(7:9) = ds_in(7:9) ./ gp.iLo_err_scale;

  % Initialize training data output
  data_len = size(ds_in, 2);
  start_index = 1;
  num_cells = gp.strides_per_sequence * floor((data_len - start_index) / gp.min_sequence_len) - 1;
  ds_out = cell(num_cells, 1);

  % Fill training samples with input data
  for i = 1:num_cells
    % Get a training sample starting from a random offset into the sample window
    rand_start = start_index + floor((rand() / gp.strides_per_sequence) * gp.min_sequence_len);
    end_index = rand_start + (gp.min_sequence_len - 1);

    % Copy measurements and errors into training sample
    ds_out{i} = ds_in(20:28, rand_start:end_index);

    % Increment start index to next sample window
    start_index = start_index + (gp.min_sequence_len / gp.strides_per_sequence);
  end
end
