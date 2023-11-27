function ds_data = validation_ds_reader(filename, cycle_count)
  arguments
    filename = [];
    cycle_count = 3;
  end

  persistent gp;
  if isempty(gp)
    gp = global_params();
  end
  
  filedata = matfile(filename, 'Writable', false);
  ds_data = cell(1, 3);

  sequence_len = gp.samples_per_cycle * cycle_count;

  % Get input data
  data = DTInfo.get_model_input(filedata.dt_info);
  data = data(:, :, 1:sequence_len);
  ds_data{1} = dlarray(reshape(data, [size(data, 1), size(data, 3)]), 'CT');

  % Get labels
  scenario_name = DTInfo.get_scenario_name(filedata.dt_info);
  scenario_label = DTInfo.get_scenario_label(scenario_name);

  event_timestep = DTInfo.get_event_timestep(filedata.dt_info);
  if (isempty(event_timestep) || (event_timestep < 1))
    event_timestep = 1;
  end

  ds_data{3} = dlarray(event_timestep, 'C');
  
  downsampled_event_time = floor(event_timestep / 16);
  labels = repmat(scenario_label, [1 floor(sequence_len/16)]);
  labels(:, 1:downsampled_event_time) = repmat(DTInfo.get_scenario_label("No events"), [1 downsampled_event_time]);
  ds_data{2} = dlarray(labels, 'CT');

end
