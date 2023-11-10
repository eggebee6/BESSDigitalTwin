function ds_data = dtinfo_ds_reader(filename)
  arguments
    filename = [];
  end

  persistent gp;
  if isempty(gp)
    gp = global_params();
  end
  
  filedata = matfile(filename, 'Writable', false);
  ds_data = cell(1, 3);

  % Get feature training data
  feature_training_data = DTInfo.get_feature_training_input(filedata.dt_info)';
  ds_data{1} = feature_training_data(:, 1:gp.samples_per_cycle * 5);

  % Get scenario name
  ds_data{2} = DTInfo.get_scenario_name(filedata.dt_info);

  % Get scenario event time
  event_timestep = DTInfo.get_event_timestep(filedata.dt_info);
  if isempty(event_timestep)
    event_timestep = 0;
  end
  ds_data{3} = event_timestep;
  
end
