function ds_data = dtinfo_ds_reader(filename)
  arguments
    filename = [];
  end
  
  filedata = matfile(filename, 'Writable', false);
  dt_info = filedata.dt_info;

  ds_data = cell(1, 2);
  ds_data{1} = dt_info.data';
  ds_data{2} = DTInfo.get_scenario_name(dt_info);
  
end
