function ds_data = dtinfo_ds_reader(filename)
  arguments
    filename = [];
  end
  
  filedata = matfile(filename, 'Writable', false);

  ds_data = cell(1, 4);
  ds_data{1} = DTInfo.get_all_err(filedata.dt_info)';
  ds_data{2} = DTInfo.get_all_meas(filedata.dt_info)';
  ds_data{3} = DTInfo.get_vgrid(filedata.dt_info)';
  ds_data{4} = DTInfo.get_scenario_name(filedata.dt_info);
  
end
