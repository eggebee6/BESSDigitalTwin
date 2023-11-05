function ds_data = dtinfo_ds_reader(filename)
  arguments
    filename = [];
  end
  
  filedata = matfile(filename, 'Writable', false);
  dt_info = filedata.dt_info;
  ds_data = dt_info.data';
  
end
