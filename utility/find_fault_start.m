function [event_time, fault_type, fault_location] = find_fault_start(dt_info)
  event_time = [];
  fault_type = [];
  fault_location = [];
  max_strlen = 255;
  
  var_fault_enabled = false;
  var_fault_location = -1;
  var_fault_type = -1;
  var_fault_time = -1;
  
  for v = dt_info.variables
    if (strncmp(v.Name, 'Fault_enable', max_strlen))
      var_fault_enabled = true;
    end
    
    if (strncmp(v.Name, 'Fault_Type', max_strlen))
      var_fault_type = v.Value;
    end
    
    if (strncmp(v.Name, 'Fault_Location', max_strlen))
      var_fault_location = v.Value;
    end
    
    if (strncmp(v.Name, 'Fault_Apply', max_strlen))
      var_fault_time = v.Value;
    end
  end
  
  if (var_fault_enabled)
    event_time = var_fault_time;
    fault_type = var_fault_type;
    fault_location = var_fault_location;
  end
  
end
