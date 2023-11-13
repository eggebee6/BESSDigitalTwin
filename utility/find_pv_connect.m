function [event_time] = find_pv_connect(dt_info)
  event_time = [];
  max_strlen = 255;
  
  var_pv_enable = false;
  var_start_time = -1;
  
  for v = dt_info.variables
    if (strncmp(v.Name, 'Switchgear', max_strlen))
      Switchgear = v.Value;
      if (Switchgear.PV_Enable ~= 0)
        var_pv_enable = true;
      end
    end
    
    if (strncmp(v.Name, 'PV', max_strlen))
      PV = v.Value;
      var_start_time = PV.VSC.StartTime + 0.1;
    end
  end
  
  if (var_pv_enable)
    event_time = var_start_time;
  end
  
end
