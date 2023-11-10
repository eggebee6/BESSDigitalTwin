function [event_time] = find_ess_connect(dt_info)
  event_time = [];
  max_strlen = 255;

  var_ess_enable = false;
  var_start_time = -1;
  
  for v = dt_info.variables
    if (strncmp(v.Name, 'Switchgear', max_strlen))
      Switchgear = v.Value;
      if (Switchgear.BESS_Enable ~= 0)
        var_ess_enable = true;
      end
    end
    
    if (strncmp(v.Name, 'ESS', max_strlen))
      ESS = v.Value;
      var_start_time = ESS.VSC.StartTime + 0.1;
    end
  end
  
  if (var_ess_enable)
    event_time = var_start_time;
  end

end
