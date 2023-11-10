function [event_time] = find_gen_connect(dt_info)
  event_time = [];
  max_strlen = 255;
  
  var_gen_enabled = false;
  var_start_time = -1;
  
  for v = dt_info.variables
    if (strncmp(v.Name, 'Switchgear', max_strlen))
      Switchgear = v.Value;
      if (Switchgear.Source_Enable ~= 0)
        var_gen_enabled = true;
      end
    end
    
    if (strncmp(v.Name, 'PGM', max_strlen))
      PGM = v.Value;
      if (PGM.Gen.Enable ~= 0)
        var_start_time = PGM.Gen.StartTime + 2;
      end
    end
  end
  
  if (var_gen_enabled)
    event_time = var_start_time;
  end

end
