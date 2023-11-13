function [event_time] = find_load_connect(dt_info, load_number)
  arguments
    dt_info = [];
    load_number = 2;
  end
  
  event_time = [];
  max_strlen = 255;
  
  var_load1_enable = false;
  var_load2_enable = false;
  var_load3_enable = false;
  var_load1_start = -1;
  var_load2_start = -1;
  var_load3_start = -1;
  
  for v = dt_info.variables
    if (strncmp(v.Name, 'Switchgear', max_strlen))
      Switchgear = v.Value;
      
      if (Switchgear.Load1_Enable ~= 0)
        var_load1_enable = true;
      end
      if (Switchgear.Load2_Enable ~= 0)
        var_load2_enable = true;
      end
      if (Switchgear.Load3_Enable ~= 0)
        var_load3_enable = true;
      end
      
      if (Switchgear.Load1_Connect_time ~= 0)
        var_load1_start = Switchgear.Load1_Connect_time;
      end
      if (Switchgear.Load2_Connect_time ~= 0)
        var_load2_start = Switchgear.Load2_Connect_time;
      end
      if (Switchgear.Load3_Connect_time ~= 0)
        var_load3_start = Switchgear.Load3_Connect_time;
      end
      
      break;
    end
  end
  
  if (var_load1_enable && (load_number == 1))
    event_time = var_load1_start;
  elseif (var_load2_enable && (load_number == 2))
    event_time = var_load2_start;
  elseif (var_load3_enable && (load_number == 3))
    event_time = var_load3_start;
  end
  
end
