function [event_time] = find_pmloss(dt_info)
  event_time = [];
  max_strlen = 255;
  
  var_failure_time = -1;
  
  for v = dt_info.variables
    if (strncmp(v.Name, 'PGM', max_strlen))
      PGM = v.Value;
      event_time = PGM.Gen.Prime_mover_failure_time;
      
      if (event_time > 20) % TODO: Not this silly time check
        event_time = [];
      end
      
      break;
    end
  end
end
