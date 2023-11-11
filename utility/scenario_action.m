function [action] = scenario_action(scenario_name)
  % Split scenario name
  str = string(split(scenario_name, ' '));
  
  % Get action based on scenario
  if ( (length(str) >= 4) && (str(1) == "Fault") && (str(3) == "Location") )
    % "Fault x Location y"
    location = str2double(str(4));
    if (location == 2)
      action = sprintf('Grid forming');
    elseif (location == 3)
      action = sprintf('Disconnect');
    else
      action = sprintf('No action');
    end

  elseif ( (length(str) >= 3) && (join(str(1:3), ' ') == "Prime mover loss") )
    % "Prime mover loss"
    action = sprintf('Grid forming');

  elseif ( (length(str) >= 3) && (join(str(1:2), ' ') == "Load step") )
    % "Load step x"
    action = sprintf('No action');

  elseif ( (length(str) >= 2) && (join(str(1:2), ' ') == "No events") )
    % "No events"
    action = sprintf('No action');

  else
    % No action perscribed for scenario
    error('Unknown scenario name %s', scenario_name);

  end

  action = {action};

end
