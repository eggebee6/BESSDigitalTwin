function [action] = scenario_action(scenario_name)
  % Split scenario name
  str = string(split(scenario_name, ' '));
  
  % Get action based on scenario
  if ( (length(str) >= 4) && (str(1) == "Fault") && (str(3) == "Location") )
    % "Fault x Location y"

    % Fault locations:
    % 2 - Generator
    % 3 - Bus
    % 4 - Load 3, LVAC load
    % 5 - Load 2, Induction machine
    % 6 - Load 1, LVAC load

    location = str2double(str(4));
    if (location == 2)
      action = sprintf('Gen fault');    % Switch to grid forming
    elseif (location == 3)
      action = sprintf('Bus fault');    % Disconnect
    else
      action = sprintf('Load fault');   % Ride through
    end

  elseif ( (length(str) >= 3) && (join(str(1:3), ' ') == "Prime mover loss") )
    % "Prime mover loss"

    action = sprintf('PM loss');    % Switch to grid forming

  elseif ( (length(str) >= 3) && (join(str(1:2), ' ') == "Load step") )
    % "Load step x"

    % Loads:
    % Load 1 - LVAC load 100 KW 0.95 PF
    % Load 2 - Induction machine 25 KW 0.80 PF
    % Load 3 - LVAC load 25 KW 0.90 PF

    location = str2double(str(3));
    if (location == 2)
      action = sprintf('IM loadstep');      % Ride through
    else
      %action = sprintf('LVAC loadstep');    % Ride through
      action = sprintf('No action');
    end

  elseif ( (length(str) >= 2) && (join(str(1:2), ' ') == "No events") )
    % "No events"

    action = sprintf('No action');

  else
    % No action perscribed for scenario
    error('Unknown scenario name %s', scenario_name);

  end

  action = {action};

end
