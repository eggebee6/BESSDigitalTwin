function [action] = scenario_action(scenario_name)
  % Split scenario name
  str = string(split(scenario_name, ' '));
  
  % Get action based on scenario
  if ( (length(str) >= 4) && (str(1) == "Fault") && (str(3) == "Location") )
    % "Fault x Location y"

    %action = sprintf('Fault %d Location %d', str2num(str(2)), str2num(str(4)));
    action = sprintf('Fault location %d', str2num(str(4)));

    %location = str2double(str(4));
    %if (location == 2)
    %  action = sprintf('Gen fault');
    %elseif (location == 3)
    %  action = sprintf('Disconnect');
    %else
    %  action = sprintf('Ride through');
    %end

  elseif ( (length(str) >= 3) && (join(str(1:3), ' ') == "Prime mover loss") )
    % "Prime mover loss"

    action = sprintf('Prime mover loss');

    %action = sprintf('Grid forming');

  elseif ( (length(str) >= 3) && (join(str(1:2), ' ') == "Load step") )
    % "Load step x"

    %action = sprintf('Load step %d', str2num(str(3)));

    action = sprintf('No action');

  elseif ( (length(str) >= 2) && (join(str(1:2), ' ') == "No events") )
    % "No events"

    %action = sprintf('No events');
    
    action = sprintf('No action');

  else
    % No action perscribed for scenario
    error('Unknown scenario name %s', scenario_name);

  end

  action = {action};

end
