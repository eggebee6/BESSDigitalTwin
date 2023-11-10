function [index] = time_to_index(time, start_time, sample_frequency)
  arguments
    time = 15;
    start_time = 15;
    sample_frequency = 100000;
  end
  
  index = floor((time - start_time) * sample_frequency);
end
