function [experiment] = pb_read_Experiment(buffer, buffer_start, buffer_end)
%pb_read_Experiment Reads the protobuf message Experiment.
%   function [experiment] = pb_read_Experiment(buffer, buffer_start, buffer_end)
%
%   INPUTS:
%     buffer       : a buffer of uint8's to parse
%     buffer_start : optional starting index to consider of the buffer
%                    defaults to 1
%     buffer_end   : optional ending index to consider of the buffer
%                    defaults to length(buffer)
%
%   MEMBERS:
%     language       : required enum, defaults to int32(1).
%     name           : required string, defaults to ''.
%     variable       : repeated <a href="matlab:help pb_read_Experiment__ParameterSpec">Experiment.ParameterSpec</a>, defaults to struct([]).
%
%   See also pb_read_Experiment__ParameterSpec, pb_read_Job, pb_read_Parameter.
  
  if (nargin < 1)
    buffer = uint8([]);
  end
  if (nargin < 2)
    buffer_start = 1;
  end
  if (nargin < 3)
    buffer_end = length(buffer);
  end
  
  descriptor = pb_descriptor_Experiment();
  experiment = pblib_generic_parse_from_string(buffer, descriptor, buffer_start, buffer_end);
  experiment.descriptor_function = @pb_descriptor_Experiment;
