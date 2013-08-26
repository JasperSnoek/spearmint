function [parameter_spec] = pb_read_Experiment__ParameterSpec(buffer, buffer_start, buffer_end)
%pb_read_Experiment__ParameterSpec Reads the protobuf message ParameterSpec.
%   function [parameter_spec] = pb_read_Experiment__ParameterSpec(buffer, buffer_start, buffer_end)
%
%   INPUTS:
%     buffer       : a buffer of uint8's to parse
%     buffer_start : optional starting index to consider of the buffer
%                    defaults to 1
%     buffer_end   : optional ending index to consider of the buffer
%                    defaults to length(buffer)
%
%   MEMBERS:
%     name           : required string, defaults to ''.
%     size           : required uint32, defaults to uint32(0).
%     type           : required enum, defaults to int32(1).
%     options        : repeated string, defaults to char([]).
%     min            : optional double, defaults to double(0).
%     max            : optional double, defaults to double(0).
%
%   See also pb_read_Experiment, pb_read_Job, pb_read_Parameter.
  
  if (nargin < 1)
    buffer = uint8([]);
  end
  if (nargin < 2)
    buffer_start = 1;
  end
  if (nargin < 3)
    buffer_end = length(buffer);
  end
  
  descriptor = pb_descriptor_Experiment__ParameterSpec();
  parameter_spec = pblib_generic_parse_from_string(buffer, descriptor, buffer_start, buffer_end);
  parameter_spec.descriptor_function = @pb_descriptor_Experiment__ParameterSpec;
