function [parameter] = pb_read_Parameter(buffer, buffer_start, buffer_end)
%pb_read_Parameter Reads the protobuf message Parameter.
%   function [parameter] = pb_read_Parameter(buffer, buffer_start, buffer_end)
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
%     int_val        : repeated int64, defaults to int64([]).
%     str_val        : repeated string, defaults to char([]).
%     dbl_val        : repeated double, defaults to double([]).
%
%   See also pb_read_Job, pb_read_Experiment.
  
  if (nargin < 1)
    buffer = uint8([]);
  end
  if (nargin < 2)
    buffer_start = 1;
  end
  if (nargin < 3)
    buffer_end = length(buffer);
  end
  
  descriptor = pb_descriptor_Parameter();
  parameter = pblib_generic_parse_from_string(buffer, descriptor, buffer_start, buffer_end);
  parameter.descriptor_function = @pb_descriptor_Parameter;
