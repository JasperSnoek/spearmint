function [job] = pb_read_Job(buffer, buffer_start, buffer_end)
%pb_read_Job Reads the protobuf message Job.
%   function [job] = pb_read_Job(buffer, buffer_start, buffer_end)
%
%   INPUTS:
%     buffer       : a buffer of uint8's to parse
%     buffer_start : optional starting index to consider of the buffer
%                    defaults to 1
%     buffer_end   : optional ending index to consider of the buffer
%                    defaults to length(buffer)
%
%   MEMBERS:
%     id             : required uint64, defaults to uint64(0).
%     expt_dir       : required string, defaults to ''.
%     name           : required string, defaults to ''.
%     language       : required enum, defaults to int32(1).
%     status         : optional string, defaults to ''.
%     param          : repeated <a href="matlab:help pb_read_Parameter">Parameter</a>, defaults to struct([]).
%     submit_t       : optional uint64, defaults to uint64(0).
%     start_t        : optional uint64, defaults to uint64(0).
%     end_t          : optional uint64, defaults to uint64(0).
%     value          : optional double, defaults to double(0).
%     duration       : optional double, defaults to double(0).
%
%   See also pb_read_Parameter, pb_read_Experiment.
  
  if (nargin < 1)
    buffer = uint8([]);
  end
  if (nargin < 2)
    buffer_start = 1;
  end
  if (nargin < 3)
    buffer_end = length(buffer);
  end
  
  descriptor = pb_descriptor_Job();
  job = pblib_generic_parse_from_string(buffer, descriptor, buffer_start, buffer_end);
  job.descriptor_function = @pb_descriptor_Job;
