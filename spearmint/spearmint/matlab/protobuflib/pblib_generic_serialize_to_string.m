function [buffer] = pblib_generic_serialize_to_string(msg)
%pblib_generic_serialize_to_string
%   function [buffer] = pblib_generic_serialize_to_string(msg)
  
%   protobuf-matlab - FarSounder's Protocol Buffer support for Matlab
%   Copyright (c) 2008, FarSounder Inc.  All rights reserved.
%   http://code.google.com/p/protobuf-matlab/
%  
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions are met:
%  
%       * Redistributions of source code must retain the above copyright
%   notice, this list of conditions and the following disclaimer.
%  
%       * Redistributions in binary form must reproduce the above copyright
%   notice, this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
%  
%       * Neither the name of the FarSounder Inc. nor the names of its
%   contributors may be used to endorse or promote products derived from this
%   software without specific prior written permission.
%  
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
%   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%   POSSIBILITY OF SUCH DAMAGE.

%   Author: fedor.labounko@gmail.com (Fedor Labounko)
%   Support function used by Protobuf compiler generated .m files.

  % enum values we use
  WIRE_TYPE_LENGTH_DELIMITED = 2;
  LABEL_REPEATED = 3;

  descriptor = msg.descriptor_function();
  buffer = zeros([1 pblib_get_serialized_size(msg)], 'uint8');
  num_written = 0;
  for i=1:length(descriptor.fields)
    field = descriptor.fields(i);
    if (get(msg.has_field, field.name) == 0)
      continue;
    end
    if (field.label == LABEL_REPEATED)
      if (field.options.packed)
        % two is the length delimited wire_type
        tag = pblib_write_tag(field.number, WIRE_TYPE_LENGTH_DELIMITED);
        buffer(num_written + 1 : num_written + length(tag)) = tag;
        num_written = num_written + length(tag);

        wire_values = write_packed_field(msg.(field.name), field);
        wire_value = pblib_write_wire_type(wire_values, WIRE_TYPE_LENGTH_DELIMITED);
        buffer(num_written + 1 : num_written + length(wire_value)) = wire_value;
        num_written = num_written + length(wire_value);
      else
        tag = pblib_write_tag(field.number, field.wire_type);
        for j=1:length(msg.(field.name))
          buffer(num_written + 1 : num_written + length(tag)) = tag;
          num_written = num_written + length(tag);
          if (field.matlab_type == 7 || field.matlab_type == 8) % 'string' or 'bytes'
            value = msg.(field.name){j};
          else
            value = msg.(field.name)(j);
          end
          wire_values = pblib_write_wire_type(field.write_function(value), field.wire_type);
          buffer(num_written + 1 : num_written + length(wire_values)) = wire_values;
          num_written = num_written + length(wire_values);
        end
      end
    else
      tag = pblib_write_tag(field.number, field.wire_type);
      buffer(num_written + 1 : num_written + length(tag)) = tag;
      num_written = num_written + length(tag);

      value = msg.(field.name);
      wire_value = pblib_write_wire_type(field.write_function(value), field.wire_type);
      buffer(num_written + 1 : num_written + length(wire_value)) = wire_value;
      num_written = num_written + length(wire_value);
    end
  end
  % now write the unknown fields
  for i=1:length(msg.unknown_fields)
    buffer(num_read + 1 : num_read + length(msg.unknown_fields(i).raw_data)) = ...
        msg.unknown_fields(i).raw_data;
    num_read = num_read + length(msg.unknown_fields(i).raw_data);
  end
  if (num_written ~= length(buffer))
    error('proto:pblib_generic_serialize_to_string', ...
          ['num_written, ' num2str(num_written) ...
           ', is different from precalculated length ' ...
           num2str(length(buffer))]);
  end


function [wire_values] = write_packed_field(values, field)
  wire_values = zeros([1 pblib_encoded_field_size(values, field)], 'uint8');
  values = field.write_function(values);
  bytes_written = 0;
  for i=1:length(values)
    encoded_value = pblib_write_wire_type(values(i), field.wire_type);
    wire_values(bytes_written + 1 : bytes_written + length(encoded_value)) = encoded_value;
    bytes_written = bytes_written + length(encoded_value);
  end
  wire_values = wire_values(1 : bytes_written);
  
