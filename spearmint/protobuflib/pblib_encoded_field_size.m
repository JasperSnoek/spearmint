function [len] = pblib_encoded_field_size(field_value, field_descriptor)
%pblib_encoded_field_size
%   [len] = pblib_encoded_field_size(field_value, field_descriptor)
%
%   Returns the amount of space an encoded field with this value will take up,
%   NOT INCLUDING the tag. If a field is a repeated field, you should pass the
%   whole field to this function, not each individual element.
%   
%   This function assumes this field is not empty and probably will not work
%   correctly if it is as those cases haven't been tested.
  
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


  LABEL_REPEATED = 3;
  switch(field_descriptor.wire_type)
    case 0 % 'varint'
      len = 0;
      for j=1:length(field_value)
        len = len + pblib_encoded_varint_size(...
            field_descriptor.write_function(field_value(j)));
      end
    case 1 % '64bit'
      len = length(field_value) * 8;
    case 2 % 'length_delimited'
      switch (field_descriptor.matlab_type)
        case {7, 8} % 'string' or 'bytes'
          if (field_descriptor.label == LABEL_REPEATED)
            len = 0;
            for j=1:length(field_value)
              temp_len = length(field_value{j});
              len = len + ...
                  pblib_encoded_varint_size(temp_len) + ...
                  temp_len;
            end
          else
            temp_len = length(field_value);
            len = pblib_encoded_varint_size(temp_len) + ...
                temp_len;
          end
        case 9 % 'message'
          len = 0;
          for j=1:length(field_value)
            temp_len = pblib_get_serialized_size(field_value(j));
            len = len + ...
                pblib_encoded_varint_size(temp_len) + ...
                temp_len;
          end
        otherwise
          error('proto:pblib_get_serialized_size', ...
                ['Unhandled case statement: wire_type ' ...
                 num2str(field_descriptor.wire_type)]);
      end
    case 3 % 'start_group'
      error('proto:pblib_get_serialized_size', 'start_group unsupported in matlab');
    case 4 % 'end_group'
      error('proto:pblib_get_serialized_size', 'end_group unsupported in matlab');
    case 5 % '32bit'
      len = length(field_value) * 4;
  end


