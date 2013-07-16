function [wire_value, num_read] = pblib_read_wire_type(buffer, offset, wire_type)
%pblib_read_wire_type
%   function [wire_value, num_read] = pblib_read_wire_type(buffer, offset, wire_type)
%
%   These values must match the WireType enum in
%   http://protobuf.googlecode.com/svn/trunk/src/google/protobuf/wire_format.h
%
%   All Wire Types Are (in order, 0 based):
%    - 0: 'varint'
%    - 1: '64bit'
%    - 2: 'length_delimited'
%    - 3: 'start_group'
%    - 4: 'end_group'
%    - 5: '32bit'
  
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

  switch wire_type
    case 0
      [wire_value, num_read] =  pblib_read_varint64(buffer, offset);
    case 1
      num_read = 8;
      wire_value = buffer(offset : offset + num_read - 1);
    case 2
      [len, len_len] = pblib_read_varint32(buffer, offset);
      num_read = len + len_len;
      if len == 0
        wire_value = {uint8([]), 1, 0};
      else
        wire_value = {buffer, offset + len_len, offset + num_read - 1};
      end
    case 3     
      error('proto:lib:read_wire_type', 'Start Group not implemented.');
    case 4      
      error('proto:lib:read_wire_type', 'End Group not implemented.');
    case 5
      num_read = 4;
      wire_value = buffer(offset : offset + num_read - 1);
    otherwise
      error('proto:lib:read_wire_type', 'Invalid wire value. This is likely due to a malformed message.');
  end




