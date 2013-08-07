function len = pblib_type_to_estimated_encoded_length(type)
% pblib_type_to_estimated_encoded_length
%   function len = pblib_type_to_estimated_encoded_length(type)
%
%   Converts from proto type to estimated encoded length.
%
%   This function is currently only used in the local function read_packed_field in
%   pblib_generic_parse_from_string.m.  It is used to estimate how much space we need to
%   store the incoming packed values to avoid many reallocations. We can probably improve
%   this to switch on wire type and read all the values in, without storing them, to
%   calculate how many values are encoded in a buffer, but this is good enough until it
%   isn't.
  
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

  type_to_estimated_encoded_length = [...
      8, ... % double => always 8 bytes
      4, ... % float => always 4 bytes
      7, ... % int64 => probably large or else int32 would be used (varint can be as large as 10 bytes)
      7, ... % uint64 => same as int64
      2, ... % int32 => values < 16384 take up 2 bytes
      8, ... % fixed64 => always 8 bytes
      4, ... % fixed32 => always 4 bytes
      1, ... % bool => probably either 0 or 1, which is 1 varint encoded byte
      -1, ... % string => this function shouldn't be used to estimate string length
      -1, ... % group => groups are unsupported in matlab
      -1, ... % message => this function shouldn't be used to estimate message length
      -1, ... % bytes => this function shouldn't be used to estimate byte length
      2, ... % uint32 => values < 16384 take up 2 bytes
      1, ... % enum => values expected to be < 128, which is 1 varint encoded byte
      4, ... % sfixed32 => always 4 bytes
      8, ... % sfixed64 => always 8 bytes
      2, ... % sint32 => signed values abs value < 8192 take up 2 varint encoded bytes
      7];    % sint64 => see int64
  len = type_to_estimated_encoded_length(type);
