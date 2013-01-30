function [num, num_read] = pblib_read_varint64(buffer, offset)
%pblib_read_varint64
%   function [num, num_read] = pblib_read_varint64(buffer, offset)
  
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

  current_byte = uint64(buffer(offset));
  num = bitset(current_byte, 8, 0);
  index = 1;
  while (current_byte > 127)
    current_byte = uint64(buffer(offset + index));
    num = bitor(bitshift(bitset(current_byte, 8, 0), 7*index), num);
    index = index + 1;
  end
  num_read = index;

