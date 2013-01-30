function [buffer] = pblib_write_varint(value)
%pblib_write_varint
%   buffer = pblib_write_varint(value)
%
%   Encodes a uint value as a varint. 
%
%   The value passed in can be any uint up to uint64.  We do a hand coded
%   binary search tree like Google's C++ code generation to quickly decide how
%   many bytes we'll need for the encoding, or at least estimate.
  
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

  
  if (value < uint64(268435456)) % 2^28
    if (value < uint64(16384)) % 2^14
      if (value < uint64(128)) % 2^7
        buffer = zeros([1 1], 'uint8');
      else
        buffer = zeros([1 2], 'uint8');
      end
    else
      if (value < uint64(2097152)) % 2^21
        buffer = zeros([1 3], 'uint8');
      else
        buffer = zeros([1 4], 'uint8');
      end
    end
  else
    if (value < uint64(4398046511104)) % 2^42
      if (value < uint64(34359738368)) % 2^35
        buffer = zeros([1 5], 'uint8');
      else
        buffer = zeros([1 6], 'uint8');
      end
    else
      if (value < uint64(72057594037927936)) % 2^56
        buffer = zeros([1 8], 'uint8');
      else
        buffer = zeros([1 10], 'uint8');
      end
    end
  end
  num_bytes = 0;
  while (value > 127)
    num_bytes = num_bytes + 1;
    buffer(num_bytes) = bitset(bitand(value, 127), 8);
    value = bitshift(value, -7);
  end
  num_bytes = num_bytes + 1;
  buffer(num_bytes) = value;
  buffer = buffer(1 : num_bytes);
