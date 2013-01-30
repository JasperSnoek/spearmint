function val = pblib_helpers_iff(condition, if_true, if_else)
%pblib_helpers_iff Simulates an iff statement.
%   function val = pblib_helpers_iff(condition, if_true, if_else)
%
%   Helper functions that turns the if else construct into one expression to be
%   used in an anonymous function
%
%   If given an array of conditionals, then it selects from if_true or if_else
%   on a per element basis.
  
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

  val = if_true;
  for i=1:length(condition)
    if condition(i)
      val(i) = if_true(i);
    else
      val(i) = if_else(i);
    end
  end

