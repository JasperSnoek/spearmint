function result = braninhoo(job_id, params)
  
  x = bsxfun(@plus, bsxfun(@times, params.X, [15 15]), [0 -5]);
  
  y = (x(:,2) - (5.1./(4*pi.^2)).*x(:,1).^2 + (5/pi).*x(:,1) - 6).^2 +...
      10.*(1-(1./(8*pi))).*cos(x(:,1)) + 10;
  
  result = y;

  % Really hard problem!
  pause(5);
  
end
