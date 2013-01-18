function matlab_wrapper(job_file)
  addpath(genpath('protobuflib'))
  
  fprintf('Matlab loading job from file %s\n', job_file)

  job = load_job(job_file);
  
  params = struct('job_id', job.id);
  
  for ii=1:length(job.param)
    if ~isempty(job.param(ii).int_val)
      params = setfield(params, job.param(ii).name, ...
                                job.param(ii).int_val);
    elseif ~isempty(job.param(ii).dbl_val)
      params = setfield(params, job.param(ii).name, ...
                                job.param(ii).dbl_val);      
    elseif ~isempty(job.param(ii).str_val)
      params = setfield(params, job.param(ii).name, ...
                                job.param(ii).str_val{1});      
    else
      fprintf('Unknown field type for %s\n', job.param(ii).name);
    end
  end
  
  params
  
  curdir = cd;
  
  cd(job.expt_dir);

  result = feval(job.name, job.id, params);
  fprintf('Function valuated with result %f\n', result);
  
  cd(curdir);
  
  job = pblib_set(job, 'value', result);
  
  save_job(job_file, job);
  fprintf('Job file updated.\n');
  
  function job = load_job(job_file)
    fid = fopen(job_file);
    buffer = fread(fid, [1 inf], '*uint8');
    fclose(fid);
    job = pb_read_Job(buffer);
  end
  
  function save_job(job_file, job)  
    buffer = pblib_generic_serialize_to_string(job);
    fid = fopen(job_file, 'w');
    fwrite(fid, buffer, 'uint8');
    fclose(fid);
  end

end
