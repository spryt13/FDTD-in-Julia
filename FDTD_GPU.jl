using CUDA
using BenchmarkTools
using Plots

max_len = 200; # length of the environment
max_time = 1000; # number of timesteps
number_of_threads = 16; # number of GPU threads
save_step = 50; # number of time iterations between saving results

imp0::Float64 = 377.0;
imp = 3imp0;
ez = CUDA.zeros(max_len);
hy = CUDA.zeros(max_len-1);
series_ez = zeros(div(max_time, save_step), max_len);
series_hy = zeros(div(max_time, save_step), max_len-1);

function kernel_hy(hy, ez, imp0) # calculate magnetic field
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(hy)
        @inbounds hy[i] = hy[i] + (ez[i+1] - ez[i]) / imp0
    end
    return nothing
end

function kernel_ez(hy, ez, imp0, imp, time) # calculate electric field
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(ez)
        if i == 1 # boundary point
            ez[i] = exp(-(time-30.) * (time-30.) / 100.) # source of electric field
        elseif i == length(ez) # boundary point
            ez[i] = ez[i] * (1 - imp0/imp) - hy[i-1] * 2imp0
            ez[i] = ez[i] / (1 + imp0/imp)
        else
            @inbounds ez[i] = ez[i] + (hy[i] - hy[i-1]) * imp0
        end
    end
    return nothing
end

function calculate_rows(time) # calculate FDTD for a given time moment
    @cuda threads=number_of_threads kernel_hy(hy, ez, imp0)
    @cuda threads=number_of_threads kernel_ez(hy, ez, imp0, imp, time)
end

function save_FDTD(time)
    series_ez[time, :] = Array(ez)
    series_hy[time, :] = Array(hy)
end

function FDTD()
    count = 1;
    for time = 1:max_time
        calculate_rows(time)
        i = divrem(count, save_step) # this is used to verify if current timestep should be saved and helps find precise space in the array
        if i[2] == 0
            save_FDTD(i[1])
        end
        count += 1;
    end
end

