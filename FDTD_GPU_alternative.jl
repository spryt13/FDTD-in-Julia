using CUDA
using BenchmarkTools
using Plots

max_len = 200; # length of the environment
max_time = 1000; # number of timesteps
number_of_threads = 16; number of GPU threads

imp0::Float64 = 377.0;
imp = 3imp0;
ez = CUDA.zeros(max_time, max_len); 
hy = CUDA.zeros(max_time, max_len-1);
# Please note that there is no separate array for saving data and both ez and hy entities are arrays instead of vectors.
# All the data calculated is saved and kept within GPU memory.

function kernel_hy(hy, ez, imp0, time) # calculate magnetic field
    index = threadIdx().x;
    stride = blockDim().x;
    for i = index:stride:size(hy)[2]
        @inbounds hy[time, i] = hy[time-1, i] + (ez[time-1, i+1] - ez[time-1, i]) / imp0
    end
    return nothing
end

function kernel_ez(hy, ez, imp0, imp, time) # calculate electric field
    index = threadIdx().x;
    stride = blockDim().x;
    for i = index:stride:size(ez)[2]
        if i == 1 # boundary point
            ez[time, i] = exp(-(time-30.) * (time-30.) / 100.) # source of electric field
        elseif i == size(ez)[2] # boundary point
            ez[time, i] = ez[time-1, i] * (1 - imp0/imp) - hy[time, i-1] * 2imp0
            ez[time, i] = ez[time, i] / (1 + imp0/imp)
        else
            @inbounds ez[time, i] = ez[time-1, i] + (hy[time, i] - hy[time, i-1]) * imp0
        end
    end
    return nothing
end

function calculate_rows(time) # calculate FDTD for a given time moment
    @cuda threads=number_of_threads kernel_hy(hy, ez, imp0, time)
    @cuda threads=number_of_threads kernel_ez(hy, ez, imp0, imp, time)
end

function FDTD()
    for time = 2:max_time
        calculate_rows(time)
    end
end
