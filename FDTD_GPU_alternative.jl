using CUDA
using BenchmarkTools
using Plots

max_len = 200;
max_time = 1000;
number_of_threads = 16;

imp0::Float64 = 377.0;
imp = 3imp0;
ez = CUDA.zeros(max_time, max_len);
hy = CUDA.zeros(max_time, max_len-1);

function kernel_hy(hy, ez, imp0, time)
    index = threadIdx().x;
    stride = blockDim().x;
    for i = index:stride:size(hy)[2]
        @inbounds hy[time, i] = hy[time-1, i] + (ez[time-1, i+1] - ez[time-1, i]) / imp0
    end
    return nothing
end

function kernel_ez(hy, ez, imp0, imp, time)
    index = threadIdx().x;
    stride = blockDim().x;
    
    for i = index:stride:size(ez)[2]
        if i == 1
            ez[time, i] = exp(-(time-30.) * (time-30.) / 100.)
        elseif i == size(ez)[2]
            ez[time, i] = ez[time-1, i] * (1 - imp0/imp) - hy[time, i-1] * 2imp0
            ez[time, i] = ez[time, i] / (1 + imp0/imp)
        else
            @inbounds ez[time, i] = ez[time-1, i] + (hy[time, i] - hy[time, i-1]) * imp0
        end
    end
    return nothing
end

function calculate_rows(time)
    @cuda threads=number_of_threads kernel_hy(hy, ez, imp0, time)
    @cuda threads=number_of_threads kernel_ez(hy, ez, imp0, imp, time)
end

function FDTD()
    for time = 2:max_time
        calculate_rows(time)
    end
end
