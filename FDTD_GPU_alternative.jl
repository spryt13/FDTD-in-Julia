using CUDA
using BenchmarkTools
using Plots

size = 200;
max_time = 200;
number_of_threads = 256;

imp0::Float64 = 377.0;
imp = 3*imp0;
ez = CUDA.zeros(max_time, size);
hy = CUDA.zeros(max_time, size-1);

function kernel_hy(hy, ez, imp0, time)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(hy)
        @inbounds hy[time, i] = hy[time-1, i] + (ez[time-1, i+1] - ez[time-1, i]) / imp0
    end
    return nothing
end

function kernel_ez(hy, ez, imp0, time)
    index = threadIdx().x
    stride = blockDim().x
    #@cuprintln("thread $index")
    for i = index:stride:length(ez)-2
        @inbounds ez[time, i+1] = ez[time-1, i+1] + (hy[time, i+1] - hy[time, i]) * imp0
    end
    return nothing
end

function border_ez(size, imp0, imp, time)
    ez[time, size] = ez[time-1, size] * (1 - imp0/imp) - hy[time, size-1] * 2imp0
    ez[time, size] = ez[time-1, size] / (1 + imp0/imp)
    return nothing
end

function calculate_rows(time)
    @cuda threads=number_of_threads kernel_hy(hy, ez, imp0, time)
    @cuda threads=number_of_threads kernel_ez(hy, ez, imp0, time)
    ez[time, 1] = exp(-(time-30.) * (time-30.) / 100.)
    #ez[1] += exp(-(time - 30.) * (time - 30.) / 100.);
    border_ez(size, imp0, imp, time)
end

function FDTD()
    for time = 2:max_time
        calculate_rows(time)
    end
end