using CUDA
using BenchmarkTools
using Plots

size = 200;
number_of_threads = 16;
max_time = 1000;

imp0::Float64 = 377.0;
imp = 3*imp0;
ez = CUDA.zeros(size);
hy = CUDA.zeros(size-1);
series_ez = zeros(max_time, size);
series_hy = zeros(max_time, size);

function kernel_hy(hy, ez, imp0) # obliczanie natężenia pola magnetycznego
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(hy)
        @inbounds hy[i] += (ez[i+1] - ez[i]) / imp0
    end
    return nothing
end

function kernel_ez(hy, ez, imp0) # obliczanie natężenia pola elektrycznego
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(ez)-1
        @inbounds ez[i+1] += (hy[i+1] - hy[i]) * imp0
    end
    return nothing
end

function border_ez(size, imp0, imp) # warunek brzegowy
    ez[size] = ez[size] * (1 - imp0/imp) - hy[size-1] * 2imp0
    ez[size] = ez[size] / (1 + imp0/imp)
end

function calculate_rows(time) # obliczenie wszystkiego dla danej chwili
    @cuda threads=number_of_threads kernel_hy(hy, ez, imp0)
    @cuda threads=number_of_threads kernel_ez(hy, ez, imp0)
    ez[1] = exp(-(time-30.) * (time-30.) / 100.) # źródło pola elektrycznego
    border_ez(size, imp0, imp)
end

function save_FDTD(time) # zapis wartości pól
    for count = 1:size
        series_ez[time, count] = ez[count]
    end
    for count = 1:size-1
        series_hy[time, count] = hy[count]
    end
end

function FDTD() # wykomentowanie save_FDTD() wyłącza zapis
    for time = 1:max_time
        calculate_rows(time)
	save_FDTD()
    end
end

