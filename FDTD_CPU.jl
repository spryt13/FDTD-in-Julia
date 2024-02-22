using BenchmarkTools
using Plots

# size (length) of the object and number of timesteps
size = 200;
max_time = 1000;

imp0::Float64 = 377.0;
imp = 3*imp0;
# vectors for calculations
ez = zeros(size);
hy = zeros(size-1);
# arrays for saving results in
series_ez = zeros(max_time, size);
series_hy = zeros(max_time, size);

function count_hy(count, imp0) # calculation of electric field
    hy[count] = hy[count] + (ez[count+1] - ez[count]) / imp0
end

function count_ez(count, imp0) # calculation of magnetic field
    ez[count] = ez[count] + (hy[count] - hy[count-1]) * imp0
end

function border_ez(size, imp0, imp) # boundary point
    ez[size] = ez[size] * (1 - imp0/imp) - hy[size-1] * 2imp0
    ez[size] = ez[size] / (1 + imp0/imp)
end

function calculate_rows(time) # iteration in a given time
    for count = 1:size-1
        count_hy(count, imp0)
    end
    for count = 2:size-1
        count_ez(count, imp0)
    end
    ez[1] = exp(-(time-30.) * (time-30.) / 100.) # electric field source (boundary point)
    border_ez(size, imp0, imp)
end

function save_FDTD(time) # save FDTD results
    for count = 1:size
        series_ez[time, count] = ez[count]
    end
    for count = 1:size-1
        series_hy[time, count] = hy[count]
    end
end

function FDTD()
    for time = 1:max_time          
        calculate_rows(time)
        save_FDTD(time)
    end
end
