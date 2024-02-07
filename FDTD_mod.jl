using Plots

size = 200;
max_time = 1000;

imp0::Float64 = 377.0;
imp = imp0;
time::Int64 = 0;
ez = zeros(size)
hy = zeros(size-1)
series_ez = zeros(max_time, size);
series_hy = zeros(max_time, size);

function count_hy(count, imp0)
    hy[count] = hy[count] + (ez[count+1] - ez[count]) / imp0
end

function count_ez(count, imp0)
    ez[count] = ez[count] + (hy[count] - hy[count-1]) * imp0
end

function border_ez(size, imp0, imp)
    ez[size] = ez[size] * (1 - imp0/imp) - hy[size-1] * 2imp0
    ez[size] = ez[size] / (1 + imp0/imp)
end

function FDTD()
    for time = 1:max_time # ogarnąć CUDĘ, zrobić repo #ewentualnie zrozumieć dlaczego działa
        for count = 1:size-1
            count_hy(count, imp0)
        end
        for count = 2:size-1
            count_ez(count, imp0)
        end
        ez[1] = exp(-(time-30.) * (time-30.) / 100.)# + exp(-(time-100.) * (time-100.) / 100.)
        border_ez(size, imp0, imp)
        for count = 1:size
            series_ez[time, count] = ez[count]
        end
        for count = 1:size-1
            series_hy[time, count] = hy[count]
        end
    end
end