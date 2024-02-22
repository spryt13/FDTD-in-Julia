using BenchmarkTools
using Plots

# rozmiar obliczanej przestrzeni
size = 200;
max_time = 1000;

imp0::Float64 = 377.0;
imp = 3*imp0;
# wektory do wykonywania obliczeń
ez = zeros(size);
hy = zeros(size-1);
# macierze do zapisywania wyników
series_ez = zeros(max_time, size);
series_hy = zeros(max_time, size);

function count_hy(count, imp0) # oblicz natężenie pola magnetycznego
    hy[count] = hy[count] + (ez[count+1] - ez[count]) / imp0
end

function count_ez(count, imp0) # oblicz natężenie pola magnetycznego
    ez[count] = ez[count] + (hy[count] - hy[count-1]) * imp0
end

function border_ez(size, imp0, imp) # punkt brzegowy
    ez[size] = ez[size] * (1 - imp0/imp) - hy[size-1] * 2imp0
    ez[size] = ez[size] / (1 + imp0/imp)
end

function calculate_rows(time) # iteracja wszystkiego w danej chwili czasowej
    for count = 1:size-1
        count_hy(count, imp0)
    end
    for count = 2:size-1
        count_ez(count, imp0)
    end
    ez[1] = exp(-(time-30.) * (time-30.) / 100.) # źródło pola elektrycznego
    border_ez(size, imp0, imp)
end

function save_FDTD(time) # zapis wyników FDTD
    for count = 1:size
        series_ez[time, count] = ez[count]
    end
    for count = 1:size-1
        series_hy[time, count] = hy[count]
    end
end

function FDTD()                    # wywołanie FDTD. Wykomentowanie save_FDTD() spowoduje brak zapisu
    for time = 1:max_time          # i zapamiętanie jedynie ostatniej chwili czasowej w wektorach ez i hy.
        calculate_rows(time)
        save_FDTD(time)
    end
end
