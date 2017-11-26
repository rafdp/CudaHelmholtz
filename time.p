set terminal png
set output "time.png"
set xrange [0:100]
set yrange [0:13000]
f(x) = x**4*log(x)*a
fit f(x) "asympttime.txt" using 1:2 via a
plot "asympttime.txt" using 1:2 with points, f(x) notitle
set output "size.png"
set xrange [0:100]
set yrange [0:60]
plot "asymptsize.txt" using 1:2 with points, "asymptsize.txt" using 1:2 smooth csplines
