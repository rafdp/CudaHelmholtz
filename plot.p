set terminal png
set output "Real.png"
plot "output.txt" using 1:2 lw 3 with lines, "ql.txt" using 1:2 lw 1 with lines, "output_.txt" using 1:2 lw 3 with lines
set output "Imag.png"
plot "output.txt" using 1:3 lw 3 with lines, "ql.txt" using 1:3 lw 1 with lines, "BornCpu.txt" using 1:3 lw 3 with lines
