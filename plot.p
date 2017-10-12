set terminal png
set output "Real.png"
plot "ql.txt" using 1:2 lw 3 with lines, "output.txt" using 1:2 lw 3 with lines, "BornCpu.txt" using 1:2 lw 2 with lines
set output "Imag.png"
plot "output.txt" using 1:3 lw 3 with lines, "ql.txt" using 1:3 lw 2 with lines
