set terminal png
set output "qReal.png"
plot "output.txt" using 1:2 lw 3 with lines, "ql.txt" using 1:2 lw 2 with lines
set output "qImag.png"
plot "output.txt" using 1:3 lw 3 with lines, "ql.txt" using 1:3 lw 2 with lines
