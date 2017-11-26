set terminal png
set output "pt.png"
set xtics nomirror
set ytics nomirror 
set yrange [:10]
set size 1,1
set xlabel "recievers coordinates"
set ylabel "p * 10^{6}"

set grid xtics ytics mxtics mytics 

set bmargin 4

set lmargin 8 
plot "ql.txt" using 1:($2*10**6) title "Re" lw 3 with lines, "ql.txt" using 1:($3*10**6) title "Im" lw 3 with lines

