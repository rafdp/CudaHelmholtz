set terminal png
set output "Diff.png"
set grid
set xlabel "Receivers coordinates"
set ylabel "Deviation"
#set yrange [-0.0000004:0.0000004]
f(x) = 0
plot f(x) notitle with lines lw 3 lt rgb "green",\
     '<paste -d " " ql.txt BornGpu.txt'  using 1:($2 - $5) title "Born"   with lines lw 3 lt 3           ,\
     '<paste -d " " ql.txt qa.txt'  using 1:($2 - $5) title "QA"    with lines lw 3 lt rgb "red"
     
