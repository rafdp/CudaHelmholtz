set terminal png
set output "Real.png"
set grid
set xlabel "Receivers coordinates"
set ylabel "pressure"
plot "ql.txt"      using 1:2 title "QL"   with lines lw 2 lt rgb "green"           ,\
     "output.txt"  using 1:2 title "0"    with lines lw 2 lt rgb "red"   , \
     "BornGpu.txt" using 1:2 title "Born" with lines lw 2 lt 3           , \
     "qa.txt"      using 1:2 title "QA"   with lines lw 2 lt rgb "red" 
set output "Imag.png"
plot "output.txt"  using 1:3 title "0"    with lines lw 2 lt rgb "red"   ,\
     "ql.txt"      using 1:3 title "QL"   with lines lw 2 lt rgb "green"           ,\
     "qa.txt"      using 1:3 title "QA"   with lines lw 2 lt rgb "red" ,\
     "BornGpu.txt" using 1:3 title "Born" with lines lw 2 lt 3           
