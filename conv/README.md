scratch, to be finished benchmark of intrinsic implementation.

<<<<<<<<< Performance Use all the samples!!!

       5828.238318      task-clock (msec)         #    0.999 CPUs utilized          
               489      context-switches          #    0.084 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
                60      page-faults               #    0.010 K/sec                  
    16,370,684,734      cycles                    #    2.809 GHz                      (49.99%)
        21,312,254      stalled-cycles-frontend   #    0.13% frontend cycles idle     (49.99%)
     3,180,021,068      stalled-cycles-backend    #   19.43% backend cycles idle      (50.09%)
    38,803,586,780      instructions              #    2.37  insn per cycle         
                                                  #    0.08  stalled cycles per insn  (50.08%)
     5,315,244,513      branches                  #  911.981 M/sec                    (50.04%)
           742,741      branch-misses             #    0.01% of all branches          (50.01%)

       5.831264041 seconds time elapsed

<<<<<<<<< Performance Use all the samples, switch the loops!

       1895.264868      task-clock (msec)         #    0.999 CPUs utilized          
               159      context-switches          #    0.084 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
                58      page-faults               #    0.031 K/sec                  
     5,320,025,493      cycles                    #    2.807 GHz                      (50.00%)
        22,671,053      stalled-cycles-frontend   #    0.43% frontend cycles idle     (50.00%)
     2,166,976,466      stalled-cycles-backend    #   40.73% backend cycles idle      (50.00%)
     7,702,088,469      instructions              #    1.45  insn per cycle         
                                                  #    0.28  stalled cycles per insn  (50.24%)
       317,685,505      branches                  #  167.621 M/sec                    (50.14%)
         2,958,856      branch-misses             #    0.93% of all branches          (50.03%)

       1.896844359 seconds time elapsed

<<<<<<<<< Performance basic, but decimate half samples.

       3370.863484      task-clock (msec)         #    0.997 CPUs utilized          
               313      context-switches          #    0.093 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
                59      page-faults               #    0.018 K/sec                  
     9,435,384,979      cycles                    #    2.799 GHz                      (50.03%)
         7,029,153      stalled-cycles-frontend   #    0.07% frontend cycles idle     (50.13%)
     1,053,444,456      stalled-cycles-backend    #   11.16% backend cycles idle      (50.19%)
    20,245,095,906      instructions              #    2.15  insn per cycle         
                                                  #    0.05  stalled cycles per insn  (50.11%)
     2,662,000,673      branches                  #  789.709 M/sec                    (49.95%)
           457,638      branch-misses             #    0.02% of all branches          (49.83%)

       3.380434001 seconds time elapsed

<<<<<<<<<< Performance switch the loop but output half sample!

       3749.668196      task-clock (msec)         #    0.993 CPUs utilized          
               319      context-switches          #    0.085 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
                60      page-faults               #    0.016 K/sec                  
    10,527,815,589      cycles                    #    2.808 GHz                      (50.07%)
        28,506,135      stalled-cycles-frontend   #    0.27% frontend cycles idle     (50.08%)
     1,050,381,601      stalled-cycles-backend    #    9.98% backend cycles idle      (50.10%)
    28,276,585,648      instructions              #    2.69  insn per cycle         
                                                  #    0.04  stalled cycles per insn  (49.96%)
       316,619,687      branches                  #   84.439 M/sec                    (49.93%)
         2,841,972      branch-misses             #    0.90% of all branches          (50.02%)

       3.776574550 seconds time elapsed

###########current best SSE Performance counter stats for './basic':

       1066.662922      task-clock (msec)         #    0.971 CPUs utilized          
               105      context-switches          #    0.098 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
                61      page-faults               #    0.057 K/sec                  
     2,987,636,028      cycles                    #    2.801 GHz                      (49.44%)
        11,404,536      stalled-cycles-frontend   #    0.38% frontend cycles idle     (50.00%)
       452,626,497      stalled-cycles-backend    #   15.15% backend cycles idle      (50.30%)
     6,546,246,316      instructions              #    2.19  insn per cycle         
                                                  #    0.07  stalled cycles per insn  (50.72%)
       421,527,308      branches                  #  395.183 M/sec                    (50.53%)
           703,488      branch-misses             #    0.17% of all branches          (50.04%)

       1.098956123 seconds time elapsed
