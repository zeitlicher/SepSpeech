#!/bin/sh

#M041 M042 M043
#M040 M006 M005
#F122 F123 F124
#F109 

ouput_dir=csv/normal
for speaker in M041 M042 M043 F122 F123 F124;
do
    python3 filter_by_speaker.py --csv csv/waves.csv --output-csv temp.csv --speakers M042
done
