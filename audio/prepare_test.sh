#!/bin/sh

rootdir=/media/fujie/hdd1/Speech/test
csvdir=csv/deaf/

mkdir ${csvdir}/mix
for speaker in M023 M024 F013 F014 F018;
do
    for gender in  male female;
    do
	for snr in 0 10 20;
	do
	    outdir=${rootdir}/${snr}/${gender}/${speaker}
	    mkdir -p $outdir
	    python3 mix.py --speech-csv ${csvdir}/${speaker}B.csv \
		    --noise-csv ${csvdir}/noise_${gender}_3p.csv \
		    --output-csv ${csvdir}/mix/${speaker}_${gender}_${snr}.csv \
		    --max-snr ${snr} --min-snr ${snr} \
		    --output-dir $outdir
	done
    done
done
