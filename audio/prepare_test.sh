#!/bin/sh

rootdir=/media/fujie/hdd1/Speech/test
csvdir=csv/deaf/

mkdir ${csvdir}/mix
mkdir ${rootdir}/premix
mkdir ${csvdir}/premix

for speaker in M023 M024 F013 F014 F018;
do
    for gender in  male female;
    do
	part_noise_dir=${rootdir}/premix/${gender}/${speaker}/
	mkdir -p $part_noise_dir
	python3 extract_noise.py --speech-csv ${csvdir}/${speaker}B.csv \
		--noise-csv ${csvdir}/noise_${gender}_3p.csv \
		--output-csv ${csvdir}/premix/${speaker}_${gender}.csv \
		--output-dir $part_noise_dir
	for snr in 0 5 10 20;
	do
	    outdir=${rootdir}/${snr}/${gender}/${speaker}
	    mkdir -p $outdir
	    python3 mix_pre_noise.py --speech-csv ${csvdir}/premix/${speaker}_${gender}.csv \
		    --output-csv ${csvdir}/mix/${speaker}_${gender}_${snr}.csv \
		    --max-snr ${snr} --min-snr ${snr} \
		    --output-dir $outdir
	done
    done
done
