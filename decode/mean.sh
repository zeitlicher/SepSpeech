#!/bin/sh

dir=/media/fujie/hdd1/Speech/test/sctk/ 

cer=`cat ${dir}/source/F*/*prf | perl mean.pl`
echo F,clean,,,,$cer

for snr in 20 10 5 0;
do
    for gender in male female;
    do
	for model in tasnet;
	do
	    cer=`cat ${dir}/${snr}/${gender}/F*/${model}/mix_result/*prf | perl mean.pl`
	    echo F,$snr,$gender,,mix,$cer
	    
	done
	for model in tasnet unet unet_sdr unet2;
	do
	    for target in est;
	    do
		cer=`cat ${dir}/${snr}/${gender}/F*/${model}/${target}_result/*prf | perl mean.pl`
		echo F,$snr,$gender,$model,$target,$cer
	    done
	done
    done
done
