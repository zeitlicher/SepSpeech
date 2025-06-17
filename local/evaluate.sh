#!/bin/sh

rootdir=/media/fujie/hdd1/Speech/test
csvdir=audio/csv/deaf/

# conv-asnet
#config=tasnet_logs/version_2/hparams.yaml
#model_type=tasnet
#checkpoint=tasnet_logs/version_2/checkpoints/checkpoint_epoch\=394-step\=308890-valid_loss\=0.047.ckpt

# unet
#config=lightning_logs/version_3/hparams.yaml
#model_type=unet
#checkpoint=lightning_logs/version_3/checkpoints/checkpoint_epoch\=371-step\=290904-valid_loss\=0.042.ckpt

# unet2
#config=lightning_logs/version_5/hparams.yaml
#model_type=unet2
#checkpoint=lightning_logs/version_5/checkpoints/checkpoint_epoch=394-step=308890-valid_loss=0.043.ckpt

# unet + sdr_loss
config=lightning_logs/version_7/hparams.yaml
model_type=unet
output_tag=unet_sdr
checkpoint=lightning_logs/version_7/checkpoints/checkpoint_epoch=394-step=308890-valid_loss=-20.091.ckpt

for speaker in M023 M024 F013 F014 F018;
do
    for gender in  male female;
    do
	for snr in 0 5 10 20;
	do
	    input_csv=${csvdir}/mix/${speaker}_${gender}_${snr}.csv
	    enroll_csv=${csvdir}/${speaker}C.csv
	    output_csv_dir=${csvdir}/mix/output/${output_tag}
	    if [ ! -e $output_csv_dir ];then
		mkdir -p $output_csv_dir
	    fi
	    output_csv=${output_csv_dir}/${speaker}_${gender}_${snr}_output.csv
	    output_dir=${rootdir}/estimate/$output_tag/${snr}/${gender}/${speaker}
	    if [ ! -e $output_dir ];then
		mkdir -p $output_dir
	    fi
	    
	    python3 -m bin.evaluate --input_csv $input_csv \
		    --enroll_csv $enroll_csv \
		    --output_csv $output_csv \
		    --output_dir $output_dir \
		    --model_type $model_type \
		    --config $config \
		    --checkpoint $checkpoint
	done
    done
done
