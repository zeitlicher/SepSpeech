#!/bin/sh

rootdir=/media/fujie/hdd1/Speech/test
csvdir=/home/fujie/SepSpeech/audio/csv/deaf/

ref=/home/fujie/SepSpeech/audio/b_set_reference.txt

for speaker in M023 M024 F013 F014 F018;
do
    decode_csv_dir=${csvdir}/mix/output/source
    decode_csv=${decode_csv_dir}/${speaker}B_output.csv
    if [ ! -e $decode_csv ];then
	echo $decode_csv
    fi
    sctk_dir=${rootdir}/sctk/source/$speaker/

    if [ ! -e $sctk_dir ];then
	mkdir -p $sctk_dir
    fi

    typ=source_result
    ref_trn=$sctk_dir/ref.trn
    hyp_trn=$sctk_dir/hyp.trn
    python3 csv2trn.py --csv $decode_csv --type $typ --speaker $speaker --ref $ref --outdir $sctk_dir
    docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media sctk \
	   sclite -r $ref_trn trn -h $hyp_trn trn -i rm -o sum prf dtl > /dev/null

    # 4 x 2 x 4 x 2 x 2 = 32 = 128
    for gender in  male female;
    do
	for snr in 0 5 10 20;
	do
	    #for model_type in tasnet unet;
	    for model_type in unet_sdr;
	    do
		#for typ in mix_result est_result;
		for typ in mix_result est_result;
		do
		    decode_csv_dir=${csvdir}/mix/output/${model_type}
		    decode_csv=${decode_csv_dir}/${speaker}_${gender}_${snr}_output.csv
		    if [ ! -e $decode_csv ];then
		       echo $decode_csv
		    fi
		    sctk_dir=${rootdir}/sctk/${snr}/${gender}/${speaker}/${model_type}/${typ}
		    if [ ! -e $sctk_dir ];then
			mkdir -p $sctk_dir
		    fi

		    ref_trn=$sctk_dir/ref.trn
		    hyp_trn=$sctk_dir/hyp.trn
		    python3 csv2trn.py --csv $decode_csv --type $typ --speaker $speaker --ref $ref --outdir $sctk_dir
		    docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media sctk \
			   sclite -r $ref_trn trn -h $hyp_trn trn -i rm -o sum prf dtl > /dev/null
		done
	    done
	done
    done
done
