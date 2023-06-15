#!/bin/sh

#0/female/F014/tasnet/est_result/
#0/male/M024/tasnet/mix_result/
#0/male/M024/unet/mix_result

ref=../audio/b_set_reference.txt
typ=source_result
ref_trn=temp/ref.trn
hyp_trn=temp/hyp.trn

model_type=tasnet
speaker=F014
gender=female
snr=0
typ=est_result
decode_csv=../audio/csv/deaf/mix/output/$model_type/${speaker}_${gender}_${snr}_output.csv
python3 csv2trn.py --csv $decode_csv --type $typ --speaker $speaker --ref $ref --outdir temp/
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media sctk sclite -r $ref_trn trn -h $hyp_trn trn -i rm -o sum prf dtl >/dev/null

model_type=tasnet
speaker=M024
gender=male
snr=0
typ=mix_result
decode_csv=../audio/csv/deaf/mix/output/$model_type/${speaker}_${gender}_${snr}_output.csv
python3 csv2trn.py --csv $decode_csv --type $typ --speaker $speaker --ref $ref --outdir temp/
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media sctk sclite -r $ref_trn trn -h $hyp_trn trn -i rm -o sum prf dtl >/dev/null

model_type=unet
speaker=M024
gender=male
snr=0
typ=mix_result
decode_csv=../audio/csv/deaf/mix/output/$model_type/${speaker}_${gender}_${snr}_output.csv
python3 csv2trn.py --csv $decode_csv --type $typ --speaker $speaker --ref $ref --outdir temp/
docker run -it -v $PWD:/var/sctk -v /home:/home -v /media:/media sctk sclite -r $ref_trn trn -h $hyp_trn trn -i rm -o sum prf dtl > /dev/null
