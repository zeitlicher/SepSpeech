#!/bin/sh

jnasdir=/media/fujie/hdd1/Speech/jnas/
python3 find_waves.py --root $jnasdir --output-csv csv/waves.csv

noisedir=/media/fujie/hdd1/Speech/Chiba3Party/Mono
python3 find_waves.py --root $noisedir --output-csv csv/noise_3p.csv --noise

python3 filter_by_utterance.py --csv csv/waves.csv \
	--output-csv csv/waves_remove_b_c.csv \
	--remove --utterance B C
# use first 10 files, note that files are randomely ordered
head -11 csv/noise_3p.csv > csv/noise_3p_remove.csv

python3 mix.py --speech-csv csv/waves_remove_b_c.csv \
	--noise-csv csv/noise_3p_remove.csv \
	--output-csv csv/mix_3p.csv \
	--output-dir /media/fujie/hdd1/Speech/mix/ \
	--max-snr 20 --min-snr 0

python3 split_train_valid.py \
	--mixture-csv csv/mix_3p.csv \
	--source-csv csv/waves_remove_b_c.csv \
	--num-enroll 5 \
	--output-train csv/train_3p.csv \
	--output-valid csv/valid_3p.csv \
	--output-enroll csv/enroll_3p.csv


