#!/bin/sh

cwd=`pwd`
	    
input_csv=${cwd}/mapped.csv
output_dir=${cwd}/outputs/nisqa/
	    
mkdir -p $output_dir
# nisqa
cd /opt/tools/NISQA
python3 run_predict.py --mode predict_csv --pretrained_model weights/nisqa.tar \
	--csv_file $input_csv --csv_deg target --num_workers 0 --bs 10 \
	--output_dir $output_dir
cd $cwd
	    
# dnsmos
testset_dir=${cwd}/
output_dir=${cwd}/outputs/dnsmos/
output_csv=${cwd}/outputs/dnsmos.csv
cd /opt/tools/DNS-Challenge/DNSMOS/
python3 dnsmos_local.py --testset_dir $testset_dir -o $output_csv -p
cd $cwd
	    
	    
# visqol
output_dir=${cwd}/outputs/visqol/
mkdir -p $output_dir
output_csv=${cwd}/outputs/visqol.csv
cd /opt/tools/visqol
python3 visqol.py --input-csv $input_csv  --output-csv $output_csv
cd $cwd
	    
# mosnet
output_dir=${cwd}/outputs/mosnet/
mkdir -p $output_dir
output_csv=${cwd}/outputs/mosnet.csv
python3 mosnet.py --input-csv $input_csv --output-csv $output_csv

