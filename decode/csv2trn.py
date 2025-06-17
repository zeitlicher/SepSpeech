import os, sys,re
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--csv', type=str)
parser.add_argument('--type', type=str)
parser.add_argument('--speaker', type=str)
parser.add_argument('--ref', type=str)
parser.add_argument('--outdir', type=str)
args = parser.parse_args()

ref_path = os.path.join(args.outdir, 'ref.trn')
hyp_path = os.path.join(args.outdir, 'hyp.trn')

refs={}
pattern=re.compile(r'^(\S+)\s+')
with open(args.ref, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line=line.strip()
        tag=re.match(pattern, line).groups()[0]
        line=re.sub(pattern,'', line)
        refs[args.speaker+tag] = line
with open(ref_path, 'w') as w:
    keys = sorted(refs.keys())
    for key in keys:
        trn='{0} ({1})\n'.format(refs[key], key)
        w.write(trn)
            
hyps={}
pattern=re.compile(r'\s|、|。|！|？|\?')
df = pd.read_csv(args.csv)
for index, row in df.iterrows():
    key = row['key']
    line = row[str(args.type)]
    try:
        line = re.sub(pattern, '', line)
    except:
        line = ""
    line = ' '.join(list(line))
    hyps[key] = line
with open(hyp_path, 'w') as w:
    keys = sorted(hyps.keys())
    for key in keys:
        trn = '{0} ({1})\n'.format(hyps[key], key)
        w.write(trn)
