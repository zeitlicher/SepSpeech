import os, glob, sys
import argparse
import pandas as pd
import wave

def main(args):

    if args.noise:
        df = pd.DataFrame(index=None, columns=['noise', 'length'])
    else:
        df = pd.DataFrame(index=None, columns=['source', 'length', 'speaker', 'index'])
    spk2id={}
    id=1

    for path in glob(os.path.join(args.root, '**/*.wav')):
        if args.relative:
            path=os.path.relpath(path)
        else:
            path=os.relative(path)
        wav = wave.open(path, 'r')
        wavlen = wav.readframes(wav.getnframes())
        file=os.path.basename(path)
        if args.noise:
            df.append(data=[[path, wavlen]])
        else:
            if '_DT' in file: # JNAS Desktop recordings
                file = file.replace('_DT.wav', '')
                if file.startwith('N'): # JNAS newspaper
                    spk = re.search(r'N(\S+)\d\d\d', file).group()
                else:   # JNAS balance sentences
                    spk = re.search(r'B(\S+)\S\d\d', file).group()
            elif file.startwith('M') or file startwith('F'): # Deaf male/female balance sentences
                spk = 'D'+re.search(r'(\S\S\S)\S\S\S\.wav', file).group()
            if not spk2id.get(spk):
                spk2id[spk] = id
                id+=1
            df.append(data=[[path, wavlen, spk, spk2id[spk]]])

    df.to_csv(args.output_csv)
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='./', required=True)
    parser.add_argument('--relative', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--output-csv', type=str, required=True)

    args=parser.parse_args()

    main(args)
