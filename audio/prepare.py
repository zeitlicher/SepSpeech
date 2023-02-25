import os, glob, sys
import argparse

def main(args):

    spk2id={}
    id=1
    for path in glob(os.path.join(args.root, '**/*.wav')):
        if args.relative:
            os.relative_to(path)
        else:
            os.relative(path)
        file=os.path.basename(path)
        if '_DT' in file: # JNAS Desktop recordings
            file.replace('_DT.wav', '')
            if file.startwith('N'): # JNAS newspaper
                spk = re.search(r'N(\S+)\d\d\d', file).group()
            else:   # JNAS balance sentences
                spk = re.search(r'B(\S+)\S\d\d', file).group()
        elif file.startwith('M') or file startwith('F'): # Deaf male/female balance sentences
            spk = 'D'+re.search(r'(\S\S\S)\S\S\S\.wav', file).group()
        if spk2id[spk] is None:
            spk2id[spk] = id
            id+=1
            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--speech-csv', type=str, required=True)
    parser.add_argument('--noise-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    args.parser.parse_args()

    main(args)
