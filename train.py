from argparse import ArgumentParser
import yaml

def main(config:dict):
    model = LitSepSpeaker(config)
    data = SpeechDataModule(config)
    trainer = Trainer(
        max_epochs=config['train']['max_epochs'],
        gpus=1,
        gradient_clip_val=config['train']['gradient_clip_val'],
        precision=config['train']['precision']
    )
    trainer.fit(model, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    main(config)
