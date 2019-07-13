import lib.model_serializer as serializer
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--snapshot_path')
parser.add_argument('-o', '--output', default='results/model.pt')
parser.add_argument('-c', '--config', default='configs/m2det320_resnet101.py')
args = parser.parse_args()


def main():
    from train import get_model
    from utils.core import Config, config_compile

    cfg = Config.fromfile(args.config)
    config_compile(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.snapshot_path, map_location=device)

    model = get_model(device, cfg)

    model.load_state_dict(checkpoint['model'])
    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    main()
