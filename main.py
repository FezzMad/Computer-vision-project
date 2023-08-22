from engine.model import NNModel
import argparse

# for command line input
# parser = argparse.ArgumentParser()
# parser.add_argument('-cfg', type=str, required=False, help='Path to .yaml configuration file or name file from ./cfg')
# args = parser.parse_args()
# cfg_path = args.cfg
# model = NNModel(cfg_path)
# model.train()

model = NNModel('1.yaml')
model.train()
