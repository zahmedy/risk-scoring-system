import argparse

from risk_system.train import train
from risk_system.evaluate import evaluate
from risk_system.config import load_yaml



def main():
    parser = argparse.ArgumentParser("Risk Assessment Model - Train the model or Evaluate it")

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--evaluate")
    
    args = parser.parse_args()

    cfg_base = load_yaml("configs/base.yaml")
    cfg_model = load_yaml("configs/model.yaml")
    policy = load_yaml("configs/policy.yaml")

    if args.train:
        train(cfg_base, cfg_model)

    if args.evaluate:
        evaluate(cfg_base,threshold=policy['threshold'])

if __name__ == "__main__":
    main()