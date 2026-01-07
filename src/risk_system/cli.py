import argparse

from risk_system.train import train
from risk_system.evaluate import evaluate
from risk_system.config import load_yaml
from risk_system.monitor import monitor



def main():
    parser = argparse.ArgumentParser("Risk Assessment Model - Train the model or Evaluate it")
    subparser = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    train_parser = subparser.add_parser("train", help="Train the model")
    train_parser.add_argument("--base", help="Base config", default="configs/base.yaml")
    train_parser.add_argument("--model", help="Model config", default="configs/model.yaml")
    train_parser.add_argument("--monitor", help="Monitor config", default="configs/monitor.yaml")
    train_parser.add_argument("--artifact-dir", help="Dirctory to store training artifacts", default="artifacts")

    eval_parser = subparser.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--base", help="Base config", default="configs/base.yaml")
    eval_parser.add_argument("--policy", help="Evaluation policy", default="configs/policy.yaml")
    eval_parser.add_argument("--artifact-dir", help="Dirctory to store training artifacts", default="artifacts")

    monitor_parser = subparser.add_parser("monitor", help="Run drift monitoring")
    monitor_parser.add_argument("--base", default="configs/base.yaml")
    monitor_parser.add_argument("--monitor", default="configs/monitor.yaml")
    monitor_parser.add_argument("--artifact-dir", default="artifacts")


    args = parser.parse_args()

    if args.command == "train":
        cfg_base = load_yaml(args.base)
        cfg_model = load_yaml(args.model)
        cfg_monitor = load_yaml(args.monitor)
        train(cfg_base, 
              cfg_model, 
              cfg_monitor, 
              artifacts_dir=args.artifact_dir)

    elif args.command == "evaluate":
        cfg_base = load_yaml(args.base)
        policy = load_yaml(args.policy)
        evaluate(cfg_base, 
                 artifacts_dir=args.artifact_dir, 
                 policy=policy)

    elif args.command == "monitor":
        cfg_base = load_yaml(args.base)
        monitor_cfg = load_yaml(args.monitor)
        monitor(
            cfg_base=cfg_base,
            monitor_cfg=monitor_cfg,
            artifacts_dir=args.artifact_dir,
        )


if __name__ == "__main__":
    main()