from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tomato-recon")
    parser.add_argument("command", choices=["infer", "train-encoder", "train-diffusion", "train-graph", "train-parametric", "train-joint"])
    args, remainder = parser.parse_known_args(argv)
    if args.command == "infer":
        from tomato_recon.infer import main as command
    else:
        module_name = args.command.replace("-", "_")
        module = __import__(f"tomato_recon.train.{module_name}", fromlist=["main"])
        command = module.main
    command(remainder)


if __name__ == "__main__":
    main()

