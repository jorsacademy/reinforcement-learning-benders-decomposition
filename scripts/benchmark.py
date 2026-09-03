from rl_benders.cli import main

raise SystemExit(main(["benchmark", *(__import__("sys").argv[1:])]))
