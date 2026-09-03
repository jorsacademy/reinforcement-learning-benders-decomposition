from rl_benders.cli import main

raise SystemExit(main(["research", *(__import__("sys").argv[1:])]))
