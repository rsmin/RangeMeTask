#!/usr/bin/env bash
./scripts/setup_env.sh
python3 Stage1.py > Data/Stage1Output.log
python3 Stage2.py > Data/Stage2Output.log
python3 Stage3.py > Data/Stage3Output.log