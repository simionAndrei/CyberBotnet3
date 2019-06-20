#!/usr/bin/env bash
BUILDER=""
if [[ "$OSTYPE" == "linux-gnu" ]]; then
        BUILDER=apt-get
elif [[ "$OSTYPE" == "darwin"* ]]; then
        BUILDER=apt-get
fi


${BUILDER} install git-lfs
git lfs install
git lfs pull

