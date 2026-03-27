#! /bin/bash

docker run --rm -v $(pwd)/data:/data -v $(pwd)/checkpoints:/checkpoints session6:1.0 $@
