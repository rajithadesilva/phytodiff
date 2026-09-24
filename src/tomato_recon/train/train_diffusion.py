"""Compatibility entry point for DiffTS stage 2 training."""

from tomato_recon.train.train_diffts import main

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
