.PHONY: train tensorboard clean train-seeds

train:
	python -m src.main_train --run-name v1

# usage: make train-seed-77
train-seed-%:
	python -m src.main_train --seed $* --run-name seed-$*

# 3-seed sweep
train-seeds:
	@for s in 13 42 77; do \
		python -m src.main_train --seed $$s --run-name seed-$$s ; \
	done

tensorboard:
	tensorboard --logdir=logs/tensorboard

clean:
	rm -rf logs/tensorboard/* models/checkpoints/* models/final/*
