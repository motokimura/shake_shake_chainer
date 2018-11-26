#!/usr/bin/env python
# -*- coding: utf-8 -*-

## TODO
## * CIFAR10 data augmentation
## * Learning rate scheduling with cosine annealing

from __future__ import print_function
import argparse
import os

import chainer
import chainer.links as L
from chainer import training
from chainer.training import triggers, extensions
from chainer.datasets import get_cifar10
#from chainer.datasets import get_cifar100 # not supported for now..

from tensorboardX import SummaryWriter
from tboard_logger import TensorboardLogger

from shake_shake import ShakeShake


def main():
	parser = argparse.ArgumentParser(description='Shake-shake resularization CIFAR10 w/ Chainer')
	parser.add_argument('--dataset', '-d', default='cifar10',
						help='The dataset to use: cifar10 or cifar100')
	parser.add_argument('--batchsize', '-b', type=int, default=128,
						help='Number of images in each mini-batch')
	parser.add_argument('--lr', '-l', type=float, default=0.1,
						help='Learning rate for SGD')
	parser.add_argument('--epoch', '-e', type=int, default=300,
						help='Number of sweeps over the dataset to train')
	parser.add_argument('--gpu', '-g', type=int, default=0,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='run_0',
						help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	log_dir = os.path.join("results", args.out)
	writer = SummaryWriter(log_dir=log_dir)

	# Set up a neural network to train.
	# Classifier reports softmax cross entropy loss and accuracy at every
	# iteration, which will be used by the PrintReport extension below.
	if args.dataset == 'cifar10':
		print('Using CIFAR10 dataset.')
		class_labels = 10
		train, test = get_cifar10()
	elif args.dataset == 'cifar100':
		raise RuntimeError('Sorry, shake-shake for CIFAR100 is not yet ready..')
		#print('Using CIFAR100 dataset.')
		#class_labels = 100
		#train, test = get_cifar100()
	else:
		raise RuntimeError('Invalid dataset choice.')
	model = L.Classifier(ShakeShake(class_labels))
	if args.gpu >= 0:
		# Make a specified GPU current
		chainer.backends.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()  # Copy the model to the GPU

	optimizer = chainer.optimizers.MomentumSGD(args.lr, momentum=0.9)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
												 repeat=False, shuffle=False)

	# Set up a trainer
	updater = training.updaters.StandardUpdater(
		train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=log_dir)

	# Evaluate the model with the test dataset for each epoch
	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

	# Reduce the learning rate by half every 25 epochs.
	#trainer.extend(extensions.ExponentialShift('lr', 0.5),
	#			   trigger=(25, 'epoch'))

	# Dump a computational graph from 'loss' variable at the first iteration
	# The "main" refers to the target link of the "main" optimizer.
	trainer.extend(extensions.dump_graph('main/loss'))

	# Take a snapshot at each epoch
	trainer.extend(extensions.snapshot(
		filename='snaphot_epoch_{.updater.epoch}'))

	# Write a log of evaluation statistics for each epoch
	trainer.extend(extensions.LogReport())

	# Monitor learning rate at every iteration
	trainer.extend(extensions.observe_lr(), trigger=(1, 'iteration'))

	# Save two plot images to the result dir
	if extensions.PlotReport.available():
		trainer.extend(
			extensions.PlotReport(
				['main/loss', 'validation/main/loss'],
				'epoch', file_name='loss.png'))

		trainer.extend(
			extensions.PlotReport(
				['main/accuracy', 'validation/main/accuracy'],
				'epoch', file_name='accuracy.png'))
		
		trainer.extend(
			extensions.PlotReport(
				['lr'],
				'epoch', file_name='lr.png'))
	
	# Print selected entries of the log to stdout
	# Here "main" refers to the target link of the "main" optimizer again, and
	# "validation" refers to the default name of the Evaluator extension.
	# Entries other than 'epoch' are reported by the Classifier link, called by
	# either the updater or the evaluator.
	trainer.extend(extensions.PrintReport(
		['epoch', 'main/loss', 'validation/main/loss',
		 'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time']))

	# Print a progress bar to stdout
	trainer.extend(extensions.ProgressBar())

	# Write training log to TensorBoard log file
	trainer.extend(TensorboardLogger(writer,
		['main/loss', 'validation/main/loss',
		'main/accuracy', 'validation/main/accuracy',
		'lr']))
	
	if args.resume:
		# Resume from a snapshot
		chainer.serializers.load_npz(args.resume, trainer)

	# Run the training
	trainer.run()


if __name__ == '__main__':
	main()