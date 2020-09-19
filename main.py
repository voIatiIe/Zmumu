import numpy as np
import uproot

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import *


def calc_mass(vector):
	'''Вычисление инвариантной массы'''
	E = vector[3]
	psq = np.sum((vector[0:3])**2)

	return(((E**2 - psq)**0.5)*(1e-3))


def plot(dataset, bins=50):
	'''Построение гистограммы'''
	fig, ax = plt.subplots()
	ax.hist(dataset, bins, edgecolor="black", alpha=0.5)

	mean = np.mean(dataset)
	stdev = (np.std(dataset))**0.5
	entries = len(dataset)

	ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

	ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(2000))

	info = "Mean: {0:.2f} GeV\n".format(mean) + \
		   "Stdev: {0:.2f}\n".format(stdev) + \
		   "Width: {0:.2f}\n".format(calc_width(dataset)) + \
		   "Entries: {}".format(entries)

	ax.legend(title=info)
	ax.set_title("Invariant mass")
	ax.set_xlabel("Mass, GeV")
	plt.show()	


def calc_width(dataset, bins=50):
	'''Вычисление ширины распределения на полувысоте'''
	dataset.sort()
	batch_size = int(len(dataset)/bins)
	container = [[]]

	mx, mn = np.max(dataset), np.min(dataset)
	interval_len = (mx - mn)/bins
	cur_max = mn + interval_len

	for element in dataset:
		if element < cur_max:
			container[-1].append(element)
		else:
			cur_max += interval_len
			container.append([])
			container[-1].append(element)

	counts = [len(element) for element in container]
	peak = np.max(counts)
	peak_ind = counts.index(peak)
	borders = []

	for step in (1, -1):
		ind = peak_ind
		while counts[ind] > peak/2:
			ind += step
		borders.append(ind)

	return np.mean(container[borders[0]]) - np.mean(container[borders[1]])


def file_read():
	'''Читает ветви лептонов, складывает 4-векторы, подсчитывает инвариантную массу'''
	file = uproot.open(FILENAME)

	dirc = file[DIRNAME]
	tree = dirc[HOMETREENAME]
	branch0 = tree["lep_0_p4"].array()
	branch1 = tree["lep_1_p4"].array()

	container = []

	for vector0, vector1 in zip(branch0, branch1):
		temp0 = np.array([vector0.x, vector0.y, vector0.z, vector0.E])
		temp1 = np.array([vector1.x, vector1.y, vector1.z, vector1.E])
		sum_vect = temp0 + temp1
		container.append(calc_mass(sum_vect))

	return container


if __name__ == "__main__":
	dataset = file_read()
	plot(dataset)

