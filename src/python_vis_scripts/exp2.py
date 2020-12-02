import numpy as np
import matplotlib.pyplot as plt
import os

directory = '../graph_data/exp2/'
out_directory = '../graph_data/exp2/out/'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

num_dimensions = 3
distr_data = {}
data_info = {}

def plot_data(data, info):
    for i in range(num_dimensions):
        for key in data.keys():
            x = data[key][i, :, 0]
            y = data[key][i, :, 1]
            plt.plot(x, y, label=key)
        plt.xscale("log")
        plt.title("%sD data, %s points" % (i + 2, info['num_samples']))
        plt.xlabel('max points in leaf')
        plt.ylabel('t [s]')
        # plt.xlim(0, x[-1])
        plt.legend()
        plot_name = "%sD_%spoints.pdf" % (i + 2, info['num_samples'])
        plt.savefig(os.path.join(out_directory, plot_name))
        plt.clf()


def get_data_info(line):
    d = line.strip('\n').split(" ")
    info = {}
    info['dim'] = int(d[0])
    info['num_mlp'] = int(d[1])
    info['num_samples'] = int(d[2])
    return info

def process_line(line):
    d = line.strip('\n').split(" ")
    mlp = int(d[0])
    time = float(d[1])
    return np.array([mlp, time])

def process_data(fname, data_info):
    with open(directory + fname, mode='r') as f:
        dist = fname.split('_')[0]
        data_info = get_data_info(f.readline())

        for i in range(data_info['num_mlp']):
            data = process_line(f.readline())

            if dist in distr_data.keys():
                distr_data[dist][data_info['dim'] - 2, i, :] = data
            else:
                distr_data[dist] = np.zeros((num_dimensions, data_info['num_mlp'], 2))
                distr_data[dist][data_info['dim'] - 2, i, :] = data
    return data_info

info = None
for filename in os.listdir(directory):
    if filename.endswith("txt"):
        info = process_data(filename, data_info)

plot_data(distr_data, info)
