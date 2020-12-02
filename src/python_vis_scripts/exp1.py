import numpy as np
import matplotlib.pyplot as plt
import os

directory = '../graph_data/exp1/'
out_directory = '../graph_data/exp1/out/'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

# params
uniform_params = {
    0: [15.0, 50.0, 35.0, 90.0, -60.0, 40.0, -50.0, 36.0],
    1: [-200.0, 200.0, -200.0, 200.0, -200.0, 200.0, -200.0, 200.0],
    2: [-200.0, 200.0, 0.0, 10.0, -200.0, 200.0, 50.0, 55.0]
}
params = {
    'uniform' : uniform_params
}

def get_data_info(line):
    d = line.strip('\n').split(" ")
    info = {}
    info['num_distr_parms'] = int(d[0])
    info['dim'] = int(d[1])
    info['num_samples'] = int(d[2])
    info['mlp'] = int(d[3])
    return info

def process_line(line):
    d = line.strip('\n').split(" ")
    n = int(d[0])
    time = float(d[1])
    return np.array([n, time])

def plot_data(data, data_info, dist):
    for i in range(data.shape[0]):
        x = data[i, :, 0]
        y = data[i, :, 1]
        plt.plot(x, y, label='params ' + str(i))
    plt.title("%s distribution, %sD data, %s mlp" % (dist, data_info['dim'], data_info['mlp']))
    plt.xlabel('N')
    plt.ylabel('t [s]')
    plt.xlim(0, x[-1])
    plt.legend()
    plot_name = "%s_%sD_%smlp.pdf" % (dist, data_info['dim'], data_info['mlp'])
    plt.savefig(os.path.join(out_directory, plot_name))
    plt.clf()

def process_data(fname):
    with open(directory + fname, mode='r') as f:
        dist = fname.split('_')[0]
        data_info = get_data_info(f.readline())

        matrix = np.zeros((data_info['num_distr_parms'], data_info['num_samples'], 2))

        for i in range(data_info['num_distr_parms']):
            for j in range(data_info['num_samples']):
                data = process_line(f.readline())
                matrix[i, j, :] = data
        plot_data(matrix, data_info, dist)


for filename in os.listdir(directory):
    if filename.endswith("txt"):
        process_data(filename)
