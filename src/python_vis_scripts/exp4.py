import numpy as np
import matplotlib.pyplot as plt
import os

directory = '../graph_data/exp4_solarium/10_7_pq_noskew_exp/'
out_directory = '../graph_data/exp4_solarium/10_7_pq_noskew_exp/out/'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

num_dimensions = 3
data_info = {}

distr_data = {}

k_data_time = {}
kNN_data_time = {}

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
        plt.legend()
        plot_name = "%sD_%spoints.pdf" % (i + 2, info['num_samples'])
        plt.savefig(os.path.join(out_directory, plot_name))
        plt.clf()


def get_data_info_kd(line):
    d = line.strip('\n').split(" ")
    info = {}
    info['dim'] = int(d[0])
    info['num_samples'] = int(d[1])
    info['num_mlp'] = int(d[2])
    info['num_search_params'] = int(d[3])
    return info

def get_data_info_naive(line):
    d = line.strip('\n').split(" ")
    info = {}
    info['dim'] = int(d[0])
    info['num_samples'] = int(d[1])
    info['num_search_params'] = int(d[2])
    return info

def process_line_kd_kNN(line):
    d = line.strip('\n').split(" ")
    k = int(d[0])
    time = float(d[1])
    vis_leaves = int(d[2])
    return k, time, vis_leaves

def process_line_kd_sph(line):
    d = line.strip('\n').split(" ")
    radius = float(d[0])
    time = float(d[1])
    vis_leaves = int(d[2])
    return radius, time, vis_leaves

def process_line_kd_rect(line):
    d = line.strip('\n').split(" ")
    time = float(d[0])
    vis_leaves = int(d[1])
    return time, vis_leaves

def process_line_naive_kNN(line):
    d = line.strip('\n').split(" ")
    k = int(d[0])
    time = float(d[1])
    return k, time

def process_line_naive_sph(line):
    d = line.strip('\n').split(" ")
    radius = float(d[0])
    time = float(d[1])
    return radius, time

def process_line_naive_rect(line):
    d = line.strip('\n').split(" ")
    time = float(d[0])
    return time

naive_kNN = {}
naive_sph = {}
naive_rect = {}

def process_naive_kNN(fname):
    with open(directory + fname, mode='r') as f:
        file_info = get_data_info_naive(f.readline())
        distr_type = fname.split('_')[2]

        for i in range(file_info['num_search_params']):
            k, time = process_line_naive_kNN(f.readline())

            if k in naive_kNN.keys():
                if distr_type in naive_kNN[k].keys():
                    naive_kNN[k][distr_type][file_info['dim']] = time
                else:
                    naive_kNN[k][distr_type] = {}
                    naive_kNN[k][distr_type][file_info['dim']] = time
            else:
                naive_kNN[k] = {}
                naive_kNN[k][distr_type] = {}
                naive_kNN[k][distr_type][file_info['dim']] = time

def process_naive_sph(fname):
    with open(directory + fname, mode='r') as f:
        file_info = get_data_info_naive(f.readline())
        distr_type = fname.split('_')[2]

        for i in range(file_info['num_search_params']):
            radius, time = process_line_naive_sph(f.readline())

            if radius in naive_sph.keys():
                if distr_type in naive_sph[radius].keys():
                    naive_sph[radius][distr_type][file_info['dim']] = time
                else:
                    naive_sph[radius][distr_type] = {}
                    naive_sph[radius][distr_type][file_info['dim']] = time
            else:
                naive_sph[radius] = {}
                naive_sph[radius][distr_type] = {}
                naive_sph[radius][distr_type][file_info['dim']] = time

def process_naive_rect(fname):
    with open(directory + fname, mode='r') as f:
        file_info = get_data_info_naive(f.readline())
        distr_type = fname.split('_')[2]

        for i in range(file_info['num_search_params']):
            time = process_line_naive_rect(f.readline())

            if i in naive_rect.keys():
                if distr_type in naive_rect[i].keys():
                    naive_rect[i][distr_type][file_info['dim']] = time
                else:
                    naive_rect[i][distr_type] = {}
                    naive_rect[i][distr_type][file_info['dim']] = time
            else:
                naive_rect[i] = {}
                naive_rect[i][distr_type] = {}
                naive_rect[i][distr_type][file_info['dim']] = time

kd_kNN = {}
kd_sph = {}
kd_rect = {}

def process_kd_kNN(fname):
    with open(directory + fname, mode='r') as f:
        file_info = get_data_info_kd(f.readline())
        distr_type = fname.split('_')[1]

        for i in range(file_info['num_search_params']):
            k, time, vis_leaves = process_line_kd_kNN(f.readline())

            if k in kd_kNN.keys():
                if distr_type in kd_kNN[k].keys():
                    if file_info['dim'] in kd_kNN[k][distr_type]:
                        kd_kNN[k][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
                    else:
                        kd_kNN[k][distr_type][file_info['dim']] = {}
                        kd_kNN[k][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
                else:
                    kd_kNN[k][distr_type] = {}
                    kd_kNN[k][distr_type][file_info['dim']] = {}
                    kd_kNN[k][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
            else:
                kd_kNN[k] = {}
                kd_kNN[k][distr_type] = {}
                kd_kNN[k][distr_type][file_info['dim']] = {}
                kd_kNN[k][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]

def process_kd_sph(fname):
    with open(directory + fname, mode='r') as f:
        file_info = get_data_info_kd(f.readline())
        distr_type = fname.split('_')[1]

        for i in range(file_info['num_search_params']):
            radius, time, vis_leaves = process_line_kd_sph(f.readline())

            if radius in kd_sph.keys():
                if distr_type in kd_sph[radius].keys():
                    if file_info['dim'] in kd_sph[radius][distr_type]:
                        kd_sph[radius][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
                    else:
                        kd_sph[radius][distr_type][file_info['dim']] = {}
                        kd_sph[radius][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
                else:
                    kd_sph[radius][distr_type] = {}
                    kd_sph[radius][distr_type][file_info['dim']] = {}
                    kd_sph[radius][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
            else:
                kd_sph[radius] = {}
                kd_sph[radius][distr_type] = {}
                kd_sph[radius][distr_type][file_info['dim']] = {}
                kd_sph[radius][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]

def process_kd_rect(fname):
    with open(directory + fname, mode='r') as f:
        file_info = get_data_info_kd(f.readline())
        distr_type = fname.split('_')[1]

        for i in range(file_info['num_search_params']):
            time, vis_leaves = process_line_kd_rect(f.readline())

            if i in kd_rect.keys():
                if distr_type in kd_rect[i].keys():
                    if file_info['dim'] in kd_rect[i][distr_type]:
                        kd_rect[i][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
                    else:
                        kd_rect[i][distr_type][file_info['dim']] = {}
                        kd_rect[i][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
                else:
                    kd_rect[i][distr_type] = {}
                    kd_rect[i][distr_type][file_info['dim']] = {}
                    kd_rect[i][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]
            else:
                kd_rect[i] = {}
                kd_rect[i][distr_type] = {}
                kd_rect[i][distr_type][file_info['dim']] = {}
                kd_rect[i][distr_type][file_info['dim']][file_info['num_mlp']] = [time , vis_leaves]


for filename in os.listdir(directory):
    if filename.endswith("txt"):
        if filename.startswith('naive_kNN'):
            process_naive_kNN(filename)
        elif filename.startswith('kNN'):
            process_kd_kNN(filename)
        elif filename.startswith('naive_sph'):
            process_naive_sph(filename)
        elif filename.startswith('sph'):
            process_kd_sph(filename)
        elif filename.startswith('naive_rect'):
            process_naive_rect(filename)
        elif filename.startswith('rect'):
            process_kd_rect(filename)

num_ks = 4
ks = [100, 500, 1000, 10000]
distributions = ['uniform', 'normal']
mpl = [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
mpl_size = len(mpl)

def plot_kNN_data(naive_kNN, kd_kNN, info):
    for i in range(num_dimensions):
        for j in range(num_ks):
            for d in distributions:
                x = []
                y = []
                for m in range(mpl_size):
                    x.append(mpl[m])
                    y.append(kd_kNN[ks[j]][d][i + 2][mpl[m]][0] * 1000)
                plt.plot(x, y, label=d)
                y_n = []
                for n in range(mpl_size):
                    y_n.append(naive_kNN[ks[j]][d][i + 2] * 1000)
                plt.plot(mpl, y_n, label=d + ' naive')

            plt.xscale("log")
            plt.yscale("log")
            plt.title("%sD data, %s points, k = %s " % (i + 2, info['num_samples'], ks[j]))
            plt.xlabel('max points in leaf')
            plt.ylabel('t [ms]')
            plt.legend()
            plot_name = "%sD_%spoints%sk.pdf" % (i + 2, info['num_samples'], ks[j])
            plt.savefig(os.path.join(out_directory, plot_name))
            plt.clf()

    for i in range(num_dimensions):
        for j in range(num_ks):
            for d in distributions:
                x = []
                y = []
                for m in range(mpl_size):
                    x.append(mpl[m])
                    y.append(kd_kNN[ks[j]][d][i + 2][mpl[m]][1])
                plt.plot(x, y, label=d)

            plt.xscale("log")
            plt.yscale("log")
            plt.title("%sD data, %s points, k = %s " % (i + 2, info['num_samples'], ks[j]))
            plt.xlabel('max points in leaf')
            plt.ylabel('num visited leaves')
            plt.legend()
            plot_name = "%sD_%spoints%sk_vis_leaves.pdf" % (i + 2, info['num_samples'], ks[j])
            plt.savefig(os.path.join(out_directory, plot_name))
            plt.clf()


radii = [5, 15, 50, 100]
num_radii = 4
def plot_sph_data(naive_sph, kd_sph, info):
    for i in range(num_dimensions):
        for j in range(num_radii):
            for d in distributions:
                x = []
                y = []
                for m in range(mpl_size):
                    x.append(mpl[m])
                    y.append(kd_sph[radii[j]][d][i + 2][mpl[m]][0] * 1000)
                plt.plot(x, y, label=d)
                y_n = []
                for n in range(mpl_size):
                    y_n.append(naive_sph[radii[j]][d][i + 2] * 1000)
                plt.plot(mpl, y_n, label=d + ' naive')

            plt.xscale("log")
            plt.yscale("log")
            plt.title("%sD data, %s points, radius = %s " % (i + 2, info['num_samples'], radii[j]))
            plt.xlabel('max points in leaf')
            plt.ylabel('t [ms]')
            plt.legend()
            plot_name = "%sD_%spoints%sradius.pdf" % (i + 2, info['num_samples'], radii[j])
            plt.savefig(os.path.join(out_directory, plot_name))
            plt.clf()

    for i in range(num_dimensions):
        for j in range(num_radii):
            for d in distributions:
                x = []
                y = []
                for m in range(mpl_size):
                    x.append(mpl[m])
                    y.append(kd_sph[radii[j]][d][i + 2][mpl[m]][1])
                plt.plot(x, y, label=d)

            plt.xscale("log")
            plt.yscale("log")
            plt.title("%sD data, %s points, radius = %s " % (i + 2, info['num_samples'], radii[j]))
            plt.xlabel('max points in leaf')
            plt.ylabel('num visited leaves')
            plt.legend()
            plot_name = "%sD_%spoints%sradius_vis_leaves.pdf" % (i + 2, info['num_samples'], radii[j])
            plt.savefig(os.path.join(out_directory, plot_name))
            plt.clf()

rect_range = ["small", "big"]
num_rect_r = 2
def plot_rect_data(naive_rect, kd_rect, info):
    for i in range(num_dimensions):
        for j in range(num_rect_r):
            for d in distributions:
                x = []
                y = []
                for m in range(mpl_size):
                    x.append(mpl[m])
                    y.append(kd_rect[j][d][i + 2][mpl[m]][0] * 1000)
                plt.plot(x, y, label=d)
                y_n = []
                for n in range(mpl_size):
                    y_n.append(naive_rect[j][d][i + 2] * 1000)
                plt.plot(mpl, y_n, label=d + ' naive')

            plt.xscale("log")
            plt.yscale("log")
            plt.title("%sD data, %s points, range = %s " % (i + 2, info['num_samples'], rect_range[j]))
            plt.xlabel('max points in leaf')
            plt.ylabel('t [ms]')
            plt.legend()
            plot_name = "%sD_%spoints%s_rect_range.pdf" % (i + 2, info['num_samples'], rect_range[j])
            plt.savefig(os.path.join(out_directory, plot_name))
            plt.clf()

    for i in range(num_dimensions):
        for j in range(num_rect_r):
            for d in distributions:
                x = []
                y = []
                for m in range(mpl_size):
                    x.append(mpl[m])
                    y.append(kd_rect[j][d][i + 2][mpl[m]][1])
                plt.plot(x, y, label=d)
            plt.xscale("log")
            plt.yscale("log")
            plt.title("%sD data, %s points, range = %s " % (i + 2, info['num_samples'], rect_range[j]))
            plt.xlabel('max points in leaf')
            plt.ylabel('num visited leaves')
            plt.legend()
            plot_name = "%sD_%spoints%s_rect_range_vis_leaves.pdf" % (i + 2, info['num_samples'], rect_range[j])
            plt.savefig(os.path.join(out_directory, plot_name))
            plt.clf()

inf = {}
inf['num_samples'] = 1000000
plot_kNN_data(naive_kNN, kd_kNN, inf)
plot_sph_data(naive_sph, kd_sph, inf)
plot_rect_data(naive_rect, kd_rect, inf)