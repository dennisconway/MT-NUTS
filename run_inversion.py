# -*- coding: utf-8 -*-


import numpy as np
from bisect import bisect as bis
from numpy.random import randn
import pystan
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from glob import glob
import matplotlib.gridspec as gridspec
from mtpy.core.edi import Edi


def get_prob_dist(fit, evals=np.linspace(-1, 6, 100), depths=np.linspace(0, 2000, 2001)):
    resistivities = fit.extract('R')['R'][:, 1:][:, ::-1]
    layers = fit.extract('T')['T'][:, ::-1]
    halfspaces = fit.extract('R')['R'][:, 0]
    end_res = []
    if len(resistivities.shape) == 1:
        for l in depths:
            curves = []
            for i, j, k in zip(layers, resistivities, halfspaces):
                if i > l:
                    curves.append(j)
                else:
                    curves.append(k)
            end_res.append(curves)
        data = np.array(end_res).transpose()
    else:
        for k in tqdm(range(len(layers)), desc='converting to r(z)'):
            layer, resistivity, halfspace = layers[k], resistivities[k], halfspaces[k]
            curves = []
            layer_interfaces = np.cumsum(layer)
            for i in depths:
                res = None
                for j, inter in enumerate(layer_interfaces):
                    if inter > i:
                        res = (resistivity[j])
                        break
                if not res:
                    res = halfspace
                curves.append(res)
            end_res.append(curves)
        data = np.array(end_res)
    disties = []
    for i in tqdm(range(data.shape[1]), desc='getting prob dists'):
        kde = gaussian_kde(data[:, i])
        disties.append(kde.evaluate(evals))
    return np.array(disties)


def convert_res(fit, min_app=0, max_app=4, freq_evals=300, r_evals=300):
    impedance = fit.extract('z_s')['z_s']
    frequencies = fit.data['FREQ']
    mt_data = fit.data['z_o']
    res_phs = np.zeros([freq_evals, 2, r_evals])
    phs_evals = np.linspace(0, 90, res_phs.shape[2])
    res_evals = np.linspace(min_app, max_app, res_phs.shape[2])
    dres = float(max_app - min_app) / res_phs.shape[2]
    dphs = 90. / res_phs.shape[2]
    freq_evals = np.linspace(np.log10(min(frequencies)),
                             np.log10(max(frequencies)), res_phs.shape[2])
    true_data = np.zeros([impedance.shape[1], 2])
    res, phs, freqs = [], [], []
    for fdx, freq in enumerate(frequencies):
        for sdx in range(impedance.shape[0]):
            res.append(np.log10(np.linalg.norm(
                impedance[sdx, fdx])**2 / (2 * np.pi * freq)))
            phs.append(np.degrees(np.arctan2(
                impedance[sdx, fdx, 1], impedance[sdx, fdx, 0])))
            freqs.append(np.log10(freq))
        true_data[fdx, 0] = np.log10(np.linalg.norm(
            mt_data[fdx])**2 / (2 * np.pi * freq))
        if len(mt_data[fdx]) == 1:
            true_data[fdx, 1] = np.degrees(np.arctan2(
                np.imag(mt_data[fdx]), np.real(mt_data[fdx])))
        else:
            true_data[fdx, 1] = np.degrees(
                np.arctan2(mt_data[fdx][1], mt_data[fdx][0]))
    kde_res = gaussian_kde(np.vstack([freqs, res]))
    kde_phs = gaussian_kde(np.vstack([freqs, phs]))
    for fdx in range(freq_evals.size):
        for i in range(res_evals.size):
            res_phs[fdx, 0, i] = kde_res.evaluate(
                [freq_evals[fdx], res_evals[i]])
            res_phs[fdx, 1, i] = kde_phs.evaluate(
                [freq_evals[fdx], phs_evals[i]])
        res_phs[fdx, 0] = res_phs[fdx, 0] / (np.sum(res_phs[fdx, 0]) * dres)
        res_phs[fdx, 1] = res_phs[fdx, 1] / (np.sum(res_phs[fdx, 1]) * dphs)
    return true_data, res_phs


def plot_resphs(res_phs, true_data, data_frequencies, min_app=0, max_app=4):
    plt.style.use('classic')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(wspace=0.05, hspace=0.15)
    new_labels = [0, 15, 30, 45, 60, 75, 90]
    res_evals = np.linspace(min_app, max_app, res_phs.shape[2])
    phs_evals = np.linspace(0, 90, res_phs.shape[2])
    data_periods = 1 / np.array(data_frequencies)
    periods = np.linspace(np.log10(min(data_periods)),
                          np.log10(max(data_periods)), res_phs.shape[0])
    plt.subplot(gs[0])
    plt.ylabel(r'Log$_{10}$ $\rho_{app}$ ($\Omega$m)')
    plt.xticks([])
    plt.xlim([periods[0], periods[-1]])
    plt.ylim([0, 3])
    plt.pcolor(periods[::-1], res_evals, res_phs[:, 0, :].transpose(),
               cmap='viridis_r')
    label = r'Log$_{10}$ $\rho_{app}$ probability density ($\Omega^{-1}m^{-1}$)'
    cbar = plt.colorbar(label=label, aspect=10, ticks=[0, 1, 2, 3, 4, 5],
                        format='%.2f')
    plt.clim([0, 5])
    plt.scatter(np.log10(data_periods),
                true_data[:, 0], c='r', lw=0, label='Inverted data', alpha=0.5)
    plt.legend(loc=2, fancybox=True, framealpha=0.5)
    plt.subplot(gs[1])
    plt.xlabel('Log$_{10}$ Period (s)')
    plt.ylabel('$\phi$')
    plt.yticks(new_labels)
    plt.xlim([periods[0], periods[-1]])
    plt.ylim([0, 90])
    plt.pcolor(periods[::-1], phs_evals, res_phs[:, 1, :].transpose(),
               cmap='viridis_r')
    plt.colorbar(label=r'$\phi$ probability density', aspect=10,
                 ticks=[0, 0.05, 0.1, 0.15, 0.2])
    plt.clim([0, 0.2])
    plt.grid(False)
    plt.scatter(np.log10(data_periods),
                true_data[:, 1], lw=0, c='r', label='Inverted data', alpha=0.5)
    if raw_input('Save fig? ').lower() == 'y':
        plt.savefig('resphs.pdf')
    else:
        plt.show()


def plot_distribution(disties, evals=np.linspace(-1, 6, 100),
                      depths=np.linspace(0, 2000, 2001), rho=False, clip=0.1):
    plt.cla()
    plt.clf()
    plt.style.use('classic')
    if rho:
        plt.plot(np.log10(rho[:len(depths)]), depths,
                 c='r', lw=2, ls='--', label='Inverted profile')
        print len(rho), len(depths)
        plt.legend(loc=3, fancybox=True, framealpha=0.5)
    else:
        rho = [10**evals[np.argmax(i)] for i in disties]
    plt.pcolor(evals, depths, disties / 14.3, cmap='viridis_r')
    plt.clim([0, clip])
    plt.xlim([np.min(evals), np.max(evals)])
    plt.ylim([0, 2000])
    plt.gca().invert_yaxis()
    plt.xlabel('Log$_{10}$ resistivity ($\Omega$m)')
    plt.ylabel('Depth (m)')
    plt.grid(zorder=0)
    # plt.text(-0.8, 200, 'a.', fontsize=28, color='white')
    plt.tight_layout()
    label = 'Log$_{10}$ resistivity probability density ($\Omega^{-1}m^{-1}$)'
    plt.colorbar(label=label)
    if raw_input('Save figure? ').lower() == 'y':
        plt.savefig('Probability.pdf')
    else:
        plt.show()


def depth_to_basement(fit, base_res=3, bins=50):
    plt.cla()
    plt.clf()
    plt.style.use('classic')
    plt.style.use('bmh')
    plt.xlim([0,2000])
    resistivities = fit.extract('R')['R'][:, ::-1]
    layers = fit.extract('T')['T'][:, ::-1]
    basements = []
    for i in range(resistivities.shape[0]):
        ressies = (resistivities[i] - base_res > 0)
        basement = np.sum(layers[i])
        for j in range(ressies.shape[0]):
            if sum(ressies[j:]) == ressies[j:].size:
                basement = np.sum(layers[i, :j])
                break
        basements.append(basement)
    evals = np.linspace(0, 2000, 10000)
    kde = gaussian_kde(basements)
    disty = kde.evaluate(evals)
    n, bins, patches = plt.hist(basements, bins, normed=1, alpha=0.75)
    plt.plot(evals, disty, lw=3, c='green')
    plt.ylabel('Probability density ($m^{-1}$)')
    plt.xlabel('Depth to basement (m)')
    plt.tight_layout()
    if raw_input('Save fig? ').lower() == 'y':
        plt.savefig('depth_cs.pdf')
    else:
        plt.show()
    return basements


def plot_traces(data1, data2, number_chains):
    import matplotlib.ticker as ticker
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    plt.cla()
    plt.clf()
    chains = []
    plt.style.use('bmh')
    ax = plt.subplot(gs[0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    chain_length = len(data1) / number_chains
    plt.scatter(range(len(data1)), data1, alpha=1.,
                c=np.array([(i // chain_length)**2 for i in range(len(data1))]),
                cmap='Dark2_r')
    plt.xlim([0, len(data1)])
    # plt.xlabel('Sample number')
    plt.ylabel(r'$\rho_1$')
    ax = plt.subplot(gs[1])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    plt.scatter(range(len(data2)), data2, alpha=1.,
                c=np.array([(i // chain_length)**2 for i in range(len(data1))]),
                cmap='Dark2_r')
    plt.xlim([0, len(data1)])
    plt.ylabel(r'$\rho_4$')
    plt.xlabel('Sample number')


if __name__ == "__main__":
    edis = glob('*.edi')
    if not edis:
        raise ValueError('please run in directory with edi files')
    for idx, fn in enumerate(edis):
        print idx + 1, fn
    number = int(raw_input('which number edi?: '))
    mode = raw_input('xy mode or yx mode? (default xy): ').lower()
    if not mode:
        mode = 'xy'
    elif mode not in ['xy', 'yx']:
        raise ValueError('please specify xy or yx')
    ef = raw_input('error floor (default 0.05): ')
    if ef:
        ef = float(ef)
    else:
        ef = 0.05
    edi = Edi(edis[number - 1])
    freq = edi.freq
    zm = np.empty([len(freq), 2])
    sig = np.empty([len(freq)])
    for i in range(len(freq)):
        if mode == 'xy':
            zm[i, 0] = -np.real(edi.Z.z[i, 1, 0])
            zm[i, 1] = -np.imag(edi.Z.z[i, 1, 0])
            zerr = max(edi.Z.zerr[i, 1, 0], np.linalg.norm(
                edi.Z.z[i, 1, 0]) * ef)
            sig[i] = zerr
        else:
            zm[i, 0] = np.real(edi.Z.z[i, 0, 1])
            zm[i, 1] = np.imag(edi.Z.z[i, 0, 1])
            zerr = max(edi.Z.zerr[i, 0, 1], np.linalg.norm(
                edi.Z.z[i, 0, 1]) * ef)
            sig[i] = zerr
    stan_file = 'conductance_sharp.h'
    try:
        layers = int(raw_input('how many layers (default 4): '))
    except ValueError:
        layers = 4
    mt_data = {'F': len(freq), 'FREQ': freq, 'z_o': zm,
               'sigma': sig, 'L': layers}
    chains = (raw_input('how many chains (default 3)?: '))
    if chains:
        chains = int(chains)
    else:
        chains = 3
    iterations = (raw_input('how many iterations (default 1000): '))
    if iterations:
        iterations = int(iterations)
    else:
        iterations = 1000
    warmup = raw_input('how many warmup iterations (default 1000): ')
    if warmup:
        warmup = int(warmup)
    else:
        warmup = 1000
    fit = pystan.stan(file=stan_file, data=mt_data, iter=iterations + warmup,
                      chains=chains, control={'adapt_delta': 1 - 1e-5},
                      warmup=warmup)
    plot = raw_input('plot results? (y/n, default y): ')
    if plot.lower() != 'n':
        min_res = raw_input('minimum log10 resistivity (default -1): ')
        if min_res:
            min_res = float(min_res)
        else:
            min_res = -1
        max_res = raw_input('maximum log10 resistivity (default 6): ')
        if max_res:
            max_res = float(max_res)
        else:
            min_res = 6
        max_depth = raw_input('maximum depth in metres (default 2000): ')
        if max_depth:
            max_depth = int(max_depth)
        else:
            max_depth = 2000
        disties = get_prob_dist(
                    fit,
                    evals=np.linspace(min_res, max_res, 100),
                    depths=np.linspace(0, max_depth, max_depth+1)
                    )
        plot_distribution(disties, evals=linspace(min_res, max_res, 100),
                          depths=np.linspace(0, max_depth, max_depth+1))
