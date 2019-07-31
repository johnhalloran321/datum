#!/usr/bin/env python
#
# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2018 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0

from __future__ import with_statement

__authors__ = ['John Halloran <halloj3@uw.edu>' ]

import os
import sys
import argparse
import cPickle as pickle
import multiprocessing as mp
import struct
import itertools
import numpy as np
import math
import random
import re
import csv

from pyFiles.spectrum import MS2Spectrum, MS2Iterator
from pyFiles.peptide import Peptide
from pyFiles.normalize import pipeline
from pyFiles.constants import allPeps, mass_h, mass_h2o
from digest import (check_arg_trueFalse,
                    parse_var_mods, load_digested_peptides_var_mods)
from pyFiles.dideaEncoding import histogram_spectra, simple_uniform_binwidth, bins_to_vecpt_ratios

from pyFiles.constants import allPeps
from pyFiles.shard_spectra import (load_spectra,
                                   load_spectra_ret_dict)
from subprocess import call, check_output

# set stdout and stderr for subprocess
# stdo = open(os.devnull, "w")

# stdo = sys.stdout
stdo = open(os.devnull, "w")
stde = sys.stdout
# stde = stdo

# stde = open('gmtk_err', "w")

# # set stdout for all default streams
# sys.stdout = open("dripSearch_output", "w")

def parseInputOptions():
    parser = argparse.ArgumentParser(conflict_handler='resolve', 
                                     description="Run a DRIP database search given an MS2 file and the results of a FASTA file processed using dripDigest.")
    ############## input and output options
    ##### inputs
    iFileGroup = parser.add_argument_group('iFileGroup', 'Necessary input files.')
    help_spectra = '<string> - The name of the file from which to parse fragmentation spectra, in ms2 format.'
    iFileGroup.add_argument('--spectra', type = str, action = 'store',
                            help = help_spectra)
    help_psms = '<string> - Peptides for training'
    iFileGroup.add_argument('--psms', type = str, action = 'store',
                            help = help_psms)
    ###### output
    oFileGroup = parser.add_argument_group('oFileGroup', 'Output file options')
    oFileGroup.add_argument('--output', type = str,
                      help = 'Output file to write learned parameters')
    ############## training parameters
    trainingParamsGroup = parser.add_argument_group('trainingParamsGroup', 'Training parameter options.')
    help_charge = """<int> - Charge state to learn. Default=2."""
    trainingParamsGroup.add_argument('--charge', type = str, action = 'store', default = "2", help = help_charge)
    help_num_threads = '<integer> - the number of threads to run on a multithreaded CPU. If supplied value is greater than number of supported threads, defaults to the maximum number of supported threads minus one. Multithreading is not suppored for cluster use as this is typically handled by the cluster job manager. Default=1.'
    trainingParamsGroup.add_argument('--num-threads', type = int, action = 'store', 
                                     default = 1, help = help_num_threads)
    help_num_bins = '<integer> - The number of bins to quantize observed spectrum. Default=2000.'
    trainingParamsGroup.add_argument('--num_bins', type = int, action = 'store', 
                                     default = 2000, help = help_num_bins)
    help_learning_rate = '<float> - The learning rate to be used during training.'
    trainingParamsGroup.add_argument('--lrate', type = float, action = 'store', 
                                     default = 0.01, help = help_learning_rate)
    help_thresh = '<float> - Threshold for training convergence.'
    trainingParamsGroup.add_argument('--thresh', type = float, action = 'store', 
                                     default = 0.01, help = help_thresh)
    help_max_iters = '<int> - Max iterations during training'
    trainingParamsGroup.add_argument("--maxIters", type = int, action = 'store', 
                                      default = 10000, help = help_max_iters)
    help_shift_prior = '<float> - Prior on spectra shift = 0.'
    trainingParamsGroup.add_argument('--lmb0Prior', type = float, action = 'store', 
                                     default = 0.3, help = help_shift_prior)
    help_l2Reg = '<T|F> - L2-regularize training objective function'
    trainingParamsGroup.add_argument('--l2Reg', type = str, action = 'store', 
                                     default = 'true', help = help_l2Reg)
    help_cve = '<T|F> - Use general CVE emission'
    trainingParamsGroup.add_argument('--cve', type = str, action = 'store', 
                                     default = 'false', help = help_cve)
    help_alpha = '<float> - L2-regularization strength.'
    trainingParamsGroup.add_argument('--alpha', type = float, action = 'store', 
                                     default = 0.4, help = help_alpha)
    help_max_mass = '<float> - Maximum peptide mass.'
    trainingParamsGroup.add_argument("--max_mass", type = int, action = 'store', 
                                      default = 6325, help = help_max_mass)
    ######### encoding options
    parser.add_argument('--normalize', dest = "normalize", type = str,
                        help = "Name of the spectrum preprocessing pipeline.", 
                        default = 'rank0')
    parser.add_argument('--vekind', dest = "vekind", type = str,
                        help = "Emission function to use.", 
                        default = 'intensity')
    return parser.parse_args()

def process_args(args):
    """ Check whether relevant directories already exist (remove them if so), create
        necessary directories.  Adjust parameters.

        pre:
        - args has been created by calling parse_args()
        
        post:
        - Relevant directories will be first removed then created, boolean parameters
          will be adjusted to reflect their boolean command line strings
    """

    # set true or false strings to booleans
    args.l2Reg = check_arg_trueFalse(args.l2Reg)
    args.cve = check_arg_trueFalse(args.cve)
    # make sure number of input threads does not exceed number of supported threads
    if args.num_threads > mp.cpu_count():
        args.num_threads = mp.cpu_count()

def find_sid(array, sid):
    for ind, el in zip(range(len(array)), array):
        if(el==sid):
            return ind
    return -1

def byIons(peptide, charge):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic', mass_op)
    nterm_fragments = []
    cterm_fragments = []
    # iterate through possible charges
    for c in range(1,charge):
        nterm_fragments += [int(round((b+c)/c)) for b in ntm[1:-1]]
        cterm_fragments += [int(round((y+18+c)/c)) for y in ctm[1:-1]]
    return (nterm_fragments,cterm_fragments)

def dideaShiftDotProducts(peptide, charge, bins, num_bins):
    """ For observed spectrum s and peptide x with corresponding b- and y-ion vectors b and y, calculate (b'+y')^T(s_\tau - s),
    where s_\tau is s shifted by \tau units, and b'/y' are boolean vectors of length len(s) with ones only in indices corresponding to 
    b and y-ions
    """
    (bions, yions) = byIons(peptide,charge)
    tauCard = 75.0
    lastBin = num_bins-1
    # output - vector corresponding to above dot-product per every shift
    shiftDotProducts = [0.0]*int(tauCard)

    for ind,tau in enumerate(range(-37,38)):
        bySum = 0
        for b,y in zip(bions,yions):
            bt = b-tau
            yt = y-tau
            # start with b-ions
            if bt < num_bins: # make sure we haven't shifted outside of the bins
                if bt >= 0: # make sure we haven't shifted to far left
                    if b < num_bins: # make sure original b-ion was in range
                        bySum += bins[bt]-bins[b]
                    else: # original b-ion was out of range
                        bySum += bins[bt]
                else: # we've shifted too far left
                    if b < num_bins: # make sure original b-ion was in range
                        bySum -= bins[b]
            else: # check original b-ion
                if b < num_bins: # make sure original b-ion was in range
                    bySum -= bins[b]

            # now y-ions
            if yt < num_bins: # make sure we haven't shifted outside of the bins
                if yt >= 0: # make sure we haven't shifted to far left
                    if y < num_bins: # make sure original b-ion was in range
                        bySum += bins[yt]-bins[y]
                    else: # original b-ion was out of range
                        bySum += bins[yt]
                else: # we've shifted too far left
                    if b < num_bins: # make sure original b-ion was in range
                        bySum -= bins[y]
            else: # check original y-ion
                if y < num_bins: # make sure original b-ion was in range
                    bySum -= bins[y]
        shiftDotProducts[ind] = bySum
    return shiftDotProducts

def dideaShiftSepDotProducts(peptide, charge, bins, num_bins):
    """ For observed spectrum s and peptide x with corresponding b- and y-ion vectors b and y, calculate (b'+y')^T(s_\tau - s),
    where s_\tau is s shifted by \tau units, and b'/y' are boolean vectors of length len(s) with ones only in indices corresponding to 
    b and y-ions
    """
    (bions, yions) = byIons(peptide,charge)
    tauCard = 75.0
    lastBin = num_bins-1
    # output - vector corresponding to above dot-product per every shift
    shiftSepDotProducts = [[0.0, 0.0] for _ in range(int(tauCard))]

    for ind,tau in enumerate(range(-37,38)):
        bSum = 0
        ySum = 0
        for b,y in zip(bions,yions):
            bt = b-tau
            yt = y-tau
            # start with b-ions
            if bt < num_bins: # make sure we haven't shifted outside of the bins
                if bt >= 0: # make sure we haven't shifted to far left
                    if b < num_bins: # make sure original b-ion was in range
                        bSum += bins[bt]-bins[b]
                    else: # original b-ion was out of range
                        bSum += bins[bt]
                else: # we've shifted too far left
                    if b < num_bins: # make sure original b-ion was in range
                        bSum -= bins[b]
            else: # check original b-ion
                if b < num_bins: # make sure original b-ion was in range
                    bSum -= bins[b]

            # now y-ions
            if yt < num_bins: # make sure we haven't shifted outside of the bins
                if yt >= 0: # make sure we haven't shifted to far left
                    if y < num_bins: # make sure original b-ion was in range
                        ySum += bins[yt]-bins[y]
                    else: # original b-ion was out of range
                        ySum += bins[yt]
                else: # we've shifted too far left
                    if b < num_bins: # make sure original b-ion was in range
                        ySum -= bins[y]
            else: # check original y-ion
                if y < num_bins: # make sure original b-ion was in range
                    ySum -= bins[y]
        shiftSepDotProducts[ind][0] = bSum
        shiftSepDotProducts[ind][1] = ySum
    return shiftSepDotProducts

def dideaShiftAllIons(peptide, charge, bins, num_bins):
    """ For observed spectrum s and peptide x with corresponding b- and y-ion vectors b and y, calculate (b'+y')^T(s_\tau - s),
    where s_\tau is s shifted by \tau units, and b'/y' are boolean vectors of length len(s) with ones only in indices corresponding to 
    b and y-ions
    """
    (bions, yions) = byIons(peptide,charge)
    tauCard = 75.0
    lastBin = num_bins-1
    # output - vector corresponding to above dot-product per every shift
    peakList = []

    for ind,tau in enumerate(range(-37,38)):
        ionIntensities = []
        bSum = 0
        ySum = 0
        for b,y in zip(bions,yions):
            bt = b-tau
            yt = y-tau
            # start with b-ions
            if bt>= 0 and bt < num_bins: # make sure we haven't shifted outside of the bins
                ionIntensities.append(bins[bt])
            if yt>= 0 and yt < num_bins: # make sure we haven't shifted outside of the bins
                ionIntensities.append(bins[yt])
            if not ionIntensities:
                ionIntensities.append(0.0)
        peakList.append(ionIntensities)

    return peakList

def genDideaTrainingData(options):
    """Create didea log-sum-exp training data"""
    output = {}
    tbl = 'monoisotopic'
    preprocess = pipeline(options.normalize)
    # Generate the histogram bin ranges
    ranges = simple_uniform_binwidth(0, options.num_bins, bin_width = 1.0)

    vc = int(options.charge)
    validcharges = set([vc])

    spectra = list(s for s in MS2Iterator(options.spectra, False) if
                   set(s.charges) & validcharges)

    if len(spectra) == 0:
        print >> sys.stderr, 'There are no spectra with charge_line (+%d).' % vc
        exit(-1)

    psm_sid = re.compile('\d+')
    psm_peptide = re.compile('[a-zA-Z]+$')
    #generate vector of sids/peptides
    print options.psms
    f = open(options.psms, "r")
    targets = [(int(l["Scan"]), l["Peptide"]) for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
# Scan	Peptide	Charge
    sids = []
    for (st,pt) in targets: sids.append(st)

    sidsPer = []
    for spec_ind,s in enumerate(spectra):
        sid = s.spectrum_id
        preprocess(s)
        # Generate the spectrum observations.
        # truebins = histogram_spectra(s, ranges, max)
        # bins = bins_to_vecpt_ratios(truebins, 'intensity', 0.0)
        bins = histogram_spectra(s, ranges, max)
        bins[0] = 0.0
        bins[-1] = 0.0
        # find psm
        ind = find_sid(sids, s.spectrum_id)

        sidsPer.append(sid)
        if (ind  != -1):
            p = Peptide(targets[ind][1])
            if not options.cve:
                for tau, tauProd in zip(range(-37,38),dideaShiftDotProducts(p, vc, bins, len(bins))):
                    output[spec_ind, tau] = tauProd
            else:
                # output[spec_ind] = np.array([peaks for peaks in dideaShiftAllIons(p, vc, bins, len(bins))])
                # output[spec_ind] = [peaks for peaks in dideaShiftAllIons(p, vc, bins, len(bins))]
                a = [peaks for peaks in dideaShiftAllIons(p, vc, bins, len(bins))]
                # pad uneven peak lists
                max_len = np.array([len(array) for array in a]).max()
                default_value = 0
                # note: have to work with transpose of the data matrix, so that we can do a simple
                # element-wise multiplication with the theta vector, i.e., so that the data matrix
                # has |\tau| columns
                output[spec_ind] = np.transpose([np.pad(array, (0, max_len - len(array)), mode='constant', constant_values=default_value) for array in a])
                # if len(output[spec_ind].shape) == 1:
                #     print p
                # else:
                #     print spec_ind, output[spec_ind].shape
                
                # for tau, peaks in zip(range(-37,38),dideaShiftAllIons(p, vc, bins, len(bins))):
                    # output[spec_ind, tau] = peaks
            # else:
            #     for tau, tauProd in zip(range(-37,38),dideaShiftSepDotProducts(p, vc, bins, len(bins))):
            #         output[spec_ind, tau] = tauProd

    # output is the number of spectra and a dictionary which has all of the shifted dot-products per spectrum
    return (len(spectra),output, sidsPer)

def genDideaTrainingData_par(options, numThreads):
    """Create didea log-sum-exp training data"""
    tbl = 'monoisotopic'
    preprocess = pipeline(options.normalize)
    # Generate the histogram bin ranges
    ranges = simple_uniform_binwidth(0, options.num_bins, bin_width = 1.0)

    vc = int(options.charge)
    validcharges = set([vc])

    spectra = list(s for s in MS2Iterator(options.spectra, False) if
                   set(s.charges) & validcharges)

    if len(spectra) == 0:
        print >> sys.stderr, 'There are no spectra with charge_line (+%d).' % vc
        exit(-1)

    psm_sid = re.compile('\d+')
    psm_peptide = re.compile('[a-zA-Z]+$')
    #generate vector of sids/peptides
    print options.psms
    f = open(options.psms, "r")
    targets = [(int(l["Scan"]), l["Peptide"]) for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
# Scan	Peptide	Charge
    sids = []
    for (st,pt) in targets: sids.append(st)

    output = [{} for _ in range(numThreads)]
    sidsPer = [0 for _ in range(numThreads)]
    for spec_ind,s in enumerate(spectra):
        output_ind = min(int(spec_ind / numThreads), numThreads-1)
        sid = s.spectrum_id
        preprocess(s)
        # Generate the spectrum observations.
        bins = histogram_spectra(s, ranges, max)
        bins[0] = 0.0
        bins[-1] = 0.0
        # find psm
        ind = find_sid(sids, s.spectrum_id)

        # sidsPer.append(sid)
        sidsPer[output_ind] += 1
        if (ind  != -1):
            p = Peptide(targets[ind][1])
            if not options.cve:
                for tau, tauProd in zip(range(-37,38),dideaShiftDotProducts(p, vc, bins, len(bins))):
                    output[output_ind][sidsPer[output_ind], tau] = tauProd
            else:
                for tau, peaks in zip(range(-37,38),dideaShiftAllIons(p, vc, bins, len(bins))):
                    output[output_ind][sidsPer[output_ind], tau] = peaks
    # output is the number of spectra and a dictionary which has all of the shifted dot-products per spectrum
    return (len(spectra),output, sidsPer)

def cve0(theta, s):
    # from NeurIPS 2018 paper:
    return math.exp(theta * s)

def cve_grad0(theta, s):
    # from NeurIPS 2018 paper:
    return s * math.exp(theta * s)

def batchFuncEvalLambdas(lambdas, options, numData, data):
    logSumExp = 0.0
    grad = {}
    numer = {}
    for tau in range(-37,38):
        grad[tau] = 0.0
        numer[tau] = 0.0

    for i in range(numData):
        # print "sid=%d" % sids[i]
        denom = 0.0
        sumExp = 0.0
        for tau in range(-37,38):
            currx = data[i,tau]
            lmb = lambdas[tau]
            sumExp += math.exp(lmb*currx)
            numer[tau] = currx*math.exp(lmb*currx)
            denom += math.exp(lmb*currx)
        logSumExp += math.log(sumExp)
        for tau in range(-37,38):
            if numer[tau] == 0.0:
                if options.l2Reg:
                    grad[tau] += (options.alpha/2.0 * lambdas[tau]) / float(numData)
                continue
            if numer[tau] < 0.0:
                grad[tau] -= math.exp(math.log(-numer[tau])-math.log(denom)) / float(numData)
            else:
                grad[tau] += math.exp(math.log(numer[tau])-math.log(denom)) / float(numData)
            if options.l2Reg:
                grad[tau] += (options.alpha/2.0 * lambdas[tau]) / float(numData)
        # exit(-1)
    return grad, logSumExp

def cve(theta, s):
    # try quadratic
    return 0.5 * (theta * s) ** 2 + 1.0
    # return math.exp(sum([math.log(0.5 * (theta * si) ** 2 + 1.0) for si in s]))

def cve_grad(theta, s):
    # from NeurIPS 2018 paper:
    # return s * math.exp(theta * s)
    # try quadratic
    return theta * s * s

def cve_general_ll(lambdas, options, numData, data):
    # calculate log-likelihood over the training data
    # data is serialized as: data[sid,\tau] = list of peak intensities
    # for shift \tau of PSM sid
    const = options.alpha/2.0 # * float(numData))
    totalLogProbEv = 0.0
    grad = np.array([0.0 for _ in range(-37,38)])
    # grad = {}
    # numer = {}
    # for tau in range(-37,38):
    #     grad[tau] = 0.0
        # numer[tau] = 0.0

    for i in range(numData):
        # print "sid=%d" % sids[i]
        denom = 0.0
        probEv = 0.0
        probs = cve(lambdas, data[i]) # matrix
        derivs = cve_grad(lambdas, data[i]) # matrix
        # since we transposed the data matrix to begin with, sum
        # along the columns of the above matrices
        likelihoods = np.exp(np.sum(np.log(probs), axis=0)) # column-wise sum
        g = np.sum(derivs / probs, axis=0) # column-wise sum
        # print lambdas.shape, data[i].shape, probs.shape, likelihoods.shape, g.shape
        probEv = np.sum(likelihoods)
        denom = probEv
        numer = likelihoods * g
        totalLogProbEv += math.log(probEv)
        grad += (const * lambdas + numer / denom) / float(numData)
        # for ind,tau in enumerate(range(-37,38)):
        #     if numer[ind] == 0.0:
        #         if options.l2Reg:
        #             grad[ind] += (options.alpha/2.0 * lambdas[ind]) / float(numData)
        #         continue
        #     if numer[ind] < 0.0:
        #         grad[ind] -= math.exp(math.log(-numer[ind])-math.log(probEv)) / float(numData)
        #     else:
        #         grad[ind] += math.exp(math.log(numer[ind])-math.log(probEv)) / float(numData)
        #     if options.l2Reg:
        #         grad[ind] += (options.alpha/2.0 * lambdas[ind]) / float(numData)
        # exit(-1)
    return grad, totalLogProbEv

def batchFuncEvalLambdas_mp(lambdas, options, numData, data, partitions, numThreads):
    # # calculate indices to be evaluated in partition
    # dataPoints = range(numData)
    # random.shuffle(dataPoints)
    # # calculate num spectra to score per thread
    # inc = int(numData / numThreads)
    # partitions = []
    # l = 0
    # r = inc
    # for i in range(numThreads-1):
    #     partitions.append(dataPoints[l:r])
    #     l += inc
    #     r += inc
    # partitions.append(dataPoints[l:])

    pool = mp.Pool(processes = numThreads)
    # perform map: distribute jobs to CPUs

    results = []
    for i in range(numThreads):
        res = pool.apply_async(funcEvalLambdas,
                               args=(lambdas, partitions[i], 
                                     data, numData, options.alpha, options.l2Reg))
        results.append(res)

    # results = [pool.apply_async(funcEvalLambdas,
    #                             args=(lambdas, partitions[i], 
    #                                   data, numData, options.alpha, options.l2Reg)) for i in range(numThreads)]

    pool.close()
    pool.join() # wait for jobs to finish before continuing

    # reduce results
    logSumExp = 0.0
    grad = {}
    for tau in range(-37,38):
        grad[tau] = 0.0
    for p in results:
        (g,l) = p.get()
        logSumExp += l
        for tau in range(-37,38):
            grad[tau] += g[tau]

    return grad, logSumExp

def batchFuncEvalLambdas_mpB(lambdas, options, numData, data, numThreads, numSamples):
    pool = mp.Pool(processes = numThreads)
    # perform map: distribute jobs to CPUs

    results = []
    for i in range(numThreads):
        res = pool.apply_async(funcEvalLambdasB,
                               args=(lambdas, data[i], numSamples[i], numData, options.alpha, options.l2Reg))
        results.append(res)

    # results = [pool.apply_async(funcEvalLambdasB,
    #                             args=(lambdas, data[i], numSamples[i], numData, options.alpha, options.l2Reg)) for i in range(numThreads)]

    pool.close()
    pool.join() # wait for jobs to finish before continuing

    # reduce results
    logSumExp = 0.0
    grad = {}
    for tau in range(-37,38):
        grad[tau] = 0.0
    for p in results:
        (g,l) = p.get()
        logSumExp += l
        for tau in range(-37,38):
            grad[tau] += g[tau]

    return grad, logSumExp

def funcEvalLambdas(lambdas, dataPoints, data, numData, alpha = 0.4, l2Reg = True):
    logSumExp = 0.0
    grad = {}
    numer = {}
    for tau in range(-37,38):
        grad[tau] = 0.0
        numer[tau] = 0.0

    for i in dataPoints:
        denom = 0.0
        sumExp = 0.0
        for tau in range(-37,38):
            currx = data[i,tau]
            lmb = lambdas[tau]
            sumExp += math.exp(lmb*currx)
            numer[tau] = currx*math.exp(lmb*currx)
            denom += math.exp(lmb*currx)
        logSumExp += math.log(sumExp)
        for tau in range(-37,38):
            if numer[tau] == 0.0:
                if l2Reg:
                    grad[tau] += (alpha/2.0 * lambdas[tau]) / float(numData)
                continue
            if numer[tau] < 0.0:
                grad[tau] -= math.exp(math.log(-numer[tau])-math.log(denom)) / float(numData)
            else:
                grad[tau] += math.exp(math.log(numer[tau])-math.log(denom)) / float(numData)
            if l2Reg:
                grad[tau] += (alpha/2.0 * lambdas[tau]) / float(numData)
    return grad, logSumExp

def funcEvalLambdasB(lambdas, dataPoints, data, numSamples, numData, alpha = 0.4, l2Reg = True):
    logSumExp = 0.0
    grad = {}
    numer = {}
    for tau in range(-37,38):
        grad[tau] = 0.0
        numer[tau] = 0.0

    for i in range(numSamples):
        denom = 0.0
        sumExp = 0.0
        for tau in range(-37,38):
            currx = data[i,tau]
            lmb = lambdas[tau]
            sumExp += math.exp(lmb*currx)
            numer[tau] = currx*math.exp(lmb*currx)
            denom += math.exp(lmb*currx)
        logSumExp += math.log(sumExp)
        for tau in range(-37,38):
            if numer[tau] == 0.0:
                if l2Reg:
                    grad[tau] += (alpha/2.0 * lambdas[tau]) / float(numData)
                continue
            if numer[tau] < 0.0:
                grad[tau] -= math.exp(math.log(-numer[tau])-math.log(denom)) / float(numData)
            else:
                grad[tau] += math.exp(math.log(numer[tau])-math.log(denom)) / float(numData)
            if l2Reg:
                grad[tau] += (alpha/2.0 * lambdas[tau]) / float(numData)
    return grad, logSumExp

def batchGradientAscentShiftPrior(options):
    """Run batch gradient ascent on the training data"""
    numThreads = min(mp.cpu_count() - 1, options.num_threads)
    (numData, data, sids) = genDideaTrainingData(options)
    optEval = 100.0
    optEvalPrev = float("-inf")
    thresh = options.thresh
    epoch = 0
    lrate = options.lrate # learning rate
    iters = 0

    if not options.cve:
        print "Optimized training for CVE0"
    parallel_partitions = []
    if options.num_threads > 1:
        # calculate indices to be evaluated in partition
        dataPoints = range(numData)
        random.shuffle(dataPoints)
        inc = int(numData / numThreads)
        l = 0
        r = inc
        for i in range(numThreads-1):
            parallel_partitions.append(dataPoints[l:r])
            l += inc
            r += inc
        parallel_partitions.append(dataPoints[l:])

    # initialize parameters
    lambdas = {}
    if not options.cve:
        for tau in range(-37,38):
            lambdas[tau] = 0.0
    else:
        lambdas = np.array([1.0 for _ in range(-37,38)])

    # take first step
    if not options.cve:
        # optimized: this is actually a CVE, but the structure of this CVE
        # is exploited to significantly optimize learning
        if options.num_threads > 1:
            grad, logSumExp = batchFuncEvalLambdas_mp(lambdas, options, numData, data, parallel_partitions, numThreads)
        else:
            grad, logSumExp = batchFuncEvalLambdas(lambdas, options, numData, data)
    else:
        grad, logSumExp = cve_general_ll(lambdas, options, numData, data)
    optEval = -logSumExp
    l2 = 0.0
    if not options.cve:
        for ind,tau in enumerate(range(-37,38)):
            lambdas[tau] -= lrate * grad[tau]
            l2 += grad[tau] * grad[tau]
        l2 = math.sqrt(l2)
    else:
        lambdas -= lrate * grad
        l2 = math.sqrt(np.sum([grad * grad]))

    # for ind,tau in enumerate(range(-37,38)):
    #     if not options.cve:
    #         lambdas[tau] -= lrate * grad[tau]
    #         l2 += grad[tau] * grad[tau]
    #     else:
    #         lambdas[ind] -= lrate * grad[ind]
    #         l2 += grad[ind] * grad[ind]
    # l2 = math.sqrt(l2)
    iters += 1
    print "iter %d: f(lmb*)/N = %f, norm(grad) = %f" % (iters, optEval, l2)

    while(l2 > thresh):
        if iters > options.maxIters:
            break
        if not options.cve:
            # optimized: this is actually a CVE, but the structure of this CVE
            # is exploited to significantly optimize learning
            if options.num_threads > 1:
                grad, logSumExp = batchFuncEvalLambdas_mp(lambdas, options, numData, data, parallel_partitions, numThreads)
                # grad, logSumExp = batchFuncEvalLambdas_mpB(lambdas, options, numData, data, numThreads, sids)
            else:
                grad, logSumExp = batchFuncEvalLambdas(lambdas, options, numData, data)
        else:
            grad, logSumExp = cve_general_ll(lambdas, options, numData, data)
        optEval = -logSumExp
        l2 = 0.0
        if not options.cve:
            for ind,tau in enumerate(range(-37,38)):
                lambdas[tau] -= lrate * grad[tau]
                l2 += grad[tau] * grad[tau]
            l2 = math.sqrt(l2)
        else:
            lambdas -= lrate * grad
            l2 = math.sqrt(np.sum([grad * grad]))
        # for ind, tau in enumerate(range(-37,38)):
        #     if not options.cve:
        #         lambdas[tau] -= lrate * grad[tau]
        #         l2 += grad[tau] * grad[tau]
        #     else:
        #         lambdas[ind] -= lrate * grad[ind]
        #         l2 += grad[ind] * grad[ind]
        # l2 = math.sqrt(l2)
        print "iter %d: f(lmb*)/N = %f, norm(grad) = %f" % (iters, optEval, l2)
        iters += 1

        optEval = -logSumExp
        # write output matrix
    fid = open(options.output, "w")
    if not options.cve:
        lambdas[0] = options.lmb0Prior
    else:
        # lambdas[37] = options.lmb0Prior
        lambdas[37] = np.mean(lambdas)
    for ind, tau in enumerate(range(-37,38)):
        if not options.cve:
            fid.write("%e\n" % lambdas[tau])
        else:
            fid.write("%e\n" % lambdas[ind])
    fid.close()

if __name__ == '__main__':
    # read in options and process input arguments
    args = parseInputOptions()
    process_args(args)
    batchGradientAscentShiftPrior(args)
        
    if stdo:
        stdo.close()
    if stde:
        stde.close()
