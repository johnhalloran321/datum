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

# General emission function, need not be normalized
from pyFiles.dideaEncoding import genVe
from pyFiles.spectrum import MS2Spectrum, MS2Iterator
from pyFiles.peptide import Peptide #, amino_acids_to_indices
from pyFiles.normalize import pipeline
from pyFiles.constants import allPeps, mass_h, mass_h2o, max_mz
from digest import (check_arg_trueFalse,
                        parse_var_mods, load_digested_peptides_var_mods)
from pyFiles.dideaEncoding import (histogram_spectra, simple_uniform_binwidth,
                                   round_op, 
                                   peptide_mod_offset_screen, peptide_var_mod_offset_screen,
                                   by_sepTauShift, by_tauShift)

from pyFiles.shard_spectra import (load_spectra,
                                   load_spectra_ret_dict,
                                   didea_candidate_generator,
                                   didea_candidate_binarydb_generator,
                                   pickle_candidate_spectra,
                                   pickle_candidate_binarydb_spectra,
                                   candidate_spectra_generator,
                                   candidate_spectra_memeffic_generator,
                                   candidate_binarydb_spectra_generator)
from pyFiles.peptide_db import PeptideDB, SimplePeptideDB

def didea_load_database(args, varMods, ntermVarMods, ctermVarMods):
    """ 
    """
    t = PeptideDB(load_digested_peptides_var_mods(os.path.join(args.digest_dir, 'targets.bin'), args.max_length, varMods, ntermVarMods, ctermVarMods))
    d = PeptideDB(load_digested_peptides_var_mods(os.path.join(args.digest_dir, 'decoys.bin'), args.max_length, varMods, ntermVarMods, ctermVarMods))

    return t,d

from subprocess import call, check_output

# stdo = open(os.devnull, "w")
# stde = sys.stdout

# set stdout and stderr for subprocess
# stdo = open(os.devnull, "w")

stdo = sys.stdout
stde = stdo

# stde = open('gmtk_err', "w")

# # set stdout for all default streams
# sys.stdout = open("dripSearch_output", "w")

class dideaPSM(object):
    """ Didea Peptide-Spectrum Match (PSM) class, similar to DRIP's PSM class
    """

    def __init__(self, sequence = '',
                 scan = -1, kind = 't', charge = 0,
                 # score = float("-inf"),
                 # foreground_score = float("-inf"), background_score = float("inf"),
                 flanking_nterm = '', flanking_cterm = '', var_mod_sequence = '',
                 protein = -1):
        # todo: decide whether we should just do away with scan being a member, but
        # do to the above this becomes tricky
        self.peptide = sequence
        self.spectrum = None
        self.scan = scan
        self.score = float("-inf")
        self.kind = kind
        self.charge = charge
        self.foreground_score = float("-inf")
        self.background_score = float("inf")
        self.flanking_nterm = flanking_nterm # used by dideaSearch
        self.flanking_cterm = flanking_cterm # used by dideaSearch
        self.var_mod_sequence = var_mod_sequence # currently only used by dideaSearch
        self.protein = protein

    def __cmp__(self,other):
        if self.scan == other.scan:
            if self.score < other.score:
                return -1
            else:
                return 1
        else:
            return 0

    def __hash__(self):
        return hash((self.scan, self.peptide))

    def __str__(self):
        return "scan%d-%s" % (self.scan, self.peptide)

    @property
    def length(self):
        """Number of amino acids in the peptide. Read-only computed property."""
        return len(self.peptide)

    def dideaScorePSM(self,bins,num_bins,lambdas2,lambdas3, useCve = True,
                      mods = {}, ntermMods = {}, ctermMods = {},
                      varMods = {}, varNtermMods = {}, varCtermMods = {}, varModSequence = [], 
                      bin_width = 1.):
        if useCve:
            (foregroundScore,backgroundScore) = dideaMultiChargeBinBufferLearnedLambdas(Peptide(self.peptide),
                                                                                        self.charge,bins,num_bins, lambdas2, lambdas3, 
                                                                                        mods, ntermMods, ctermMods, 
                                                                                        varMods, varNtermMods, varCtermMods,
                                                                                        varModSequence, bin_width)
        else:
            (foregroundScore,backgroundScore) = genVeBinBuffer(Peptide(self.peptide),
                                                               self.charge,bins,num_bins, lambdas2, lambdas3, 
                                                               mods, ntermMods, ctermMods, 
                                                               varMods, varNtermMods, varCtermMods,
                                                               varModSequence, bin_width)
        self.score = foregroundScore - backgroundScore
        self.foreground_score = foregroundScore
        self.background_score = backgroundScore

def copyArgs(argsA, argsB):
    """ Copy previously selected arguments to current parsed arguments
    """
    argsA.max_length = argsB.max_length
    argsA.min_length = argsB.min_length
    argsA.max_mass = argsB.max_mass
    argsA.mods_spec = argsB.mods_spec
    argsA.cterm_peptide_mods_spec = argsB.cterm_peptide_mods_spec
    argsA.nterm_peptide_mods_spec = argsB.nterm_peptide_mods_spec
    argsA.max_mods = argsB.max_mods
    argsA.min_mods = argsB.min_mods
    argsA.decoy_format = argsB.decoy_format
    argsA.keep_terminal_aminos = argsB.keep_terminal_aminos
    argsA.seed = argsB.seed
    argsA.enzyme = argsB.enzyme
    argsA.custom_enzyme = argsB.custom_enzyme
    argsA.missed_cleavages = argsB.missed_cleavages
    argsA.digestion = argsB.digestion
    argsA.peptide_buffer = argsB.peptide_buffer

    argsA.monoisotopic_precursor = argsB.monoisotopic_precursor
    argsA.precursor_window = argsB.precursor_window
    argsA.precursor_window_type = argsB.precursor_window_type
    argsA.scan_id_list = argsB.scan_id_list
    argsA.charges = argsB.charges
    # argsA.precursor_filter = argsB.precursor_filter
    argsA.decoys = argsB.decoys
    argsA.num_threads = argsB.num_threads
    argsA.top_match = argsB.top_match
    argsA.beam = argsB.beam
    argsA.num_jobs = argsB.num_jobs
    argsA.cluster_mode = argsB.cluster_mode
    argsA.write_cluster_scripts = argsB.write_cluster_scripts
    argsA.random_wait = argsB.random_wait
    argsA.min_spectrum_length = argsB.min_spectrum_length
    argsA.max_spectrum_length = argsB.max_spectrum_length
    argsA.shards = argsB.shards
    argsA.ppm = argsB.ppm
    argsA.max_obs_mass = argsB.max_obs_mass
    argsA.normalize = argsB.normalize
    argsA.cluster_dir = argsB.cluster_dir

    # check input arguments, create necessary directories, 
    # set necessary global variables based on input arguments
    # constant string
    if args.max_obs_mass < 0:
        print "Supplied max-mass %d, must be greater than 0, exitting" % args.max_obs_mass
        exit(-1)
    # check arguments
    if not args.output:
        print "No output file specified for PSMs, exitting"
        exit(8)

def parseInputOptions():
    parser = argparse.ArgumentParser(conflict_handler='resolve', 
                                     description="Run a DRIP database search given an MS2 file and the results of a FASTA file processed using digest.")
    ############## input and output options
    iFileGroup = parser.add_argument_group('iFileGroup', 'Necessary input files.')
    help_spectra = '<string> - The name of the file from which to parse fragmentation spectra, in ms2 format.'
    iFileGroup.add_argument('--spectra', type = str, action = 'store',
                            help = help_spectra)
    help_pepdb = '<string> - Output directory for digest.'
    iFileGroup.add_argument('--digest-dir', type = str, action = 'store',
                            help = help_pepdb, default = 'digest-output')
    ############## search parameters
    searchParamsGroup = parser.add_argument_group('searchParamsGroup', 'Search parameter options.')
    help_precursor_window = """<float> - Tolerance used for matching peptides to spectra.  Peptides must be within +/-'precursor-window' of the spectrum value. The precursor window units depend upon precursor-window-type. Default=3."""
    searchParamsGroup.add_argument('--precursor-window', type = float, action = 'store', default = 3.0, help = help_precursor_window)
    help_precursor_window_type = """<Da|ppm> - Specify the units for the window that is used to select peptides around the precursor mass location, either in Daltons (Da) or parts-per-million (ppm). Default=Da."""
    searchParamsGroup.add_argument('--precursor-window-type', type = str, action = 'store', default = 'Da', help = help_precursor_window_type)
    help_scan_id_list = """<string> - A file containing a list of scan IDs to search.  Default = <empty>."""
    help_cve = '<T|F> - Use convex virtual emission function for extensively (speed) optimized inference.'
    searchParamsGroup.add_argument('--cve', type = str, action = 'store', 
                                   default = 'true', help = help_cve)
    searchParamsGroup.add_argument('--scan-id-list', type = str, action = 'store', default = '', help = help_scan_id_list)
    help_charges = """<comma-separated-integers|all> - precursor charges to search. To specify individual charges, list as comma-separated, e.g., 1,2,3 to search all charge 1, 2, or 3 spectra. Default=All."""
    searchParamsGroup.add_argument('--charges', type = str, action = 'store', default = 'All', help = help_charges)
    help_high_res_ms2 = """<T|F> - boolean, whether the search is over high-res ms2 spectra. When this parameter is true, Didea checks whether bin-width < 1 and recalculates the number of bins used during scoring. Default=False."""
    searchParamsGroup.add_argument('--high-res-ms2', type = str, action = 'store', default = 'false', help = help_high_res_ms2)
    # help_precursor_filter = """<T|F> - boolean, when true, filter all peaks 1.5Da from the observed precursor mass. Default=False."""
    # searchParamsGroup.add_argument('--precursor-filter', type = str, action = 'store', default = 'false', help = help_precursor_filter)
    help_decoys = '<T|F> - whether to create (shuffle target peptides) and search decoy peptides. Default = True'
    searchParamsGroup.add_argument('--decoys', type = str, action = 'store', 
                                   default = 'True', help = help_decoys)
    help_num_threads = '<integer> - the number of threads to run on a multithreaded CPU. If supplied value is greater than number of supported threads, defaults to the maximum number of supported threads minus one. Multithreading is not suppored for cluster use as this is typically handled by the cluster job manager. Default=1.'
    searchParamsGroup.add_argument('--num-threads', type = int, action = 'store', 
                                   default = 1, help = help_num_threads)
    help_top_match = '<integer> - The number of psms per spectrum written to the output files. Default=1.'
    searchParamsGroup.add_argument('--top-match', type = int, action = 'store', 
                                   default = 1, help = help_top_match)
    help_num_bins = '<integer> - The number of bins to quantize observed spectrum. Default=2000.'
    searchParamsGroup.add_argument('--num-bins', type = int, action = 'store', 
                                   default = 2000, help = help_num_bins)
    help_bin_width = '<integer> - The bin width of m/z values. Default=1.0.'
    searchParamsGroup.add_argument('--bin-width', type = float, action = 'store', 
                                   default = 1.0, help = help_bin_width)
    ############## Cluster usage parameters
    clusterUsageGroup = parser.add_argument_group('clusterUsageGroup', 'Cluster data generation options.')
    help_num_cluster_jobs = '<integer> - the number of jobs to run in parallel. Default=1.'
    clusterUsageGroup.add_argument('--num-cluster-jobs', type = int, action = 'store', 
                                   dest = 'num_jobs',
                                   default = 1, help = help_num_cluster_jobs)
    help_cluster_mode = '<T|F> - evaluate dripSearch prepared data as jobs on a cluster.  Only set this to true once dripSearch has been run to prepare data for cluster use.  Default = False'
    clusterUsageGroup.add_argument('--cluster-mode', type = str, action = 'store', 
                                   default = 'False', help = help_cluster_mode)
    help_write_cluster_scripts = '<T|F> - write scripts to be submitted to cluster queue.  Only used when num-jobs > 1.  Job outputs will be written to log subdirectory in current directory. Default = True'
    clusterUsageGroup.add_argument('--write-cluster-scripts', type = str, action = 'store', 
                                   default = 'True', help = help_write_cluster_scripts)
    help_cluster_dir = '<string> - absolute path of directory to run cluster jobs. Default = /tmp'
    clusterUsageGroup.add_argument('--cluster-dir', type = str, action = 'store', 
                                   default = '/tmp', help = help_cluster_dir)
    help_random_wait = '<integer> - randomly wait up to specified number of seconds before accessing NFS. Default=10'
    clusterUsageGroup.add_argument('--random-wait', type = int, action = 'store', 
                                   default = 10, help = help_random_wait)
    help_merge_cluster_results = '<T|F> - merge dripSearch cluster results collected in local directory log.  Default = False'
    clusterUsageGroup.add_argument('--merge-cluster-results', type = str, action = 'store', 
                                   default = 'False', help = help_merge_cluster_results)
    parser.add_argument('--ppm', action = 'store_true',
                        default = False)
    ######### encoding options
    parser.add_argument('--normalize', dest = "normalize", type = str,
                        help = "Name of the spectrum preprocessing pipeline.", 
                        default = 'rank0')
    parser.add_argument('--lmb', dest = "lmb", type = float,
                        default = 0.5)
    parser.add_argument('--vekind', dest = "vekind", type = str,
                        help = "Emission function to use.", 
                        default = 'intensity')
    parser.add_argument('--learned-lambdas-ch2', action = 'store', dest = 'learned_lambdas_ch2',
                        type = str, help = 'Ch2 learned spectra-shift weights', default = "riptideLearnedCh2Lambdas.txt")
    parser.add_argument('--learned-lambdas-ch3', action = 'store', dest = 'learned_lambdas_ch3',
                        type = str, help = 'Ch3 learned spectra-shift weights', default = "riptideLearnedCh3Lambdas.txt")

    # output file options
    oFileGroup = parser.add_argument_group('oFileGroup', 'Output file options')
    oFileGroup.add_argument('--output', type = str,
                      help = 'identification file name')
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
    args.high_res_ms2 = check_arg_trueFalse(args.high_res_ms2)
    args.cve = check_arg_trueFalse(args.cve)
    args.cluster_mode = check_arg_trueFalse(args.cluster_mode)
    args.write_cluster_scripts = check_arg_trueFalse(args.write_cluster_scripts)
    args.merge_cluster_results = check_arg_trueFalse(args.merge_cluster_results)
    # args.precursor_filter = check_arg_trueFalse(args.precursor_filter)

    args.recalibrate = False
    if args.merge_cluster_results:
        chargeRecalibrate(args.logDir,
                          args.output + '.txt', 
                          args.top_match)
        exit(0)

    if args.cluster_mode:
        return 1

    # check input arguments, create necessary directories, 
    # set necessary global variables based on input arguments
    # constant string
    # check arguments
    assert(args.shards > 0)
    if not args.output:
        print "No output file specified for PSMs, exitting"
        exit(8)

    base = os.path.abspath(args.digest_dir)
    if not os.path.exists(base) and not args.cluster_mode:
        print "Digest directory %s does not exist." % (args.digest_dir)
        print "Please run digest first and specify the resulting directory with --digest-dir."
        exit(-1)
    else:
        args.digest_dir = os.path.abspath(args.digest_dir)

    # load digest options used to digest the fasta file
    ddo = pickle.load(open(os.path.join(args.digest_dir, 'options.pickle')))
    # set parameters to those used by digest
    args.max_length = ddo.max_length
    args.max_mass = ddo.max_mass
    args.min_length = ddo.min_length
    args.monoisotopic_precursor = ddo.monoisotopic_precursor
    args.mods_spec = ddo.mods_spec
    args.cterm_peptide_mods_spec = ddo.cterm_peptide_mods_spec
    args.nterm_peptide_mods_spec = ddo.nterm_peptide_mods_spec
    args.max_mods = ddo.max_mods
    args.min_mods = ddo.min_mods
    args.decoys = ddo.decoys
    args.decoy_format = ddo.decoy_format
    args.keep_terminal_aminos = ddo.keep_terminal_aminos
    args.seed = ddo.seed
    args.enzyme = ddo.enzyme
    args.custom_enzyme = ddo.custom_enzyme
    args.missed_cleavages = ddo.missed_cleavages
    args.digestion = ddo.digestion
    args.peptide_buffer = ddo.peptide_buffer

    # create obervation directories

    # check precursor mass type
    pmt = args.precursor_window_type.lower()
    if pmt == 'da':
        args.ppm = False
    else:
        args.ppm = True

    # make sure number of input threads does not exceed number of supported threads
    if args.num_threads > mp.cpu_count():
        args.num_threads = mp.cpu_count()
        # args.num_threads = max(mp.cpu_count()-1,1)

def load_lambdas(filename, tauMin = -37, tauMax = 37):
    with open(filename) as inputf:
        lambdas = [float(l) for l in inputf]
    learnedLambdas = {}
    for tau, l in zip(range(tauMin, tauMax + 1), lambdas):
        learnedLambdas[tau] = l

    return learnedLambdas

def byIonPairsTauShift(peptide, charge, lastBin = 1999, tauCard = 75,
                       mods = {}, ntermMods = {}, ctermMods = {}, 
                       bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') # , mass_op)
    nterm_fragments = []
    cterm_fragments = []
    byPairs = []
    ntermOffset = 0
    ctermOffset = 0

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    boffset = ntermOffset
    yoffset = ctermOffset

    for b,y,aaB,aaY in zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:])):
        by = []
        if aaB in mods:
            boffset += mods[aaB]
        if aaY in mods:
            yoffset += mods[aaY]
        for c in range(1,charge):
            cf = float(c)
            denom = cf * float(bin_width)
            c_boffset = cf*mass_h + boffset
            c_yoffset = mass_h2o + cf*mass_h + yoffset

            by.append( ( round_op(b+c_boffset,denom, tauCard, rMax), 
                         round_op(y+c_yoffset,denom, tauCard, rMax) ) )
        byPairs.append(by)

    return byPairs

def byIonSepTauShift_var_mods(peptide, charge, lastBin = 1999, tauCard = 75,
                              mods = {}, ntermMods = {}, ctermMods = {}, 
                              varMods = {}, ntermVarMods = {}, ctermVarMods = {},
                              varModSequence = [], bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') # , mass_op)
    nterm_fragments = []
    cterm_fragments = []
    ntermOffset = 0
    ctermOffset = 0

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    elif p[0] in ntermVarMods:
        if varModSequence[0] == '2': # denotes an nterm variable modification
            ntermOffset = ntermVarMods[p[0]][1]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]
    elif p[-1] in ctermVarMods:
        if varModSequence[-1] == '3': # denotes a cterm variable modification
            ctermOffset = ctermVarMods[p[-1]][1]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    bSeq = []
    ySeq = []
    boffset = ntermOffset
    yoffset = ctermOffset
    for ind, (b,y,aaB,aaY) in enumerate(zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:]))):
        bs = []
        ys = []
        if aaB in mods:
            boffset += mods[aaB]
        elif aaB in varMods:
            if varModSequence[ind]=='1':
                boffset += varMods[aaB][1]
        if aaY in mods:
            yoffset += mods[aaY]
        elif aaY in varMods:
            if varModSequence[ind+1]=='1':
                yoffset += varMods[aaY][1]

        for c in range(1,charge):
            cf = float(c)
            denom = float(c) * float(bin_width)
            c_boffset = cf*mass_h + boffset
            c_yoffset = mass_h2o + cf*mass_h + yoffset
            bs.append(round_op(b+c_boffset, denom, tauCard, rMax))
            ys.append(round_op(y+c_yoffset, denom, tauCard, rMax))
        bSeq.append(bs)
        ySeq.append(ys)
    return bSeq, ySeq

def byIonSepTauShift(peptide, charge, lastBin = 1999, tauCard = 75,
                     mods = {}, ntermMods = {}, ctermMods = {}, 
                     bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') # , mass_op)
    nterm_fragments = []
    cterm_fragments = []
    ntermOffset = 0
    ctermOffset = 0

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    bSeq = []
    ySeq = []
    boffset = ntermOffset
    yoffset = ctermOffset
    for b,y,aaB,aaY in zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:])):
        bs = []
        ys = []
        if aaB in mods:
            boffset += mods[aaB]
        if aaY in mods:
            yoffset += mods[aaY]

        for c in range(1,charge):
            cf = float(c)
            denom = float(c) * float(bin_width)
            c_boffset = cf*mass_h + boffset
            c_yoffset = mass_h2o + cf*mass_h + yoffset
            bs.append(round_op(b+c_boffset, denom, tauCard, rMax))
            ys.append(round_op(y+c_yoffset, denom, tauCard, rMax))
        bSeq.append(bs)
        ySeq.append(ys)
    return bSeq, ySeq


def byIonSepTauShift_flat(peptide, charge, lastBin = 1999, tauCard = 75,
                          mods = {}, ntermMods = {}, ctermMods = {}, 
                          bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') # , mass_op)
    nterm_fragments = []
    cterm_fragments = []
    ntermOffset = 0
    ctermOffset = 0

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    bSeq = []
    ySeq = []
    boffset = ntermOffset
    yoffset = ctermOffset
    for b,y,aaB,aaY in zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:])):
        if aaB in mods:
            boffset += mods[aaB]
        if aaY in mods:
            yoffset += mods[aaY]

        for c in range(1,charge):
            cf = float(c)
            denom = float(c) * float(bin_width)
            c_boffset = cf*mass_h + boffset
            c_yoffset = mass_h2o + cf*mass_h + yoffset
            bSeq.append(round_op(b+c_boffset, denom, tauCard, rMax))
            ySeq.append(round_op(y+c_yoffset, denom, tauCard, rMax))
    return bSeq, ySeq

def byIonsTauShift(peptide, charge, lastBin = 1999, tauCard = 75, 
                   mods = {}, ntermMods = {}, ctermMods = {},
                   bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') #, mass_op)
    nterm_fragments = []
    cterm_fragments = []
    ntermOffset = 0
    ctermOffset = 0

    # if peptide.seq == 'VAGFVTHLMK':
    #     print peptide.seq
    #     print peptide.mass
    #     print ntm
    #     print peptide.seq[:-1]
    #     print ctm
    #     print peptide.seq[1:]

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin

    # # define round operation
    # round_op = lambda ion, d: min(int(round(ion/d) + tauCard, rMax))

    # iterate through possible charges
    for c in range(1,charge):
        cf = float(c)
        denom = cf * float(bin_width)
        boffset = ntermOffset + cf*mass_h
        yoffset = ctermOffset + mass_h2o + cf*mass_h
        for b,y,aaB,aaY in zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:])):
            if aaB in mods:
                boffset += mods[aaB]
            if aaY in mods:
                yoffset += mods[aaY]
            # if peptide.seq == 'VAGFVTHLMK':
            #     print (b+boffset)/denom
            #     print (y+yoffset)/denom
            nterm_fragments.append(round_op(b+boffset,denom, tauCard, rMax))
            cterm_fragments.append(round_op(y+yoffset,denom, tauCard, rMax))
            # nterm_fragments.append(min(int(round((b+boffset)/denom)) + tauCard, rMax))
            # cterm_fragments.append(min(int(round((y+yoffset)/denom)) + tauCard, rMax))
    return (nterm_fragments,cterm_fragments)

def byIonPairsTauShift_var_mods(peptide, charge, lastBin = 1999, tauCard = 75,
                                mods = {}, ntermMods = {}, ctermMods = {}, 
                                varMods = {}, ntermVarMods = {}, ctermVarMods = {},
                                varModSequence = [], bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') # , mass_op)
    nterm_fragments = []
    cterm_fragments = []
    byPairs = []
    ntermOffset = 0
    ctermOffset = 0

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    elif p[0] in ntermVarMods:
        if varModSequence[0] == '2': # denotes an nterm variable modification
            ntermOffset = ntermVarMods[p[0]][1]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]
    elif p[-1] in ctermVarMods:
        if varModSequence[-1] == '3': # denotes a cterm variable modification
            ctermOffset = ctermVarMods[p[-1]][1]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    boffset = ntermOffset
    yoffset = ctermOffset
    for ind, (b,y,aaB,aaY) in enumerate(zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:]))):
        by = []
        if aaB in mods:
            boffset += mods[aaB]
        elif aaB in varMods:
            if varModSequence[ind]=='1':
                boffset += varMods[aaB][1]
        if aaY in mods:
            yoffset += mods[aaY]
        elif aaY in varMods:
            if varModSequence[ind+1]=='1':
                yoffset += varMods[aaY][1]

        for c in range(1,charge):
            cf = float(c)
            denom = cf * float(bin_width)
            c_boffset = cf*mass_h + boffset
            c_yoffset = mass_h2o + cf*mass_h + yoffset
            by.append( (round_op(b+c_boffset, denom, tauCard, rMax), 
                        round_op(y+c_yoffset, denom, tauCard, rMax) ) )
            # by.append( (min(int(round((b+c_boffset)/denom)) + tauCard, rMax), 
            #             min(int(round((y+c_yoffset)/denom)) + tauCard, rMax) ) )
        byPairs.append(by)

    return byPairs

def byIonsTauShift_var_mods(peptide, charge, lastBin = 1999, tauCard = 75, 
                            mods = {}, ntermMods = {}, ctermMods = {},
                            varMods = {}, ntermVarMods = {}, ctermVarMods = {},
                            varModSequence = [], bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    mass_op = lambda m: int(math.floor(m))
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic') # , mass_op)
    nterm_fragments = []
    cterm_fragments = []
    ntermOffset = 0
    ctermOffset = 0

    p = peptide.seq
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        ntermOffset = ntermMods[p[0]]
    elif p[0] in ntermVarMods:
        if varModSequence[0] == '2': # denotes an nterm variable modification
            ntermOffset = ntermVarMods[p[0]][1]
    if p[-1] in ctermMods:
        ctermOffset = ctermMods[p[-1]]
    elif p[-1] in ctermVarMods:
        if varModSequence[-1] == '3': # denotes a cterm variable modification
            ctermOffset = ctermVarMods[p[-1]][1]

    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    for c in range(1,charge):
        cf = float(c)
        denom = cf * float(bin_width)
        boffset = ntermOffset + cf*mass_h
        yoffset = ctermOffset + mass_h2o + cf*mass_h
        for ind, (b,y,aaB,aaY) in enumerate(zip(ntm[1:-1], reversed(ctm[1:-1]), peptide.seq[:-1], reversed(peptide.seq[1:]))):
            if aaB in mods:
                boffset += mods[aaB]
            elif aaB in varMods:
                if varModSequence[ind]=='1':
                    boffset += varMods[aaB][1]

            if aaY in mods:
                yoffset += mods[aaY]
            elif aaY in varMods:
                if varModSequence[ind+1]=='1':
                    yoffset += varMods[aaY][1]

            nterm_fragments.append(round_op(b+boffset, denom, tauCard, rMax))
            cterm_fragments.append(round_op(y+yoffset, denom, tauCard, rMax))
    return (nterm_fragments,cterm_fragments)

    

def dideaMultiChargeBinBufferLearnedLambdas(peptide, charge, bins, num_bins, learnedLambdas2, learnedLambdas3, 
                                            mods = {}, ntermMods = {}, ctermMods = {},
                                            varMods = {}, ntermVarMods = {}, ctermVarMods = {}, varModSequence = [], 
                                            bin_width = 1.):
    """ Calculate the posterior(\tau_0 = 0 | s, x), where \tau_0 
    the prologue shift variable, s is the observed spectrum, and x is the candidate peptide.
    """
    lastBin = num_bins-1
    tauCard = 75
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic')
    if varMods or ntermVarMods or ctermVarMods:
        b_offsets, y_offsets = peptide_var_mod_offset_screen(peptide.seq, mods, ntermMods, ctermMods, 
                                                             varMods, ntermVarMods, ctermVarMods,varModSequence)
    else:
        b_offsets, y_offsets = peptide_mod_offset_screen(peptide.seq, mods, ntermMods, ctermMods)

    sB, sY = by_tauShift(ntm, ctm, 2, b_offsets, y_offsets, 
                         lastBin, tauCard, bin_width)
    # vectors containing sets of charged ions among a b- and y-ion pair, 
    # necessary when marginalizing over charge in the Diea model
    bSeq, ySeq = by_sepTauShift(ntm, ctm, 3, b_offsets, y_offsets, 
                                lastBin, tauCard, bin_width)

    sB += sY # collapse b- and y- charge2 vectors together

    bSeq = np.array(bSeq).astype(int)
    ySeq = np.array(ySeq).astype(int)
    sB = np.array(sB).astype(int)

    # # if peptide.seq == 'SILITTIENALDNEEFESHDK':
    # if peptide.seq == 'VAGFVTHLMK':
    #     print peptide.seq
    #     print b_offsets
    #     print y_offsets
    #     print ntm
    #     print ctm
    #     print bSeq - 75
    #     print ySeq - 75
    #     print sorted(sB)
    # #     # np.set_printoptions(threshold=sys.maxsize)
    # #     # print bins
    #     print sum([1 for i in sB if bins[i] > 0.])
    #     print sum([bins[i] for i in sB])
    # #     # print sum(bins)
    #     for i in sB:
    #         print i-75, bins[i-1], bins[i], bins[i+1]
    #     for i in bSeq:
    #         print i-75, bins[i]
    #     for i in ySeq:
    #         print i-75, bins[i]

    backgroundScore = 0.0
    cLogProb = math.log(2.0)

    # first calculate foreground score
    l = learnedLambdas2[0]
    foregroundScore = math.exp(l * np.sum(bins[sB]))
    a = foregroundScore
    currScore = 0.0
    l = learnedLambdas3[0]
    h = np.exp(l *(bins[bSeq] + bins[ySeq]) - cLogProb)
    foregroundScore += math.exp(np.sum(np.log(np.sum(h, axis=1))))

    backgroundScore += foregroundScore
    foregroundScore = math.log(foregroundScore)

    l = np.array([learnedLambdas2[tau] for tau in range(-37,38)])
    l2 = np.array([learnedLambdas3[tau] for tau in range(-37,38)])

    backgroundScore += np.sum(np.exp(l*np.sum(bins[sB + [[tau] for tau in range(-37,38)]], axis=1))) - a
    # h = np.exp(l2*(bins[bSeq + [[tau] for tau in range(-37,38)]] + bins[ySeq + [[tau] for tau in range(-37,38)]])) - cLogProb
    # backgroundScore += math.exp(np.sum(np.log(np.sum(h,axi=1)))
    # next, background score, eliminating iterating over \tau=0
    for tau in range(-37,0):
        l = learnedLambdas3[tau]
        l2 = learnedLambdas3[-tau]
        # marginalize over charge variable
        h = np.exp(l * (bins[bSeq+tau] + bins[ySeq+tau]) - cLogProb)
        h2 = np.exp(l2 * (bins[bSeq-tau] + bins[ySeq-tau]) - cLogProb)
        currScore = math.exp(np.sum(np.log(np.sum(h, axis=1))))
        currScore2 = math.exp(np.sum(np.log(np.sum(h2, axis=1))))
        backgroundScore += currScore + currScore2
    # final background score is log \sum_{\tau} (B + Y)^T S_{\tau}
    backgroundScore = math.log(backgroundScore)

    return (foregroundScore,backgroundScore)

def dideaMultiChargeBinBufferLearnedLambdas_og(peptide, charge, bins, num_bins, learnedLambdas2, learnedLambdas3, 
                                               mods = {}, ntermMods = {}, ctermMods = {},
                                               varMods = {}, ntermVarMods = {}, ctermVarMods = {}, varModSequence = []):
    """ Calculate the posterior(\tau_0 = 0 | s, x), where \tau_0 
    the prologue shift variable, s is the observed spectrum, and x is the candidate peptide.
    """
    lastBin = num_bins-1
    tauCard = 75
    if varMods or ntermVarMods or ctermVarMods:
        byPairs = byIonPairsTauShift_var_mods(peptide,3, lastBin, tauCard, varModSequence)
        sB, sY = byIonsTauShift_var_mods(peptide,2, lastBin, tauCard, varModSequence)
    else:    
        byPairs = byIonPairsTauShift(peptide,3, lastBin, tauCard, 
                                     mods, ntermMods, ctermMods)
        sB, sY = byIonsTauShift(peptide,2, lastBin, tauCard,
                                mods, ntermMods, ctermMods)
    foregroundScore = 0.0
    backgroundScore = 0.0
    # first calculate foreground score

    l = learnedLambdas2[0]
    h = 0.0
    for b,y in zip(sB, sY):
        h += bins[b] + bins[y]
    foregroundScore = math.exp(l * h)
    currScore = 0.0
    l = learnedLambdas3[0]
    for by in byPairs:
        h = 0.0
        for b, y in by:
            h += math.exp(l*(bins[b] + bins[y])) / 2.0
        currScore += math.log(h)
    foregroundScore += math.exp(currScore)

    backgroundScore += foregroundScore
    foregroundScore = math.log(foregroundScore)
    # next, background score, eliminating iterating over \tau=0

    for tau in range(-37,0):
        l = learnedLambdas2[tau]
        l2 = learnedLambdas2[-tau]
        h = 0.0
        h2 = 0.0
        for b,y in zip(sB, sY):
            h += bins[b+tau] + bins[y+tau]
            h2 += bins[b-tau] + bins[y-tau]
        backgroundScore += math.exp(l * h)
        backgroundScore += math.exp(l2 * h2)
        currScore = 0.0
        currScore2 = 0.0
        l = learnedLambdas3[tau]
        l2 = learnedLambdas3[-tau]
        for by in byPairs:
            h = 0.0
            h2 = 0.0
            for b, y in by:
                h += math.exp(l * (bins[b+tau] + bins[y+tau])) / 2.0
                h2 += math.exp(l2 * (bins[b-tau] + bins[y-tau])) / 2.0
            currScore += math.log(h)
            currScore2 += math.log(h2)
        backgroundScore += math.exp(currScore) + math.exp(currScore2)
    # final background score is log \sum_{\tau} (B + Y)^T S_{\tau}
    backgroundScore = math.log(backgroundScore)

    return (foregroundScore,backgroundScore)

def logGenVe(theta, s):
    # log of the CVE defined in dideaTrain
    return np.log(genVe(theta,s))

def genVeBinBuffer(peptide, charge, bins, num_bins, learnedLambdas2, learnedLambdas3, 
                   mods = {}, ntermMods = {}, ctermMods = {},
                   varMods = {}, ntermVarMods = {}, ctermVarMods = {}, varModSequence = [], 
                   bin_width = 1.):
    """ For a generally defined (virtual evidence) emission function,
    calculate the posterior(\tau_0 = 0 | s, x), where \tau_0 
    the prologue shift variable, s is the observed spectrum, and x is the candidate peptide.
    """
    lastBin = num_bins-1
    tauCard = 75
    (ntm, ctm) = peptide.ideal_fragment_masses('monoisotopic')
    if varMods or ntermVarMods or ctermVarMods:
        b_offsets, y_offsets = peptide_var_mod_offset_screen(peptide.seq, mods, ntermMods, ctermMods, 
                                                             varMods, ntermVarMods, ctermVarMods,varModSequence)
    else:
        b_offsets, y_offsets = peptide_mod_offset_screen(peptide.seq, mods, ntermMods, ctermMods)

    sB, sY = by_tauShift(ntm, ctm, 2, b_offsets, y_offsets, 
                         lastBin, tauCard, bin_width)
    # vectors containing sets of charged ions among a b- and y-ion pair, 
    # necessary when marginalizing over charge in the Diea model
    bSeq, ySeq = by_sepTauShift(ntm, ctm, 3, b_offsets, y_offsets, 
                                lastBin, tauCard, bin_width)

    # enable fancy indexing for the various lists of b-y ions
    bSeq = np.array(bSeq).astype(int)
    ySeq = np.array(ySeq).astype(int)
    sB = np.array(sB).astype(int)
    sY = np.array(sY).astype(int)

    foregroundScore = 0.0
    backgroundScore = 0.0
    # first calculate foreground score
    cLogProb = math.log(2.0)

    l = learnedLambdas2[0]
    foregroundScore = math.exp(np.sum(logGenVe(l,bins[sB]) + logGenVe(l,bins[sY])))

    l = learnedLambdas3[0]
    h = np.exp(logGenVe(l,bins[bSeq]) + logGenVe(l,bins[ySeq]) - cLogProb)
    foregroundScore += math.exp(np.sum(np.log(np.sum(h, axis=1))))

    backgroundScore += foregroundScore
    foregroundScore = math.log(foregroundScore)

    # next, background score, eliminating iterating over \tau=0
    for tau in range(-37,0):
        l = learnedLambdas2[tau]
        l2 = learnedLambdas2[-tau]
        h = np.exp(np.sum(logGenVe(l,bins[sB+tau]) + logGenVe(l,bins[sY+tau])))
        h2 = np.exp(np.sum(logGenVe(l2,bins[sB-tau]) + logGenVe(l2,bins[sY-tau])))
        backgroundScore += h + h2

        l = learnedLambdas3[tau]
        l2 = learnedLambdas3[-tau]

        h = np.exp(logGenVe(l,bins[bSeq+tau]) + logGenVe(l,bins[ySeq+tau]) - cLogProb)
        h2 = np.exp(logGenVe(l2,bins[bSeq-tau]) + logGenVe(l2,bins[ySeq-tau]) - cLogProb)
        currScore = math.exp(np.sum(np.log(np.sum(h, axis=1))))
        currScore2 = math.exp(np.sum(np.log(np.sum(h2, axis=1))))

        backgroundScore += currScore + currScore2
    # final background score is log \sum_{\tau} (B + Y)^T S_{\tau}
    backgroundScore = math.log(backgroundScore)

    return (foregroundScore,backgroundScore)

def genVeBinBuffer_bag(peptide, charge, bins, num_bins, learnedLambdas2, learnedLambdas3, 
                       mods = {}, ntermMods = {}, ctermMods = {},
                       varMods = {}, ntermVarMods = {}, ctermVarMods = {}):
    """ Calculate the posterior(\tau_0 = 0 | s, x), where \tau_0 
    the prologue shift variable, s is the observed spectrum, and x is the candidate peptide.
    """
    lastBin = num_bins-1
    tauCard = 75
    # if varMods or ntermVarMods or ctermVarMods:
    #     byPairs = byIonPairsTauShift_var_mods(peptide,3, lastBin, tauCard)
    #     sB, sY = byIonsTauShift_var_mods(peptide,2, lastBin, tauCard)
    # else:    
    #     # byPairs = byIonPairsTauShift(peptide,3, lastBin, tauCard, 
    #     #                              mods, ntermMods, ctermMods)
    #     bSeq, ySeq = byIonSepTauShift(peptide,3, lastBin, tauCard, 
    #                                  mods, ntermMods, ctermMods)
    #     sB, sY = byIonsTauShift(peptide,2, lastBin, tauCard,
    #                             mods, ntermMods, ctermMods)

    byPairs = byIonPairsTauShift(peptide,3, lastBin, tauCard, 
                                 mods, ntermMods, ctermMods)
    bSeq, ySeq = byIonSepTauShift(peptide,3, lastBin, tauCard, 
                                  mods, ntermMods, ctermMods)
    sB, sY = byIonsTauShift(peptide,2, lastBin, tauCard,
                            mods, ntermMods, ctermMods)

    bSeq = np.array(bSeq).astype(int)
    ySeq = np.array(ySeq).astype(int)
    sB = np.array(sB).astype(int)
    sY = np.array(sY).astype(int)

    foregroundScore = 0.0
    backgroundScore = 0.0
    # first calculate foreground score
    cLogProb = math.log(2.0)

    l = learnedLambdas2[0]
    foregroundScore = np.exp(np.sum(logGenVe(l,bins[sB]) + logGenVe(l,bins[sY])))
    currScore = 0.0
    l = learnedLambdas3[0]
    h = np.exp(logGenVe(l,bins[bSeq]) + logGenVe(l,bins[ySeq]) - cLogProb)
    currScore = np.sum(np.log(np.sum(h, axis=0)))
    # for by in byPairs:
    #     h = 0.0
    #     for b, y in by:
    #         h += math.exp(logGenVe(l,bins[b]) + logGenVe(l,bins[y]) - cLogProb)
    #     currScore += math.log(h)
    foregroundScore += math.exp(currScore)

    backgroundScore += foregroundScore
    foregroundScore = math.log(foregroundScore)
    # next, background score, eliminating iterating over \tau=0

    for tau in range(-37,0):
        l = learnedLambdas2[tau]
        l2 = learnedLambdas2[-tau]
        # h = np.exp(np.sum(logGenVe(l,bins[sB+tau]) + logGenVe(l,bins[sY+tau])))
        # h2 = np.exp(np.sum(logGenVe(l2,bins[sB-tau]) + logGenVe(l2,bins[sY-tau])))
        h = 0.0
        h2 = 0.0
        for b,y in zip(sB, sY):
            h += logGenVe(l,bins[b+tau]) + logGenVe(l,bins[y+tau])
            h2 += logGenVe(l2,bins[b-tau]) + logGenVe(l2,bins[y-tau])
        backgroundScore += math.exp(h) + math.exp(h2)
        currScore = 0.0
        currScore2 = 0.0
        l = learnedLambdas3[tau]
        l2 = learnedLambdas3[-tau]
        for by in byPairs:
            h = 0.0
            h2 = 0.0
            for b, y in by:
                # h += math.exp(l * (bins[b+tau] + bins[y+tau])) / 2.0
                # h2 += math.exp(l2 * (bins[b-tau] + bins[y-tau])) / 2.0
                a = logGenVe(l,bins[b+tau]) + logGenVe(l,bins[y+tau]) - cLogProb
                b = logGenVe(l,bins[b-tau]) + logGenVe(l,bins[y-tau]) - cLogProb
                h += math.exp(a)
                h2 += math.exp(b)
            currScore += math.log(h)
            currScore2 += math.log(h2)
        backgroundScore += math.exp(currScore) + math.exp(currScore2)
    # final background score is log \sum_{\tau} (B + Y)^T S_{\tau}
    backgroundScore = math.log(backgroundScore)

    return (foregroundScore,backgroundScore)

def genVeBinBuffer_og(peptide, charge, bins, num_bins, learnedLambdas2, learnedLambdas3, 
                      mods = {}, ntermMods = {}, ctermMods = {},
                      varMods = {}, ntermVarMods = {}, ctermVarMods = {}):
    """ Calculate the posterior(\tau_0 = 0 | s, x), where \tau_0 
    the prologue shift variable, s is the observed spectrum, and x is the candidate peptide.
    """
    lastBin = num_bins-1
    tauCard = 75
    if varMods or ntermVarMods or ctermVarMods:
        byPairs = byIonPairsTauShift_var_mods(peptide,3, lastBin, tauCard)
        sB, sY = byIonsTauShift_var_mods(peptide,2, lastBin, tauCard)
    else:    
        byPairs = byIonPairsTauShift(peptide,3, lastBin, tauCard, 
                                     mods, ntermMods, ctermMods)
        sB, sY = byIonsTauShift(peptide,2, lastBin, tauCard,
                                mods, ntermMods, ctermMods)

    foregroundScore = 0.0
    backgroundScore = 0.0
    # first calculate foreground score
    cLogProb = math.log(2.0)

    l = learnedLambdas2[0]
    h = 0.0
    for b,y in zip(sB, sY):
        h += logGenVe(l,bins[b]) + logGenVe(l,bins[y])
    foregroundScore = math.exp(h)
    currScore = 0.0
    l = learnedLambdas3[0]
    for by in byPairs:
        h = 0.0
        for b, y in by:
            h += math.exp(logGenVe(l,bins[b]) + logGenVe(l,bins[y]) - cLogProb)
        currScore += math.log(h)
    foregroundScore += math.exp(currScore)

    backgroundScore += foregroundScore
    foregroundScore = math.log(foregroundScore)
    # next, background score, eliminating iterating over \tau=0

    for tau in range(-37,0):
        l = learnedLambdas2[tau]
        l2 = learnedLambdas2[-tau]
        h = 0.0
        h2 = 0.0
        for b,y in zip(sB, sY):
            h += logGenVe(l,bins[b+tau]) + logGenVe(l,bins[y+tau])
            h2 += logGenVe(l2,bins[b-tau]) + logGenVe(l2,bins[y-tau])
        backgroundScore += math.exp(h)
        backgroundScore += math.exp(h2)
        currScore = 0.0
        currScore2 = 0.0
        l = learnedLambdas3[tau]
        l2 = learnedLambdas3[-tau]
        for by in byPairs:
            h = 0.0
            h2 = 0.0
            for b, y in by:
                # h += math.exp(l * (bins[b+tau] + bins[y+tau])) / 2.0
                # h2 += math.exp(l2 * (bins[b-tau] + bins[y-tau])) / 2.0
                a = logGenVe(l,bins[b+tau]) + logGenVe(l,bins[y+tau]) - cLogProb
                b = logGenVe(l,bins[b-tau]) + logGenVe(l,bins[y-tau]) - cLogProb
                h += math.exp(a)
                h2 += math.exp(b)
            currScore += math.log(h)
            currScore2 += math.log(h2)
        backgroundScore += math.exp(currScore) + math.exp(currScore2)
    # final background score is log \sum_{\tau} (B + Y)^T S_{\tau}
    backgroundScore = math.log(backgroundScore)

    return (foregroundScore,backgroundScore)

def write_dideaPSM_to_ident_var_mods(fid, psm, mods, nterm_mods, cterm_mods,
                                     var_mods, nterm_var_mods, cterm_var_mods, 
                                     isVarMods = 0):
    """header as of 10/4/2018:(1)Kind
                              (2)Scan
                              (3)Score
                              (4)Peptide
                              (5)Foreground_score
                              (6)Background_score
                              (7)Charge
                              (8)Flanking_nterm
                              (9)Flanking_ncterm
                              (10)Protein_id
                              (11)Var_mod_seq (only apppears if variable mods selected)
                    
    """
    try:
        var_mod_string = psm.var_mod_sequence.split('\x00')[0]

        c = psm.peptide[0]
        vm = psm.var_mod_sequence[0]
        pep_str = [c]

        # check n-term and c-term
        if c in nterm_mods:
            pep_str += str('[%1.0e]' % nterm_mods[c])
        elif vm == '2':
            pep_str += str('[%1.0e]' % nterm_var_mods[c][1])
        elif c in mods:
            pep_str += str('[%1.0e]' % mods[c])

        for c, vm in zip(psm.peptide[1:-1], psm.var_mod_sequence[1:-1]):
            pep_str += c
            if c in mods:
                pep_str += str('[%1.0e]' % mods[c])
            elif vm == '1': # variable mods
                pep_str += str('[%1.0e]' % var_mods[c][1])

        c = psm.peptide[-1]
        vm = psm.var_mod_sequence[-1]
        pep_str += [c]
        # check n-term and c-term
        if c in cterm_mods:
            pep_str += str('[%1.0e]' % cterm_mods[c])
        elif vm == '3':
            pep_str += str('[%1.0e]' % cterm_var_mods[c][1])
        elif c in mods:
            pep_str += str('[%1.0e]' % mods[c])

        if not isVarMods:
            fid.write('%c\t%d\t%f\t%s\t%f\t%f\t%d\t%c\t%c\t%s\n' % (psm.kind,
                                                                    psm.scan,
                                                                    psm.score,
                                                                    ''.join(pep_str),
                                                                    psm.foreground_score,
                                                                    psm.background_score,
                                                                    psm.charge,
                                                                    psm.flanking_nterm,
                                                                    psm.flanking_cterm,
                                                                    psm.kind + str(psm.protein)))
        else:
            fid.write('%c\t%d\t%f\t%s\t%f\t%f\t%d\t%c\t%c\t%s\t%s\n' % (psm.kind, 
                                                                        psm.scan,
                                                                        psm.score,
                                                                        ''.join(pep_str),
                                                                        psm.foreground_score,
                                                                        psm.background_score,
                                                                        psm.charge,
                                                                        psm.flanking_nterm,
                                                                        psm.flanking_cterm,
                                                                        psm.kind + str(psm.protein),
                                                                        var_mod_string))

    except IOError:
        print "Could not write to ident stream, exitting"
        exit(-1)

    return 1

def write_dideaSearch_ident(output, scored_psms,
                            mods, nterm_mods, cterm_mods,
                            var_mods, nterm_var_mods, cterm_var_mods):
    try:
        identFid = open(output, "w")
    except IOError:
        print "Could not open file %s for writing, exitting" % output

    isVarMods = len(var_mods) + len(nterm_var_mods) + len(cterm_var_mods)

    if isVarMods:
        identFid.write('Kind\tScan\tScore\tPeptide\tForeground_score\tBackground_score\tCharge\tFlanking_nterm\tFlanking_cterm\tProtein_id\tVar_mod_seq\n')
    else:
        identFid.write('Kind\tScan\tScore\tPeptide\tForeground_score\tBackground_score\tCharge\tFlanking_nterm\tFlanking_cterm\tProtein_id\n')

    for td in scored_psms:
        # write target
        if td[1]:
            write_dideaPSM_to_ident_var_mods(identFid, td[1],
                                             mods, nterm_mods, cterm_mods,
                                             var_mods, nterm_var_mods, cterm_var_mods,
                                             isVarMods)
        # now decoy
        if td[2]:
            write_dideaPSM_to_ident_var_mods(identFid, td[2],
                                             mods, nterm_mods, cterm_mods,
                                             var_mods, nterm_var_mods, cterm_var_mods,
                                             isVarMods)
    identFid.close()
    return 1

def score_didea_spectra(args, data, ranges,
                        preprocess, learnedLambdas2, learnedLambdas3,
                        mods, ntermMods, ctermMods,
                        varMods, ntermVarMods, ctermVarMods):
    """
    """

    scoref = lambda r: r.score
    top_psms = []
    nb = args.num_bins

    # Load the pickled representation of this shard's data
    # prune multiples
    visited_spectra = set([])
    spectra = []
    # spectra = data['spectra']
    for s in data['spectra']:
        if s.spectrum_id not in visited_spectra:
            spectra.append(s)
            visited_spectra.add(s.spectrum_id)
        
    target = data['target']
    decoy = data['decoy']

    nb = args.num_bins

    lastBin = nb - 1
    tauCard = 75
    # calculate tau-radius around bin indices
    rMin = -tauCard
    rMax = lastBin + tauCard

    bins2 = np.empty( (nb + 2 * tauCard)  )
    validcharges = args.charges

    for s in spectra:
        preprocess(s)
        # Generate the spectrum observations.
        bins = histogram_spectra(s, ranges, max)
        bins[0] = 0.0
        bins[-1] = 0.0
        # truebins = histogram_spectra(s, ranges, max)
        # bins = bins_to_vecpt_ratios(truebins, args.vekind, args.lmb)

        for i in range(rMin, rMax):
            a = bins[min(max(i,0), lastBin)]
            bins2[i+tauCard] = a

        sid = s.spectrum_id

        charged_target_psms = []
        charged_decoy_psms = []

        for charge in validcharges:
            if (s.spectrum_id, charge) not in target:
                continue
            # score PSMs
            # serialized data from digest database:
            #    peptide[0] = peptide mass (float)
            #    peptide[1] = peptide string (string of max_length character, possibly many of which are null)
            #    peptide[2] = protein name (mapped to an integer for the protein value encountered in the file)
            #    peptide[3] = nterm_flanking (character)
            #    peptide[4] = cterm_flanking (character)
            #    peptide[5] = binary string deoting variable modifications
            for tp in target[(s.spectrum_id,charge)]:
                t = tp[1].split('\x00')[0]
                pepType = 1
                if varMods or ntermVarMods or ctermVarMods:
                    varModSequence = tp[5][:len(t)]
                    theoSpecKey = t + varModSequence
                else:
                    varModSequence = ''.join(['0'] * len(t))
                    theoSpecKey = t

                curr_psm = dideaPSM(t, sid, 't', charge, 
                                    tp[3], tp[4], varModSequence, tp[2])
                curr_psm.dideaScorePSM(bins2, args.num_bins,learnedLambdas2, learnedLambdas3, args.cve,
                                       mods, ntermMods, ctermMods,
                                       varMods, ntermVarMods, ctermVarMods, 
                                       varModSequence, args.bin_width)
                charged_target_psms.append(curr_psm)

            for dp in decoy[(s.spectrum_id,charge)]:
                d = dp[1].split('\x00')[0]
                pepType = 2
                if varMods or ntermVarMods or ctermVarMods:
                    varModSequence = dp[5][:len(d)]
                    theoSpecKey = d + varModSequence
                else:
                    varModSequence = ''.join(['0'] * len(d))
                    theoSpecKey = d
                curr_psm = dideaPSM(d, sid, 'd', charge, 
                                    dp[3], dp[4], varModSequence, dp[2])
                curr_psm.dideaScorePSM(bins2, args.num_bins,learnedLambdas2, learnedLambdas3, args.cve,
                                       mods, ntermMods, ctermMods,
                                       varMods, ntermVarMods, ctermVarMods, 
                                       varModSequence, args.bin_width)
                charged_decoy_psms.append(curr_psm)

        top_target = max(charged_target_psms,key = scoref)
        top_decoy = max(charged_decoy_psms,key = scoref)

        top_psms.append((sid, top_target, top_decoy))
    return top_psms

def score_didea_spectra_incore(args, spec, targets, decoys,
                               ranges,preprocess, learnedLambdas2, learnedLambdas3,
                               mods, ntermMods, ctermMods,
                               varMods, ntermVarMods, ctermVarMods):
    """Score the spectra
    """
    scoref = lambda r: r.score
    top_psms = []
    nb = args.num_bins

    # Load the pickled representation of this shard's data
    # prune multiples
    visited_spectra = set([])
    spectra = []
    for s in spec:
        if s.spectrum_id not in visited_spectra:
            spectra.append(s)
            visited_spectra.add(s.spectrum_id)
        
    nb = args.num_bins

    lastBin = nb - 1
    tauCard = 75
    # calculate tau-radius around bin indices
    rMin = -tauCard
    rMax = lastBin + tauCard

    bins2 = np.empty( (nb + 2 * tauCard) )
    validcharges = args.charges

    for s in spectra:
        preprocess(s)
        # Generate the spectrum observations.
        bins = histogram_spectra(s, ranges, max)
        bins[0] = 0.0
        bins[-1] = 0.0

        for i in range(rMin, rMax):
            a = bins[min(max(i,0), lastBin)]
            bins2[i+tauCard] = a

        sid = s.spectrum_id

        charged_target_psms = []
        charged_decoy_psms = []

        for charge, m in s.charge_lines:
            if charge not in validcharges:
                continue
            # score PSMs
            # serialized data from digest database:
            #    peptide[0] = peptide mass (float)
            #    peptide[1] = peptide string
            #    peptide[2] = protein name (mapped to an integer for the protein value encountered in the file)
            #    peptide[3] = nterm_flanking (character)
            #    peptide[4] = cterm_flanking (character)
            #    peptide[5] = binary string deoting variable modifications
            for tp in (targets.filter(m - mass_h, args.precursor_window, args.ppm)):
                t = tp[1]
                pepType = 1
                if varMods or ntermVarMods or ctermVarMods:
                    varModSequence = tp[5]
                    theoSpecKey = t + varModSequence
                else:
                    varModSequence = ''.join(['0'] * len(t))
                    theoSpecKey = t

                curr_psm = dideaPSM(t, sid, 't', charge, 
                                    tp[3], tp[4], varModSequence, tp[2])
                curr_psm.dideaScorePSM(bins2, args.num_bins,learnedLambdas2, learnedLambdas3, args.cve,
                                       mods, ntermMods, ctermMods,
                                       varMods, ntermVarMods, ctermVarMods, 
                                       varModSequence, args.bin_width)
                charged_target_psms.append(curr_psm)

            for dp in (decoys.filter(m - mass_h, args.precursor_window, args.ppm)):
                d = dp[1]
                pepType = 2
                if varMods or ntermVarMods or ctermVarMods:
                    varModSequence = dp[5]
                    theoSpecKey = d + varModSequence
                else:
                    varModSequence = ''.join(['0'] * len(d))
                    theoSpecKey = d
                curr_psm = dideaPSM(d, sid, 'd', charge, 
                                    dp[3], dp[4], varModSequence, dp[2])
                curr_psm.dideaScorePSM(bins2, args.num_bins,learnedLambdas2, learnedLambdas3, args.cve,
                                       mods, ntermMods, ctermMods,
                                       varMods, ntermVarMods, ctermVarMods,
                                       varModSequence, args.bin_width)
                charged_decoy_psms.append(curr_psm)

        if charged_target_psms:
            top_target = max(charged_target_psms,key = scoref)
        else:
            top_target = []
        if charged_decoy_psms:
            top_decoy = max(charged_decoy_psms,key = scoref)
        else:
            top_decoy = []

        top_psms.append((sid, top_target, top_decoy))
    return top_psms

def runDidea_inCore(args):    
    """ Run Didea on a single thread
    """
    # Load spectra
    spectra, _, _, validcharges = load_spectra(args.spectra,
                                               args.charges)
    # set global variable to parsed/all encountered charges
    args.charges = validcharges
    # parse modifications
    mods, var_mods = parse_var_mods(args.mods_spec, True)
    nterm_mods, nterm_var_mods = parse_var_mods(args.nterm_peptide_mods_spec, False)
    cterm_mods, cterm_var_mods = parse_var_mods(args.cterm_peptide_mods_spec, False)

    # PeptideDB instances, where
    #    peptide[0] = peptide mass (float)
    #    peptide[1] = peptide string
    #    peptide[2] = protein name (mapped to an integer for the protein value encountered in the file)
    #    peptide[3] = nterm_flanking (character)
    #    peptide[4] = cterm_flanking (character)
    #    peptide[5] = binary string deoting variable modifications
    targets, decoys = didea_load_database(args, var_mods, nterm_var_mods, cterm_var_mods)
    preprocess = pipeline(args.normalize)

    # check whether high- or low-res mode
    if args.high_res_ms2:
        if args.bin_width < 1.:
            args.num_bins = int(round(max_mz / args.bin_width))
        learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
        learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)
        # learnedLambdas2 = {}
        # learnedLambdas3 = {}
        # for tau in range(-37,38):
        #     learnedLambdas2[tau] = 0.25
        #     learnedLambdas3[tau] = 0.25
        # learnedLambdas2[0] = 1.0
        # learnedLambdas3[0] = 1.0
    else:
        learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
        learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)

    print args.num_bins, args.bin_width

    ranges = simple_uniform_binwidth(0.0, args.num_bins,
                                     bin_width = args.bin_width)

    scored_psms = []
    ################################ load target and decoy databases using didea_load_database, score accoringly
    scored_psms = score_didea_spectra_incore(args, spectra, targets, decoys,
                                             ranges, preprocess, learnedLambdas2, learnedLambdas3,
                                             mods, nterm_mods, cterm_mods,
                                             var_mods, nterm_var_mods, cterm_var_mods)
    scored_psms.sort(key = lambda r: r[0])
    write_dideaSearch_ident(args.output+ '.txt', scored_psms,
                            mods, nterm_mods, cterm_mods,
                            var_mods, nterm_var_mods, cterm_var_mods)

def runDidea(args):    
    """ Run Didea on a single thread
    """
    # parse modifications
    mods, var_mods = parse_var_mods(args.mods_spec, True)
    nterm_mods, nterm_var_mods = parse_var_mods(args.nterm_peptide_mods_spec, False)
    cterm_mods, cterm_var_mods = parse_var_mods(args.cterm_peptide_mods_spec, False)

    preprocess = pipeline(args.normalize)

    # check whether high- or low-res mode
    if args.high_res_ms2:
        if args.bin_width < 1.:
            args.num_bins = int(round(max_mz / args.bin_width))

    ranges = simple_uniform_binwidth(0.0, args.num_bins,
                                     bin_width = args.bin_width)
    learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
    learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)

    scored_psms = []
    for data in candidate_binarydb_spectra_generator(args,
                                                     mods, nterm_mods, cterm_mods,
                                                     var_mods, nterm_var_mods, cterm_var_mods):
        top_psms = score_didea_spectra(args, data, ranges,
                                       preprocess, learnedLambdas2, learnedLambdas3,
                                       mods, nterm_mods, cterm_mods,
                                       var_mods, nterm_var_mods, cterm_var_mods)
        scored_psms += top_psms

    write_dideaSearch_ident(args.output+ '.txt', scored_psms,
                            mods, nterm_mods, cterm_mods,
                            var_mods, nterm_var_mods, cterm_var_mods)

def dideaThread(args, spectra, sidChargePreMass,
                ranges, preprocess, learnedLambdas2, learnedLambdas3,
                mods, nterm_mods, cterm_mods,
                var_mods, nterm_var_mods, cterm_var_mods):
    """ Search subset of spectra on a single thread
    """

    scored_psms = []
    for data in didea_candidate_binarydb_generator(args,
                                                   spectra, sidChargePreMass,
                                                   mods, nterm_mods, cterm_mods,
                                                   var_mods, nterm_var_mods, cterm_var_mods):
        top_psms = score_didea_spectra(args, data, ranges,
                                       preprocess, learnedLambdas2, learnedLambdas3,
                                       mods, nterm_mods, cterm_mods,
                                       var_mods, nterm_var_mods, cterm_var_mods)
        scored_psms += top_psms

    return scored_psms

def runDidea_multithread(options):
    """ Run Didea on a multiple threads
    """
    # parse modifications
    mods, var_mods = parse_var_mods(options.mods_spec, True)
    nterm_mods, nterm_var_mods = parse_var_mods(options.nterm_peptide_mods_spec, False)
    cterm_mods, cterm_var_mods = parse_var_mods(options.cterm_peptide_mods_spec, False)

    # Load spectra
    spectra, _, _, validcharges, sidChargePreMass = load_spectra_ret_dict(options.spectra,
                                                                          options.charges)
    # set global variable to parsed/all encountered charges
    options.charges = validcharges

    # shuffle precursormasses
    random.shuffle(sidChargePreMass)
    # calculate num spectra to score per thread
    numThreads = min(mp.cpu_count() - 1, options.num_threads)
    inc = int(len(spectra) / numThreads)
    partitions = []
    l = 0
    r = inc
    for i in range(numThreads-1):
        partitions.append(sidChargePreMass[l:r])
        partitions[i].sort(key = lambda r: r[2])
        l += inc
        r += inc
    partitions.append(sidChargePreMass[l:])
    partitions[-1].sort(key = lambda r: r[2])

    preprocess = pipeline(args.normalize)
    # check whether high- or low-res mode
    if args.high_res_ms2:
        if args.bin_width < 1.:
            args.num_bins = int(round(max_mz / args.bin_width))
        learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
        learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)
        # learnedLambdas2 = {}
        # learnedLambdas3 = {}
        # for tau in range(-37,38):
        #     learnedLambdas2[tau] = 0.25
        #     learnedLambdas3[tau] = 0.25
        # # learnedLambdas2[0] = 1.0
        # # learnedLambdas3[0] = 1.0
    else:
        learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
        learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)

    print args.num_bins, args.bin_width

    ranges = simple_uniform_binwidth(0.0, args.num_bins,
                                     bin_width = args.bin_width)
    # learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
    # learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)

    pool = mp.Pool(processes = numThreads)
    # perform map: distribute jobs to CPUs
    results = [pool.apply_async(dideaThread, 
                                args=(args, spectra, partitions[i], 
                                      ranges, preprocess, learnedLambdas2, learnedLambdas3,
                                      mods, nterm_mods, cterm_mods,
                                      var_mods, nterm_var_mods, cterm_var_mods )) for i in range(numThreads)]

    pool.close()
    pool.join() # wait for jobs to finish before continuing

    scored_psms = []
    for p in results:
        scored_psms += p.get()

    scored_psms.sort(key = lambda r: r[0])

    write_dideaSearch_ident(options.output+ '.txt', scored_psms,
                            mods, nterm_mods, cterm_mods,
                            var_mods, nterm_var_mods, cterm_var_mods)

def runDidea_multithread_inCore(options):
    """ Run Didea on a multiple threads
    """

    # parse modifications
    mods, var_mods = parse_var_mods(options.mods_spec, True)
    nterm_mods, nterm_var_mods = parse_var_mods(options.nterm_peptide_mods_spec, False)
    cterm_mods, cterm_var_mods = parse_var_mods(options.cterm_peptide_mods_spec, False)

    # Load spectra
    spectra, _, _, validcharges = load_spectra(args.spectra,
                                               args.charges)
    # set global variable to parsed/all encountered charges
    options.charges = validcharges
    # shuffle precursormasses
    random.shuffle(spectra)
    # calculate num spectra to score per thread
    numThreads = min(mp.cpu_count() - 1, options.num_threads)
    inc = int(len(spectra) / numThreads)
    partitions = []
    l = 0
    r = inc
    for i in range(numThreads-1):
        partitions.append(spectra[l:r])
        l += inc
        r += inc
    partitions.append(spectra[l:])

    # PeptideDB instances, where
    #    peptide[0] = peptide mass (float)
    #    peptide[1] = peptide string
    #    peptide[2] = protein name (mapped to an integer for the protein value encountered in the file)
    #    peptide[3] = nterm_flanking (character)
    #    peptide[4] = cterm_flanking (character)
    #    peptide[5] = binary string deoting variable modifications
    targets, decoys = didea_load_database(args, var_mods, nterm_var_mods, cterm_var_mods)
    preprocess = pipeline(args.normalize)

    # check whether high- or low-res mode
    if args.high_res_ms2:
        if args.bin_width < 1.:
            args.num_bins = int(math.ceil(max_mz / args.bin_width))
        learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
        learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)
        # learnedLambdas2 = {}
        # learnedLambdas3 = {}
        # for tau in range(-37,38):
        #     learnedLambdas2[tau] = 0.25
        #     learnedLambdas3[tau] = 0.25
        # learnedLambdas2[0] = 0.5
        # learnedLambdas3[0] = 0.5
    else:
        learnedLambdas2 = load_lambdas(args.learned_lambdas_ch2)
        learnedLambdas3 = load_lambdas(args.learned_lambdas_ch3)

    print args.num_bins, args.bin_width

    ranges = simple_uniform_binwidth(0.0, args.num_bins,
                                     bin_width = args.bin_width)

    pool = mp.Pool(processes = numThreads)
    # perform map: distribute jobs to CPUs
    results = [pool.apply_async(score_didea_spectra_incore,
                                args=(args, partitions[i], targets, decoys,
                                      ranges, preprocess, learnedLambdas2, learnedLambdas3,
                                      mods, nterm_mods, cterm_mods,
                                      var_mods, nterm_var_mods, cterm_var_mods )) for i in range(numThreads)]

    pool.close()
    pool.join() # wait for jobs to finish before continuing

    scored_psms = []
    for p in results:
        scored_psms += p.get()

    scored_psms.sort(key = lambda r: r[0])

    write_dideaSearch_ident(options.output+ '.txt', scored_psms,
                            mods, nterm_mods, cterm_mods,
                            var_mods, nterm_var_mods, cterm_var_mods)

if __name__ == '__main__':
    # read in options and process input arguments
    args = parseInputOptions()
    process_args(args)

    if args.num_threads <= 1:
        runDidea_inCore(args)
    else:
        runDidea_multithread_inCore(args)
        
    if stdo:
        stdo.close()
    if stde:
        stde.close()

    # # out-of-core searches, use when digested peptide database cannot be loaded into memory
    # if args.num_threads <= 1:
    #     runDidea(args)
    # else:
    #     runDidea_multithread(args)
