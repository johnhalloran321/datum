#!/usr/bin/env python
#
# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2018 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0

from __future__ import with_statement

__authors__ = ['John Halloran <jthalloran@ucdavis.edu>' ]

import os
import re
import sys
import argparse
import pyFiles.digest_fasta as df
import shlex
import math
import string
import random
import pyFiles.psm as ppsm
import copy

import csv
import itertools
import types
import numpy

from digest import parse_var_mods
from dtk import load_spectra_minMaxMz, write_lorikeet_file

try:
    import sys
    import matplotlib
    import pylab
    if not sys.modules.has_key('matplotlib'):
        matplotlib.use('Agg')
except ImportError:
    print >> sys.stderr, \
    'Module "pylab" not available. You must install matplotlib to use this.'
    exit(-1)


allPeps = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')-set('JOBZUX')

mono_elements = { 'H' : 1.007825035,
                  'C' : 12.0,
                  'N' : 14.003074,
                  'O' : 15.99491463,
                  'P' : 30.973762,
                  'S' : 31.9720707,
                  'Se' : 79.9165196
}

avg_elements = { 'H' : 1.00794,
                 'C' : 12.0107,
                 'N' : 14.0067,
                 'O' : 15.9994,
                 'P' : 30.973761,
                 'S' : 32.065,
                 'Se' : 79.96
}

aa_mono_mass = { 'A' :  mono_elements['C']*3 + mono_elements['H']*5 + mono_elements['N'] + mono_elements['O'],
                 'C' :  mono_elements['C']*3 + mono_elements['H']*5 + mono_elements['N'] + mono_elements['O'] + mono_elements['S'],
                 'D' :  mono_elements['C']*4 + mono_elements['H']*5 + mono_elements['N'] + mono_elements['O']*3,
                 'E' :  mono_elements['C']*5 + mono_elements['H']*7 + mono_elements['N'] + mono_elements['O']*3,
                 'F' :  mono_elements['C']*9 + mono_elements['H']*9 + mono_elements['N'] + mono_elements['O'],
                 'G' :  mono_elements['C']*2 + mono_elements['H']*3 + mono_elements['N'] + mono_elements['O'],
                 'H' :  mono_elements['C']*6 + mono_elements['H']*7 + mono_elements['N']*3 + mono_elements['O'],
                 'I' :  mono_elements['C']*6 + mono_elements['H']*11 + mono_elements['N'] + mono_elements['O'],
                 'J' :  mono_elements['C']*6 + mono_elements['H']*11 + mono_elements['N'] + mono_elements['O'],
                 'K' :  mono_elements['C']*6 + mono_elements['H']*12 + mono_elements['N']*2 + mono_elements['O'],
                 'L' :  mono_elements['C']*6 + mono_elements['H']*11 + mono_elements['N'] + mono_elements['O'],
                 'M' :  mono_elements['C']*5 + mono_elements['H']*9 + mono_elements['N'] + mono_elements['O'] + mono_elements['S'],
                 'N' :  mono_elements['C']*4 + mono_elements['H']*6 + mono_elements['N']*2 + mono_elements['O']*2,
                 'O' :  mono_elements['C']*12 + mono_elements['H']*21 + mono_elements['N']*3 + mono_elements['O']*3,
                 'P' :  mono_elements['C']*5 + mono_elements['H']*7 + mono_elements['N'] + mono_elements['O'],
                 'Q' :  mono_elements['C']*5 + mono_elements['H']*8 + mono_elements['N']*2 + mono_elements['O']*2,
                 'R' :  mono_elements['C']*6 + mono_elements['H']*12 + mono_elements['N']*4 + mono_elements['O'],
                 'S' :  mono_elements['C']*3 + mono_elements['H']*5 + mono_elements['N'] + mono_elements['O']*2,
                 'T' :  mono_elements['C']*4 + mono_elements['H']*7 + mono_elements['N'] + mono_elements['O']*2,
                 'U' :  mono_elements['C']*3 + mono_elements['H']*7 + mono_elements['N'] + mono_elements['O']*2 + mono_elements['Se'],
                 'V' :  mono_elements['C']*5 + mono_elements['H']*9 + mono_elements['N'] + mono_elements['O'],
                 'W' :  mono_elements['C']*11 + mono_elements['H']*10 + mono_elements['N']*2 + mono_elements['O'],
                 'Y' :  mono_elements['C']*9 + mono_elements['H']*9 + mono_elements['N'] + mono_elements['O'] * 2,
}

def extractVarModOffsets(p, stripShifts = False):
    if stripShifts:
        p = re.sub("[\[].*?[\]]", "", p)
        return p, []

    var_mod_offsets = re.findall("[\[].*?[\]]", p)
    p = re.sub("[\[].*?[\]]", "-", p)
    var_offsets = []
    ind = 0
    wasMod = False
    for aa in p[1:]:
        if wasMod: # skip to next residue
            wasMod = False
            continue
        m = 0.0
        if aa == '-':
            m = float(var_mod_offsets[ind][1:-1])
            ind += 1
            wasMod = True
        var_offsets.append(m)
    # check last residue
    if not wasMod:
        var_offsets.append(0.0)

    p = re.sub("-", "", p)

    return p, var_offsets


def gen_lorikeet_psmDict(psmDict, spectrumFile, 
                         outputDirectory,
                         plotList = 'currPsms.html',
                         mods_spec = '',  
                         nterm_mods_spec = '', 
                         cterm_mods_spec = '',
                         scanInd = 0,
                         peptideInd = 1,
                         scoreInd = 2,
                         chargeInd = 3,
                         varModOffsetInd = 4,
                         cMod = True):
    """ Generate html files for Lorikeet plugin
    """
    # parse modifications
    mods, var_mods = parse_var_mods(mods_spec, True)
    nterm_mods, nterm_var_mods = parse_var_mods(nterm_mods_spec, False)
    cterm_mods, cterm_var_mods = parse_var_mods(cterm_mods_spec, False)

    if cMod: # will typically be true
        if 'C' not in mods:
            mods['C'] = 57.021464

    # lorikeet only supports a single static mod to the n-terminus
    if nterm_mods:
        nm = []
        for aa in allPeps:
            if aa not in nterm_mods:
                print "Lorikeet only supports a single constant shift to the n-terminus, please specify only one n-terminal shift using X"
                print "Exitting"
                exit(-1)
            nm.append(nterm_mods[aa])
        if len(set(nm)) > 1:
            print "different n-teriminal shifts supplied for different amino acids, Lorikeet only supports a single constant shift to the n-terminus."
            print "Exitting"
            exit(-1)
        nterm_mods = nm[0]

    # lorikeet only supports a single static mod to the c-terminus
    if cterm_mods:
        cm = []
        for aa in allPeps:
            if aa not in cterm_mods:
                print "Lorikeet only supports a single constant shift to the n-terminus, please specify only one n-terminal shift using X"
                print "Exitting"
                exit(-1)
            cm.append(cterm_mods[aa])
        if len(set(cm)) > 1:
            print "different n-teriminal shifts supplied for different amino acids, Lorikeet only supports a single constant shift to the n-terminus."
            print "Exitting"
            exit(-1)
        cterm_mods = cm[0]                
        
    t = psmDict
    # load .ms2 spectra
    spectra, _, _, _ = load_spectra_minMaxMz(spectrumFile)

    # make output directory
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)

    # open filestream for master list of html files
    fid = open(plotList, 'w')

    for sid in t:
        if sid not in spectra:
            print "Scan number %d specified for PSM, but not appear in provided ms2 file, skipping" % sid
            continue
        spec = spectra[sid]
        psm = t[sid]

        charge = psm[chargeInd]

        varModTuple = []
        var_mod_offsets = psm[varModOffsetInd]
        if len(var_mod_offsets) > 0:
            for ind, (aa, offset) in enumerate(zip(psm[1], var_mod_offsets)):
                if offset > 0.:
                    varModTuple.append((ind+1, aa, offset))

        filename = 'scan' + str(sid) + '-' + psm[peptideInd] + '-ch' + str(charge) + '.html'
        filename = os.path.join(outputDirectory,filename)
        write_lorikeet_file(psm, spec, filename, 
                            mods, nterm_mods, cterm_mods, 
                            varModTuple)
        fid.write("<a href=\"%s\">Scan %d, %s, Charge %d</a><br>\n"  %
                  (filename, sid, psm[peptideInd], charge))
    fid.close()


def filterSpectra(didea_t, didea_d, method_t, method_d, doLog = True, thresh = 0.004, thresh2 = -5):
    # fields: (0) scan (1) peptide string (2) score (3) charge
    # build didea target and decoy dictionaries
    t_dict = {}
    d_dict = {}
    for t,d in zip(didea_t, didea_d):
        t_dict[t[0]] = t
        d_dict[d[0]] = d

    dd_dict = {}
    for d in method_d:
        dd_dict[d[0]] = d

    didea_t_filt = []
    didea_d_filt = []
    method_t_filt = []
    method_d_filt = []
    eps = 0.1
    numThresh = 20

    thresh_dictA = {}
    thresh_dictB = {}
    
    n = 0

    for scan in d_dict:
        if n < numThresh:
            if d_dict[scan][2] >= thresh2:
                if scan in t_dict and d_dict[scan][2] > t_dict[scan][2]:
                    if scan in dd_dict:
                        print "d", d_dict[scan], dd_dict[scan]
                    else:
                        print "d", d_dict[scan]
                    n += 1
                    # remove static mod indicators from didea output
                    p, var_offsets = extractVarModOffsets(d_dict[scan][1], stripShifts = True)
                    d_dict[scan][1] = p
                    d_dict[scan][4] = var_offsets
                    thresh_dictB[scan] = d_dict[scan]

    for t in method_t:
        if doLog:
            t[2] = -math.log(-t[2] + eps)
        scan = t[0]
        if scan in t_dict:
            didea_t_filt.append(t_dict[scan])
            method_t_filt.append(t)
            # if n < numThresh:
            #     if t[2] >= thresh and t_dict[scan][2] >= thresh2:
            #         print "t", t, t_dict[scan]
            #         n += 1
            #         # inspect var mod offset info
            #         p, var_offsets = extractVarModOffsets(t[1])
            #         t[1] = p
            #         t.append(var_offsets)
            #         thresh_dictA[scan] = t
            #         # remove static mod indicators from didea output
            #         p, var_offsets = extractVarModOffsets(t_dict[scan][1], stripShifts = True)
            #         t_dict[scan][1] = p
            #         t_dict[scan][4] = var_offsets
            #         thresh_dictB[scan] = t_dict[scan]

    for d in method_d:
        if doLog:
            d[2] = -math.log(-d[2] + eps)
        scan = d[0]
        if scan in d_dict:
            didea_d_filt.append(d_dict[scan])
            method_d_filt.append(d)
            if n < numThresh:
                if d[2] >= thresh and d_dict[scan][2] >= thresh2:
                    print "d", d, d_dict[scan]
                    n += 1
                    # # inspect var mod offset info
                    # p, var_offsets = extractVarModOffsets(d[1])
                    # d[1] = p
                    # d.append(var_offsets)
                    # thresh_dictA[scan] = d
                    # # remove static mod indicators from didea output
                    # p, var_offsets = extractVarModOffsets(d_dict[scan][1], stripShifts = True)
                    # d_dict[scan][1] = p
                    # d_dict[scan][4] = var_offsets
                    # thresh_dictB[scan] = d_dict[scan]

    
    return didea_t_filt, didea_d_filt, method_t_filt, method_d_filt, thresh_dictA, thresh_dictB

def histogram(targets, decoys, fn, bins = 40):
    """Histogram of the score distribution between target and decoy PSMs.

    Arguments:
        targets: Iterable of floats, each the score of a target PSM.
        decoys: Iterable of floats, each the score of a decoy PSM.
        fn: Name of the output file. The format is inferred from the
            extension: e.g., foo.png -> PNG, foo.pdf -> PDF. The image
            formats allowed are those supported by matplotlib: png,
            pdf, svg, ps, eps, tiff.
        bins: Number of bins in the histogram [default: 40].

    Effects:
        Outputs the image to the file specified in 'fn'.

    """
    pylab.clf()
    # pylab.hold(True)
    pylab.xlabel('Score')
    pylab.ylabel('Pr(Score)')

    l = min(min(decoys), min(targets))
    h = max(max(decoys), max(targets))
    _, _, h1 = pylab.hist(targets, bins = bins, range = (l,h), density = 1,
                          color = 'b', alpha = 0.25)
    _, _, h2 = pylab.hist(decoys, bins = bins, range = (l,h), density = 1,
                          color = 'm', alpha = 0.25)
    pylab.legend((h1[0], h2[0]), ('Target PSMs', 'Decoy PSMs'), loc = 2)
    pylab.savefig('%s' % fn)

def scatterplot(first_method, second_method, fn, labels = None):
    """
    """
    if len(first_method[0]) != len(second_method[0]):
        raise Exception('Methods evaluated over different spectra.')

    t1 = list(r[2] for r in first_method[0])
    d1 = list(r[2] for r in first_method[1])
    t2 = list(r[2] for r in second_method[0])
    d2 = list(r[2] for r in second_method[1])

    pylab.clf()
    # pylab.hold(True)
    pylab.scatter(t1, t2, color = 'b', alpha = 0.20, s = 2)
    pylab.scatter(d1, d2, color = 'r', alpha = 0.10, s = 1)
    pylab.xlim( (min(min(t1), min(d1)), max(max(t1), max(d1))) )
    if labels:
        pylab.xlabel(labels[0])
        pylab.ylabel(labels[1])

    pylab.savefig(fn)

def scatterCharge(targets, decoys, fn, labels = None):
    """Scatterplot of the PSM scores for two methods.

    Assumes that the (target, decoy) tuples are ordered, so that
    target[i] and decoy[i] refer to the same spectrum.

    Arguments:
        first_method: (targets, decoys) tuple, where each entry in
           targets, and in decoys, is a triplet of the form (s, p, f, c)
           where s is a spectrum id, p is a peptide, and f is the score
           of the match. See the outputs of load_ident.
        second_method: Same form as first_method, but represents the
           identifications produced by another algorithm
        fn: Name of the file to plot to.
        labels: (first_method_name, second_method_name), the labels for
           the two methods, used to labelled in the x and y axes of the
           scatterplot.

    Effects:
        Creates a plot in file 'fn'. The position of each red point encodes
        the score of the top decoy PSM, for a spectrum. The position of
        each blue point encodes the score of the top target PSM, for a
        spectrum.

    """
    # if len(first_method[0]) != len(second_method[0]):
    #     raise Exception('Methods evaluated over different spectra.')

    tscores = list(r[2] for r in targets)
    dscores = list(r[2] for r in decoys)
    tcharges = list(r[4] for r in targets)
    dcharges = list(r[4] for r in decoys)

    pylab.clf()
    # pylab.hold(True)
    pylab.scatter(tscores, tcharges, color = 'b', alpha = 0.20, s = 2)
    pylab.scatter(dscores, dcharges, color = 'r', alpha = 0.10, s = 1)
    pylab.xlim( (min(min(tscores), min(dscores)), max(max(tscores), max(dscores))) )
    if labels:
        pylab.xlabel('Scores')
        pylab.ylabel('Charges')

    pylab.savefig(fn)

def load_ident(filename, len_norm = False, distinct_sids = True, onlyScores = True):
    """ Load all PSMs and features ident file
    """
    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    sidKey = "Sid" # note that this typically denotes retention time
    if "Sid" not in l:
        sidKey = 'Scan'
        if sidKey not in l:
            raise ValueError("No Sid field, exitting")

    # spectrum identification key for PIN files
    numPeps = 0
    # look at score key and charge keys
    scoreKey = 'Score'
    # fields we have to keep track of
    psmKeys = set(["Kind", sidKey, scoreKey, "Peptide", "Proteins", "Charge"])
    for i, l in enumerate(reader):
        try:
            sid = int(l[sidKey])
        except ValueError:
            print "Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1)

        if l["Kind"] == 't':
            kind = 't'
        elif l["Kind"] == 'd':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be t or d, exitting" % l["Label"]
            exit(-1)

        el = []
        el.append(sid)
        el.append(l["Peptide"])
        el.append(float(l[scoreKey]))
        el.append(int(l["Charge"]))
        el.append(len(l["Peptide"]))

        scoreInd = 2

        if len_norm:
            el[scoreInd] /= float(el[2])

        if kind == 't':
            if sid in targets:
                if el[scoreInd] > targets[sid][scoreInd]:
                    targets[sid] = el
            else:
                targets[sid] = el
                numPeps += 1
        elif kind == 'd':
            if sid in decoys:
                if el[scoreInd] > decoys[sid][scoreInd]:
                    decoys[sid] = el
            else:
                decoys[sid] = el
                numPeps += 1

    top_targets = []
    top_decoys = []
    if distinct_sids:
        # make sure targets and decoys are distinct:
        if len(set(targets.iterkeys()) & set(decoys.iterkeys())):
            t_keys = set(targets.iterkeys())
            d_keys = set(decoys.iterkeys())
            # write out lone targets
            for k in t_keys - d_keys:
                if onlyScores:
                    top_targets.append(targets[k][scoreInd])
                else:
                    top_targets.append(targets[k])
            for k in d_keys - t_keys:
                if onlyScores:
                    top_decoys.append(decoys[k][scoreInd])
                else:
                    top_decoys.append(decoys[k])
            for k in t_keys & d_keys:
                if decoys[k][scoreInd] > targets[k][scoreInd]:
                    if onlyScores:
                        top_decoys.append(decoys[k][scoreInd])
                    else:
                        top_decoys.append(decoys[k])
                else:
                    if onlyScores:
                        top_targets.append(targets[k][scoreInd])
                    else:
                        top_targets.append(targets[k])
        else:
            for sid in targets:
                if onlyScores:
                    top_targets.append(targets[sid][scoreInd])
                else:
                    top_targets.append(targets[sid])
            for sid in decoys:
                if onlyScores:
                    top_decoys.append(decoys[sid][scoreInd])
                else:
                    top_decoys.append(decoys[sid])
    else:
        for sid in targets:
            if onlyScores:
                top_targets.append(targets[sid][scoreInd])
            else:
                top_targets.append(targets[sid])
        for sid in decoys:
            if onlyScores:
                top_decoys.append(decoys[sid][scoreInd])
            else:
                top_decoys.append(decoys[sid])
    print "Loaded %d target and %d decoy PSMs" % (len(top_targets), len(top_decoys))

    return top_targets, top_decoys

def targetsAtQ(positives, negatives):
  """Returns q-values along with target PSMs
     PSM fields: 0) Sid (1) Peptide object (2) Score (3) Charge
  """
  # build hash tables, assume positives and negatives are unordered
  targets = {}
  for t in positives:
    sid = t[0]
    if sid in targets:
      if targets[sid][2] > t[2]:
        targets[sid] = t

  decoys = {}
  for d in negatives:
    sid = d[0]
    if sid in decoys:
      if decoys[sid][2] > d[2]:
        decoys[sid] = d

  all = []
  # for p,n in zip(positives,negatives):
  for sid in targets:
    if sid in decoys: # only consider common intersection
      t = targets[sid]
      d = decoys[sid]
      if t[2] > d[2]:
        all.append((t, 1))
      else:
        all.append((d, 0))

  #--- sort descending
  all.sort( lambda x,y: cmp(y[0][2], x[0][2]) )
  fdrs = []
  posTot = 0.0
  fpTot = 0.0
  fdr = 0.0

  #--- iterates through scores
  for item in all:
    if item[1] == 1: posTot += 1.0
    else: fpTot += 1.0
    
    #--- check for zero positives
    if posTot == 0.0: fdr = 100.0
    else: fdr = fpTot / posTot

    #--- note the q
    fdrs.append(fdr)
    ps.append(posTot)

  taq = []
  lastQ = 100.0
  for idx in range(len(fdrs)-1, -1, -1):
    
    q = 0.0
    #--- q can never go up. 
    if lastQ < fdrs[idx]:
      q = lastQ
    else:
      q = fdrs[idx]
    lastQ = q
    
    if all[idx][1]:
      taq.append((qs, all[idx]))

  return taq

def calcQCompetition(positives, negatives, label="Notitle"):
  """Calculates P vs q xy points from a list of positives and negative scores"""
  all = []
  for p,n in zip(positives,negatives):
    if p > n:
      all.append((p, 1))
    else:
      all.append((n, 0))

  # random.shuffle(all)
  #--- sort descending
  all.sort( lambda x,y: cmp(y[0], x[0]) )
  # all.sort( key = lambda r : -r[0] )
  ps = []
  fdrs = []
  posTot = 0.0
  fpTot = 0.0
  fdr = 0.0

  #--- iterates through scores
  for item in all:
    if item[1] == 1: posTot += 1.0
    else: fpTot += 1.0
    
    #--- check for zero positives
    if posTot == 0.0: fdr = 100.0
    else: fdr = fpTot / posTot

    #--- note the q
    fdrs.append(fdr)
    ps.append(posTot)

  qs = []
  lastQ = 100.0
  for idx in range(len(fdrs)-1, -1, -1):
    
    q = 0.0
    #--- q can never go up. 
    if lastQ < fdrs[idx]:
      q = lastQ
    else:
      q = fdrs[idx]
    lastQ = q
    qs.append(q)
  
  qs.reverse()
  # pos = range(len(qs))
  return qs, ps

def percolatorQPlot(qvals, label="Notitle"):
  """Directly reads percolator plot"""
  qs = []
  ps = []
  tot = 0

  qvals.sort()

  currQ = qvals[0]
  qs.append(currQ)
  for q in qvals:
    if q <= currQ:
      tot += 1
    else:
      ps.append(tot)
      qs.append(q)
      tot += 1
  ps.append(tot)
  return qs, ps


def plot(scorelists, output, qrange = None, labels = None, **kwargs):
    """Plot multiple absolute ranking plots on one axis.

    The y-axis is the number of spectra, so plotting two methods is
    only comparable on the resulting plot if you evaluated them on the
    same number of spectra. Typically, the methods being compared will
    be applied to exactly the same set of spectra.

    Args:
        scorelists: List of pairs of vectors. The first entry in each
            pair is the vector of target scores, the second entry is the
            vector of decoy scores. E.g. [(t1,d1),(t2,d2),...,(tN,dN)].
            Each (ti,di) pair represents a peptide identification algorithm.
        output: Name of the output plot.
        qrange: Range of q-values to plot, must have two values (low,high).
        labels: Iterable of names for each method.

    Keyword Arguments:
        paranoid: If it evaluates to true, perform paranoid checks on the
            input scores: i.e., test that all the values are floating point.
        expandy: A floating point number > 1, which defines an percentage by
            which to expand the y-axis. Useful if the absolute ranking curve
            reaches its maximum value at a q-value < 1.0.

    Effects:
        Creates a file with the name specified in arg output.

    """
    if kwargs.has_key('publish') and kwargs['publish']:
        # linewidth = 6
        linewidth = 1
        # linewidth = [ 4.0, 3.5, 3.25, 3.0, 2.5, 2.5, 2.5, 2.5 ]
        xlabel = 'q-value'
        xlabel = 'False Discovery Rate (FDR)'
        if kwargs.has_key('ylabel') and kwargs['ylabel']:
            ylabel = kwargs['ylabel']
        else:
            ylabel = 'Spectra identified'                
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['legend.fontsize'] = kwargs['font']
        # matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 24
        matplotlib.rcParams['ytick.labelsize'] = 24
        matplotlib.rcParams['axes.labelsize'] = 22
        kwargs['tight'] = True
    else:
        linewidth = [2] * 8
        #linewidth = [ 4.0, 3.5, 3.25, 3.0, 2.5, 2.5, 2.5, 2.5 ]
        xlabel = 'q-value'
        if kwargs.has_key('ylabel') and kwargs['ylabel']:
            ylabel = kwargs['ylabel']
        else:
            ylabel = 'Number of target matches'

    if kwargs.has_key('altDash') and kwargs['altDash']:
        linestyle = [ '-', '-.', ':', '--', '-', '--', '-', '--', '-']
        # Color-blind friend line colors, as RGB triplets. The colors alternate,
        # warm-cool-warm-cool.
        linecolors = [ (0.0, 0.0, 0.0),
                       'violet',
                       (0.8, 0.4, 0.0),
                       'red',
                       (0.0, 0.45, 0.70),
                       (0.8, 0.6, 0.7),
                       'gold',
                       (0.0, 0.6, 0.5),
                       (0.35, 0.7, 0.9),
                       (0.43, 0.17, 0.60),
                       (0.95, 0.9, 0.25),
                       'gold']
    else:
        linestyle = [ '-', '-', '-', '-', '-', '-', '-', '-', '--']
        # Color-blind friend line colors, as RGB triplets. The colors alternate,
        # warm-cool-warm-cool.
        linecolors = [ (0.0, 0.0, 0.0),
                       (0.8, 0.4, 0.0),
                       (0.0, 0.45, 0.70),
                       (0.8, 0.6, 0.7),
                       'gold',
                       (0.0, 0.6, 0.5),
                       (0.35, 0.7, 0.9),
                       (0.43, 0.17, 0.60),
                       (0.95, 0.9, 0.25),
                       'gold']

    if len(scorelists) > len(linecolors):
        raise ValueError('Only have %d color, but %d curves' % (
                         len(linecolors), len(scorelists)))

    pylab.clf()
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.nipy_spectral()
    pylab.gray()

    if not qrange:
        qrange = (0.0, 1.0)

    h = -1
    i = 0
    print "Rel ranking"
    for targets, decoys in scorelists:
        x, y = calcQCompetition(targets, decoys)
        n = float(len(y))
        y = [float(iy) / n for iy in y]
        h = max(itertools.chain([h], (b for a, b in zip(x, y) if
                                      a <= qrange[1])))
        pylab.plot(x, y, color = linecolors[i], linewidth = 2, linestyle = linestyle[i])
        rr = float(len(targets)) / float(len(targets)+len(decoys))
        print "%s: %f" % (labels[i], rr)
        i = i+1

    # Don't display 0.0 on the q-value axis: makes the origin less cluttered.
    if qrange[0] == 0.0:
        pylab.xticks(numpy.delete(numpy.linspace(qrange[0], qrange[1], 6), 0), ha='right')

    if kwargs.has_key('expandy'):
        expandy = float(kwargs['expandy'])
        if expandy < 1.0:
            raise PlotException('expandy expansion factor < 1: %f' % expandy)

    pylab.xlim(qrange[0], qrange[1])
    assert(h > 0)
    pylab.ylim([0, h])
    pylab.legend(labels, loc = 'lower right', ncol=2, fontsize=8)

    yt, _ = pylab.yticks()
    if all(v % 1000 == 0 for v in yt):
        yl = []
        ytt = []
        for v in yt:
            if v < h:
                yl.append('$%d$' % int(v/1000))
                ytt.append(v)
        pylab.yticks(ytt, yl)
        pylab.ylabel(ylabel + ' (in 1000\'s)')

    pylab.savefig(output, bbox_inches='tight')

def load_benchmark_psms(psm_file, dataset = 'v07551_UofF_malaria_TMT_13.RAW'):
# scan	file	charge	spectrum.precursor.m.z	combined.p.value	combinedPeptide	combinedTargetOrDecoy	res.ev.p.value	resEvPeptide	resEvTargetOrDecoy	exact.p.value	xcorrPeptide	xcorrTargetOrDecoy	SpecEValue	MSGFPeptide	MSGFTargetOrDecoy	Weighted.Probability	amandaPeptide	amandaTargetOrDecoy	Morpheus.Score	morpheusPeptide	morpheusTargetOrDecoy
# 301	v07551_UofF_malaria_TMT_13.RAW	3	663.3505	0.999648417736384	ESIDNICAM[15.99]GFEK	target	1	GTILHDNMLSAETK	target	0.342347171822253	ESIDNICAM[15.99]GFEK	target	4.2783117e-06	+229.163IYYYYC+57.021ELTNK+229.163	decoy	NA	NA	NA	0	[TMT sixplex/tenplex on peptide N-terminus]EMSNYYYRLYK[TMT sixplex/tenplex on K]	target
# 390	v07551_UofF_malaria_TMT_13.RAW	3	530.3129	0.99999998623393	NRNEILEDK	decoy	1	DNSFFLLFK	decoy	0.765981755546727	NRNEILEDK	decoy	7.5250778e-06	+229.163EELWPSEIK+229.163	decoy	NA	NA	NA	1.02135066492911	[TMT sixplex/tenplex on peptide N-terminus]HYTQINLNK[TMT sixplex/tenplex on K]	decoy

    # Combined p-value keys: combined.p.value	combinedPeptide	combinedTargetOrDecoy
    combined_t = []
    combined_d = []
    combined_t_psms = []
    combined_d_psms = []
    # MS-GF+ keys: SpecEValue	MSGFPeptide	MSGFTargetOrDecoy
    msgf_t = []
    msgf_d = []
    msgf_t_psms = []
    msgf_d_psms = []
    with open(psm_file, 'r') as f:
        for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
            if l['file'] == dataset:
                # MS-GF+
                if l['SpecEValue'] != 'NA':
                    scoref = float(l['SpecEValue'])
                    scan = int(l['scan'])
                    c = int(l['charge'])
                    pep = l['MSGFPeptide']
                    if l['MSGFTargetOrDecoy'] == 'target':
                        msgf_t.append(-scoref)
                        msgf_t_psms.append([scan, pep, scoref, c])
                    else:
                        msgf_d.append(-scoref)
                        msgf_d_psms.append([scan, pep, scoref, c])
                # Combined p-value score
                if l['combined.p.value'] != 'NA':
                    scoref = float(l['combined.p.value'])
                    scan = int(l['scan'])
                    c = int(l['charge'])
                    pep = l['combinedPeptide']
                    if l['combinedTargetOrDecoy'] == 'target':
                        combined_t.append(-scoref)
                        combined_t_psms.append([scan, pep, scoref, c])
                    else:
                        combined_d.append(-scoref)
                        combined_d_psms.append([scan, pep, scoref, c])
    print "Loaded %d MS-GF+ targets, %d MS-GF+ decoys" % (len(msgf_t), len(msgf_d))
    print "Loaded %d Combined p-value targets, %d Combined p-value decoys" % (len(combined_t), len(combined_d))
    scorelists = [(msgf_t, msgf_d)]
    scorelists.append((combined_t, combined_d))
    psmlists = [(msgf_t_psms, msgf_d_psms)]
    psmlists.append((combined_t_psms, combined_d_psms))
    return scorelists, psmlists
    
if __name__ == '__main__':

    # print aa_mono_mass
    # exit(1)

    # process input arguments
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataset', type = str, action = 'store', 
                        default = 'v07551_UofF_malaria_TMT_13.RAW')
    parser.add_argument('--ms2File', type = str, action = 'store', 
                        default = '/home/jthalloran/highResBenchmark/v07551_UofF_malaria_TMT_13.ms2')
    parser.add_argument('--didea-ident', type = str, action = 'store')
    parser.add_argument('--output', type = str, action = 'store')
    parser.add_argument('--output2', type = str, action = 'store')
    parser.add_argument('--output3', type = str, action = 'store')
    args = parser.parse_args()

    psm_file = '/home/jthalloran/highResBenchmark/psmFile.txt'
    scorelists, psmlists = load_benchmark_psms(psm_file, dataset = args.dataset)
    didea_t, didea_d = load_ident(args.didea_ident)
    scorelists.append((didea_t,didea_d))
    labels = ['MS-GF+', 'Combined XCorr p-values', 'Didea']
    output = args.output
    qrange = (0.0,0.1)
    plot(scorelists, output, qrange, labels)
    histogram(didea_t, didea_d, args.output3)
    histogram(scorelists[1][0], scorelists[1][1], 'combined-hist.pdf', bins = 100)
    # create scatter plot
    didea_t_all, didea_d_all = load_ident(args.didea_ident, distinct_sids = False, onlyScores = False)

    didea_t, didea_d = load_ident(args.didea_ident, onlyScores = False)
    t_max = [0.0,0.0,-5.]
    d_max = [0.0,0.0,-5.]
    count = 0
    for t in didea_t:
        if t[2] > t_max[2]:
            t_max = t
    for d in didea_d:
        if d[2] > -2.5:
            count += 1
        if d[2] > d_max[2]:
            d_max = d
    print "t", t_max, "d", d_max
    print "%d decoys above score threshold %f" % (count, -2.5)
    # calculate intersection of PSMs
    # # MS-GF+
    # th = 0.004
    # didea_t, didea_d, msgf_t, msgf_d, _, _ = filterSpectra(didea_t_all, didea_d_all, psmlists[0][0], psmlists[0][1], doLog = False, thresh = th)
    # print "Intersected %d targets, %d decoys" % (len(didea_t), len(didea_d))
    # print "Intersected %d targets, %d decoys" % (len(msgf_t), len(msgf_d))
    # scatterplot((didea_t, didea_d), (msgf_t, msgf_d), 'dideaMsgf_scatter.pdf', labels = ['Didea', 'MS-GF+'])

    th = 0.0
    th2 = -3.2
    didea_t, didea_d, combined_t, combined_d, combined_dict, didea_dict = filterSpectra(didea_t_all, didea_d_all, psmlists[1][0], psmlists[1][1], doLog = False, thresh = th, thresh2 = th2)
    print "Intersected %d targets, %d decoys" % (len(didea_t), len(didea_d))
    print "Intersected %d targets, %d decoys" % (len(combined_t), len(combined_d))
    scatterplot((didea_t, didea_d), (combined_t, combined_d), args.output2, labels = ['Didea', 'Combined XCorr p-values'])

    ms2=args.ms2File
    gen_lorikeet_psmDict(combined_dict, ms2, 
                         'malaria-combinedAnalysis',
                         plotList = 'combinedPsms.html')# ,
                         # mods_spec = 'C+57.0214,K+229.16293',  
                         # nterm_mods_spec = 'X+229.16293')
    gen_lorikeet_psmDict(didea_dict, ms2, 
                         'malaria-dideaAnalysis',
                         plotList = 'dideaPsms.html')# ,
                         # mods_spec = 'C+57.0214,K+229.16293',  
                         # nterm_mods_spec = 'X+229.16293')
