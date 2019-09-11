#!/usr/bin/env python
#
# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2018 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0
# Command line parsing utilities.

"""Tools for creating a peptide database, usually using the output generated
by crux search-for-matches: i.e., search.{target,decoy}.txt.

"""

from __future__ import with_statement

__authors__ = [ 'John T. Halloran <jthalloran@ucdavis.edu>' ]

from bisect import bisect_left, bisect_right
from operator import itemgetter
from peptide import Peptide
import sys
import csv

class SimplePeptideDB(object):
    """A simpler version of a peptide database that takes the peptides.txt
    file produces by crux create-index --peptide-list T <fasta-file> <index-dir>

    The peptide database is a text file where each line is of the form
    <peptide> <neutral mass>

    Where peptide is a IUPAC sequence, and neutral mass it the average mass
    of a peptide (c.f.,
    http://noble.gs.washington.edu/proj/crux/crux-search-for-matches.html)

    """
    def __init__(self, filename):
        self.filename = filename
        self.peptides = [ ]
        self.masses = None
        self._parser(filename)

    def _parser(self, filename):
        records = [ ]
        # first detect what is the used delimiter; crux switched from a space to a tab when shifting to 2.0
        with open(filename, "r") as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(), delimiters=' \t')
            csvfile.seek(0)
            for line in csv.reader(csvfile, dialect):
                if line: records.append( (line[0], float(line[1])) )
            records.sort(key = lambda r: r[1])
        self.peptides, self.masses = zip(*records)

    def filter(self, precursor_mass, window = 3, ppm = False):
        """Return the sequence of the peptides that are within 'window'
        Daltons of the mass given: i.e., where the peptide's mass is within
        [precursor_mass - window, precursor_mass + window]

        Note: avgmass is what is reported on the neutral mass line, and
        that should be a more reasonable mass calculation since the precursor
        mass comes from MS1, which reflects the average mass.

        """
        if not ppm:
            l = bisect_left(self.masses, precursor_mass - window)
            h = bisect_right(self.masses, precursor_mass + window, lo = l)
        else:
            wl = float(window)*1e-6
            l = bisect(self.masses, precursor_mass * (1.0 - wl))
            h = bisect_right(self.masses, precursor_mass * (1.0 + wl), lo = l)
        return self.peptides[l:h]

class BenchmarkPeptideDB(object):
    """Load predigested Target and Decoy database for the benchmark data in the Lin et al paper:
    Combining High-Resolution and Exact Calibration To Boost Statistical Power: A Well-Calibrated Score Function for HighResolution MS2 Data

    example:
GNGYPI  848.4595        target  gi|296005319|ref|XP_002808988.1|,gi|225631876|emb|CAX64269.1|
STISTN  850.4599        target  gi|124511676|ref|XP_001348971.1|,gi|23498739|emb|CAD50809.1|,gi|74815317|sp|Q8IC32.1|Y7014_PLAF7
IFDDL   850.4639        target  gi|254832629|gb|AAN35752.2|,gi|258597259|ref|XP_001347839.2|
DFAAIS  851.4592        target  gi|124505807|ref|XP_001351017.1|,gi|23510660|emb|CAD49045.1|
IPPPIS  851.5319        target  gi|296004508|ref|XP_002808675.1|,gi|225631660|emb|CAX63946.1|
IPVPQA  852.5272        target  gi|296004528|ref|XP_002808685.1|,gi|225631670|emb|CAX63956.1|,gi|226700240|sp|Q8I3Y6.2|PFD6_PLAF7
GVASM[15.9949]F 855.4363        target  gi|23497404|gb|AAN36948.1|,gi|124809178|ref|XP_001348509.1|
    """
    def __init__(self, filename, target_db = True):
        self.filename = filename
        self.parse_targets = True
        self.peptides = [ ]
        self.masses = None
        self._parser(filename)

    def _parser(self, filename):
        records = [ ]
        if self.parse_target:
            lookForKind = 'target'
        else:
            lookForKind = 'decoy'
        # first detect what is the used delimiter; crux switched from a space to a tab when shifting to 2.0
        with open(filename, "r") as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(), delimiters=' \t')
            csvfile.seek(0)
            for line in csv.reader(csvfile, dialect):
                if line: 
                    kind = line[3]
                    if kind!=lookForKind:
                        continue

                    p = line[0]
                    var_mod_offsets = re.findall("[\[].*?[\]]", p)
                    p = re.sub("[\[].*?[\]]", "-", p)
                    var_offsets = []
                    ind = 0
                    wasMod = False
                    for aa in pep_sequence[1:]:
                        if wasMod: # skip to next residue
                            wasMod = False
                            continue
                        m = 0.0
                        if aa == '-':
                            m = float(var_mod_offsets[ind][1:-1])
                            ind += 1
                            wasMod = True
                        
                        var_offsets.append(m)
                    p = re.sub("-", "", p)
                    # deserialize: (0) mass (1) peptide sequence (2) var_mod_offsets
                    ma = float(line[1])
                    p = (ma, p, var_offsets)
                    records.append( (p, ma) )

            records.sort(key = lambda r: r[1])
        self.peptides, self.masses = zip(*records)

    def filter(self, precursor_mass, window = 3, ppm = False):
        """Return the sequence of the peptides that are within 'window'
        Daltons of the mass given: i.e., where the peptide's mass is within
        [precursor_mass - window, precursor_mass + window]

        Note: avgmass is what is reported on the neutral mass line, and
        that should be a more reasonable mass calculation since the precursor
        mass comes from MS1, which reflects the average mass.

        """
        if not ppm:
            l = bisect_left(self.masses, precursor_mass - window)
            h = bisect_right(self.masses, precursor_mass + window, lo = l)
        else:
            wl = float(window)*1e-6
            l = bisect(self.masses, precursor_mass * (1.0 - wl))
            h = bisect_right(self.masses, precursor_mass * (1.0 + wl), lo = l)
        return self.peptides[l:h]

class PeptideDB(object):
    def __init__(self, peptides):
        self.masses =  [ p[0] for p in peptides ]
        self.peptides = [ p for p in peptides ]

    def filter(self, precursor_mass, window = 3, ppm = False):
        """Return the sequence of the peptides that are within 'window'
        Daltons of the mass given: i.e., where the peptide's mass is within
        [precursor_mass - window, precursor_mass + window]
        """
        if not ppm:
            l = bisect_left(self.masses, precursor_mass - window)
            h = bisect_right(self.masses, precursor_mass + window, lo = l)
        else:
            wl = window*1e-6
            l = bisect_left(self.masses, precursor_mass / (1.0 + float(window)*0.000001))
            h = bisect_right(self.masses, precursor_mass / (1.0 - float(window)*0.000001), lo = l)
        return self.peptides[l:h]
