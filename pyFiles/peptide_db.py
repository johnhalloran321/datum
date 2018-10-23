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

# class SimplePeptideDB(object):
#     """A simpler version of a peptide database that takes the peptides.txt
#     file produces by crux create-index --peptide-list T <fasta-file> <index-dir>

#     The peptide database is a text file where each line is of the form
#     <peptide> <neutral mass>

#     Where peptide is a IUPAC sequence, and neutral mass it the mass
#     of a peptide (c.f.,
#     http://noble.gs.washington.edu/proj/crux/crux-search-for-matches.html)

#     """
#     def __init__(self, peptides):
#         peptides = [peptides[pep] for pep in peptides]
#         peptides.sort(key = lambda x: x.peptideMass)
#         self.peptides = [ p.peptide.seq for p in peptides ]
#         self.masses =  [ p.peptideMass for p in peptides ]

#     def filter(self, precursor_mass, window = 3, ppm = False):
#         """Return the sequence of the peptides that are within 'window'
#         Daltons of the mass given: i.e., where the peptide's mass is within
#         [precursor_mass - window, precursor_mass + window]
#         """
#         if not ppm:
#             l = bisect_left(self.masses, precursor_mass - window)
#             h = bisect_right(self.masses, precursor_mass + window, lo = l)
#         else:
#             wl = window*1e-6
#             l = bisect_left(self.masses, precursor_mass / (1.0 + float(window)*0.000001))
#             h = bisect_right(self.masses, precursor_mass / (1.0 - float(window)*0.000001), lo = l)
#         return self.peptides[l:h]

class PeptideDB(object):
    def __init__(self, peptides):
        # peptides.sort(key = lambda x: x[0])
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
