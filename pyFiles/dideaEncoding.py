# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2018 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0
# Command line parsing utilities.

from constants import mass_h, mass_h2o, max_mz
import math

# Define general emission function
def genVe(theta, s):
    # quadratic
    return 0.5 * (theta * s) ** 2 + 1.0

def genVe_grad(theta, s):
    # gradient of quadratic
    return theta * s * s

# Convex virtual emission function, from the NeurIPS 2018 paper:
# Halloran, John T., and David M. Rocke. "Learning concave conditional likelihood models for improved analysis of tandem mass spectra." Advances in Neural Information Processing Systems. 2018.

def cve0(theta, s):
    # from NeurIPS 2018 paper:
    return math.exp(theta * s)

def cve_grad0(theta, s):
    # from NeurIPS 2018 paper:
    return s * math.exp(theta * s)

def simple_uniform_binwidth(min_mass, num_bins, bin_width = 1.0005079, offset = 0.0):
    tick_points = [ ]
    last = min_mass + offset
    while len(tick_points) < num_bins:
        tick_points.append( (last, last + bin_width) )
        last += bin_width
    return tick_points

def histogram_spectra(spectrum, ranges, op = max, normalize = False):
    """Convert a spectrum into histogram bins.

    Args:
        spectrum:   Instance of MS2Spectrum.
        ranges:     List of sorted, [low, high] m/z ranges for each bin.
        op:         Takes a set of spectra intensities, and converts them
                    to one float.
        normalize:  If true, normalize the bin heights to be probabilities.

    Returns:
        A list of non-negative reals, one for each range, where the value
        corresponds to the maximum intensity of any point in the m/z range.
        Any points in the spectrum not covered by one of the ranges is ignored,
        i.e., has no influence on the output.

    """
    pairs = zip(spectrum.mz, spectrum.intensity)

    # Sort the ranges in order. Sort the pairs in order of increasing m/z value
    intensities = [ [] ] * len(ranges)
    sort_pred = lambda x: x[0]
    pairs.sort(key = sort_pred)
    ranges.sort(key = sort_pred)

    # Check that ranges don't overlap (allowing for end[0] == start[1], etc.)
    for i in range(0, len(ranges)-1):
        assert(ranges[i+1][0] >= ranges[i][1])

    # Linear sweep over the ranges and pairs, placing each pair in its bin.
    p = 0
    bins = [ ]
    lengths = [ ]
    for low, high in ranges:
       intensities = [ ]
       while p < len(pairs) and pairs[p][0] < high:
           if pairs[p][0] >= low and pairs[p][0] < high:
               intensities.append( pairs[p][1] )
           p += 1
       if intensities:
           lengths.append( len(intensities) )
           bins.append( op(intensities) )
       else:
           bins.append( 0 )
    if normalize:
        bins = [ float(b) / sum(bins) for b in bins ]

    assert(len(bins) == len(ranges))

    return bins

def bins_to_vecpt_ratios(bins, kind = 'expbased', lmb = 0.5):
    """Process observed intensities
    """
    ratios = [ None ] * len(bins)

    if kind == 'expbased':
        # Sophisticated VE function from the UAI 2012 paper
        l = lmb
        shift = 1 - l * math.exp(-l)
        el = math.exp(l)
        for i, bi in enumerate(bins):
            ratios[i] = math.log(el - l + l * math.exp(l * bi)) - math.log(2*el - 1 - l)
            # ratios[i] = math.log(el - l + l * math.exp(l * bi))
    elif kind == 'nonexp':
        l = lmb
        shift = 1 - l * math.exp(-l)
        el = math.exp(l)
        for i, bi in enumerate(bins):
            ratios[i] = math.exp(math.log(el - l + l * math.exp(l * bi)) - math.log(2*el - 1 - l))
    elif kind == 'intensity':
        # f(S) = S, called Didea-I in the UAI 2012 paper
        for i, bi in enumerate(bins):
            ratios[i] = bi
    elif kind == 'modslope':
        # f(S) = 0.2S, to mimic 'expbased' with a simpler function. Works well
        for i, bi in enumerate(bins):
            ratios[i] = 0.2 * bi
    else:
        raise ValueError('Bad kind: %s' % kind)

    assert(all(r != None for r in ratios))
    assert(all(not math.isinf(r) and not math.isnan(r) for r in ratios))
    assert len(ratios) == len(bins)

    ratios[0] = 0.0
    ratios[-1] = 0.0

    return ratios

# theoretical spectrum functions
def round_op(p_mass, denom, tauCard, rMax):
    return min(int(math.floor(p_mass/denom)) + tauCard, rMax)
    # return min(int(round(p_mass/denom)) + tauCard, rMax)

def peptide_mod_offset_screen(p, mods = {}, ntermMods = {}, ctermMods = {}):
    """ Given (static) mods, calculate what the offsets should occur for the b-/y-ions
        p - peptide string
        Note: the screen returns the offset for each b- and y-ion, traversing the peptide
              from left to right.  For y-ions, includes mass(h2o)
    """ 
    boffset = 0.
    yoffset = mass_h2o
    b_offsets = []
    y_offsets = []
    if p[0] in ntermMods:
        boffset += ntermMods[p[0]]
    if p[-1] in ctermMods:
        yoffset += ctermMods[p[-1]]
    # reverse peptide sequence to calculate y-ion offset linearly, in reverse
    for aaB,aaY in zip(p[:-1], reversed(p[1:])):
        if aaB in mods:
            boffset += mods[aaB]
        if aaY in mods:
            yoffset += mods[aaY]
        b_offsets.append(boffset)
        y_offsets.append(yoffset)
    
    # reverse y-ions offsets back to proper order
    y_offsets.reverse()

    return b_offsets, y_offsets

def peptide_var_mod_offset_screen(p, mods = {}, ntermMods = {}, ctermMods = {}, 
                                  varMods = {}, ntermVarMods = {}, ctermVarMods = {},
                                  varModSequence = []):
    """ Given (variable) mods, calculate what the offsets should occur for the b-/y-ions
        p - peptide string
        Note: the screen returns the offset for each b- and y-ion, traversing the peptide
              from left to right.  For y-ions, includes mass(h2o)
    """ 
    boffset = 0.
    yoffset = mass_h2o
    b_offsets = []
    y_offsets = []
    # check n-/c-term amino acids for modifications
    if p[0] in ntermMods:
        boffset = ntermMods[p[0]]
    elif p[0] in ntermVarMods:
        if varModSequence[0] == '2': # denotes an nterm variable modification
            boffset = ntermVarMods[p[0]][1]
    if p[-1] in ctermMods:
        yoffset = ctermMods[p[-1]]
    elif p[-1] in ctermVarMods:
        if varModSequence[-1] == '3': # denotes a cterm variable modification
            yoffset = ctermVarMods[p[-1]][1]

    # reverse peptide sequence to calculate y-ion offset linearly, in reverse
    for aaB,aaY in zip(p[:-1], reversed(p[1:])):
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

        b_offsets.append(boffset)
        y_offsets.append(yoffset)
    
    # reverse y-ions offsets back to proper order
    y_offsets.reverse()

    return b_offsets, y_offsets


def by_sepTauShift(ntm, ctm, charge, b_offsets, y_offsets, 
                   lastBin = 1999, tauCard = 75,bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    # iterate through possible charges
    bSeq = []
    ySeq = []
    for b, y, boffset, yoffset in zip(ntm[1:-1], ctm[1:-1], b_offsets, y_offsets):
        bs = []
        ys = []
        for c in range(1,charge):
            cf = float(c)
            denom = float(c) * float(bin_width)
            c_boffset = cf*mass_h + boffset
            c_yoffset = cf*mass_h + yoffset
            bs.append(round_op(b+c_boffset, denom, tauCard, rMax))
            ys.append(round_op(y+c_yoffset, denom, tauCard, rMax))
        bSeq.append(bs)
        ySeq.append(ys)
    return bSeq, ySeq


def by_tauShift(ntm, ctm, charge, b_offsets, y_offsets, 
                lastBin = 1999, tauCard = 75, bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    tauMin = (tauCard - 1 ) / 2
    # calculate tau-radius around bin indices
    rMax = lastBin + tauCard + tauMin
    nterm_fragments = []
    cterm_fragments = []
    # iterate through possible charges
    for c in range(1,charge):
        cf = float(c)
        denom = cf * float(bin_width)
        for b, y, boffset, yoffset in zip(ntm[1:-1], ctm[1:-1], b_offsets, y_offsets):
            nterm_fragments.append(round_op(b+boffset+cf*mass_h,denom, tauCard, rMax))
            cterm_fragments.append(round_op(y+yoffset+cf*mass_h,denom, tauCard, rMax))
    return (nterm_fragments,cterm_fragments)

def round_op_noshift(p_mass, denom):
    return int(p_mass/denom)

def byIons(ntm, ctm, charge, b_offsets, y_offsets, 
           lastBin = 1999, bin_width = 1.):
    """Given peptide and charge, return b- and y-ions in seperate vectors

    """
    nterm_fragments = []
    cterm_fragments = []
    # iterate through possible charges
    for c in range(1,charge):
        cf = float(c)
        denom = cf * float(bin_width)
        for b, y, boffset, yoffset in zip(ntm[1:-1], ctm[1:-1], b_offsets, y_offsets):
            nterm_fragments.append(round_op_noshift(b+boffset+cf*mass_h,denom))
            cterm_fragments.append(round_op_noshift(y+yoffset+cf*mass_h,denom))
    return (nterm_fragments,cterm_fragments)
