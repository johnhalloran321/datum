# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2018 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0
# Command line parsing utilities.

def simple_uniform_binwidth(min_mass, num_bins, offset = 0.0, bin_width = 1.0011413):
    tick_points = [ ]
    last = min_mass + offset
    while len(tick_points) < num_bins:
        tick_points.append( (last, last + bin_width) )
        last = last + bin_width
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
