# Written by John Halloran <halloj3@uw.washington.edu>
#
# Copyright (C) 2016 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0

allPeps = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')-set('JOBZUX')

# amino acid masses
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


mass_h = 1.00782503207
# mass_proton = 1.00727646677
mass_proton = 1.00782503207
mass_neutron = 1.00866491600

mass_h_mono = 1.0078246
mass_h_avg = 1.00794
mass_nh3 = 17.02655
mass_nh3_avg = 17.03056
mass_h2o = 18.010564684
mass_h2o_avg = 18.0153
mass_co = 27.9949
mass_co_avg = 28.0101

# Didea specific constants
max_mz = 2000
