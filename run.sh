#!/bin/bash

python v2cusum.py ../synthetics/high_high.dat > ../synthetics/mcusum_hh.eval &
python v2cusum.py ../synthetics/low_high.dat > ../synthetics/mcusum_lh.eval &
python v2cusum.py ../synthetics/low_low.dat > ../synthetics/mcusum_ll.eval &
python v2cusum.py ../synthetics/high_low.dat > ../synthetics/mcusum_hl.eval &
