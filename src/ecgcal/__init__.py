"""ecgcal: multi-national ECG cohort handling and model training for the calibration study.

Separated from :mod:`recalib_kit` on purpose.  The toolbox is meant to be used
by people who have their own model and their own site, and it must not drag in
ECG-specific dependencies or assumptions.  Everything that knows what a lead or
an SCP code is lives here.
"""

__version__ = "0.1.0"
