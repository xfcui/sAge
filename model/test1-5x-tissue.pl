#!/usr/bin/env perl

use strict;
use warnings;
use FindBin qw($Bin);

# Backward-compatible wrapper. New workflows should call
# run_cross_validation.py directly.
my @command = ('python3', "$Bin/../run_cross_validation.py", @ARGV);
system(@command) == 0 or die "Cross-validation runner failed: $?\n";
