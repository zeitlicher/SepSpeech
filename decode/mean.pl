#!/usr/bin/perl

$correct = $subs = $dels = $ins =0;

while(<STDIN>){
    if (/^Scores:/){
	chomp;
	/(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/; # C S D I
	$correct += $1;
	$subs += $2;
	$dels += $3;
	$ins += $4;
    }
}

$char_err_rate = 100.0 * ($subs+$dels+$ins)/($correct+$subs+$dels);
printf "%.3f\n", $char_err_rate;
