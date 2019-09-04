#!/bin/bash
#
# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2018 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0

# adjust the number of threads to be used for all tests in the variable below
NUMTHREADS=15

#################################################################
################## Low-resolution MS2 searches
#################################################################

############################################
######### Didea training and testing
############################################
function trainDidea {
    CHARGE="2"
    MS2=data/riptideTrainingData/strict-orbitrap.ms2
    PSMS=data/riptideTrainingData/strict-orbitrap.psm
    OUTPUT="charge${CHARGE}-lambdas.txt"

    echo "Searching spectra"
    time python -OO dideaTrain.py \
	--lmb0Prior 0.383435 \
	--charge $CHARGE \
    	--spectra $MS2 \
    	--output $OUTPUT \
	--psms $PSMS

    CHARGE="3"
    MS2=data/riptideTrainingData/strict-orbitrap-ch3.ms2
    PSMS=data/riptideTrainingData/strict-orbitrap-ch3.psm
    OUTPUT="charge${CHARGE}-lambdas.txt"

    echo "Searching spectra"
    time python -OO dideaTrain.py \
	--lmb0Prior 0.367028 \
	--charge $CHARGE \
    	--spectra $MS2 \
    	--output $OUTPUT \
	--psms $PSMS
}

function testDidea {
    echo "Digesting protein database"
    python -OO digest.py \
    	--min-length 6 \
    	--fasta data/yeast.fasta \
    	--enzyme 'trypsin/p' \
    	--monoisotopic-precursor true \
    	--missed-cleavages 0 \
    	--digestion 'full-digest'

    CH2="charge2-lambdas.txt"
    CH3="charge3-lambdas.txt"

    org=yeast
    echo "Searching spectra"
    time python -OO dideaSearch.py \
    	--digest-dir 'digest-output' \
    	--precursor-window 3.0 \
    	--top-match 1 \
	--charges all \
    	--spectra data/test.ms2 \
    	--output dideaSearch-test-output \
	--learned-lambdas-ch2 $CH2 \
	--learned-lambdas-ch3 $CH3 \
	--num-threads 1
}

############################################
######### train and search
############################################
function trainTest {
    echo "Training DRIP"
    python -OO dripTrain.py \
    	--psm-library data/riptideTrainingData/strict-orbitrap.psm \
    	--spectra data/riptideTrainingData/strict-orbitrap.ms2 \
    	--output-mean-file dripLearned.means \
    	--output-covar-file dripLearned.covars \
    	--mods-spec 'C+57.0214'

    echo "Digesting protein database"
    python -OO digest.py \
    	--min-length 6 \
    	--fasta data/yeast.fasta \
    	--enzyme 'trypsin/p' \
    	--monoisotopic-precursor true \
    	--missed-cleavages 0 \
    	--digestion 'full-digest'

    echo "Searching spectra"
    time python -OO dripSearch.py \
    	--digest-dir 'digest-output' \
    	--precursor-window 3.0 \
    	--learned-means dripLearned.means \
    	--learned-covars dripLearned.covars \
	--num-threads $NUMTHREADS \
    	--top-match 1 \
	--charges 2 \
    	--spectra data/test.ms2 \
    	--output dripSearch-test-output
}

############################################
######### test using beam pruning
############################################
function testBeam {
    BEAM=75
    echo "Digesting protein database"
    python -OO digest.py \
    	--min-length 6 \
    	--fasta data/yeast.fasta \
    	--enzyme 'trypsin/p' \
    	--monoisotopic-precursor true \
    	--missed-cleavages 0 \
    	--digestion 'full-digest'

    time python -OO dripSearch.py \
    	--digest-dir 'digest-output' \
    	--precursor-window 3.0 \
    	--learned-means dripLearned.means \
    	--learned-covars dripLearned.covars \
	--num-threads $NUMTHREADS \
    	--top-match 1 \
    	--beam $BEAM \
    	--spectra data/test.ms2 \
    	--output dripSearch-test-output-beam$BEAM
}

###################################################
######### Extract features for output of trainTest,
######### write output to PIN file format
###################################################
function dripExtractLowRes {
    python -OO dripExtract.py \
	--write-pin true \
    	--learned-means dripLearned.means \
    	--learned-covars dripLearned.covars \
	--psm-file dripSearch-test-output.txt \
	--num-threads $NUMTHREADS \
	--mods-spec 'C+57.0214' \
	--spectra data/test.ms2 \
	--output dripExtract-test-output.txt
}

############################################
######### train and search ch3 PSMs
############################################
function trainTestCh3 {
    echo "Training DRIP"
    python -OO dripTrain.py \
    	--psm-library data/riptideTrainingData/strict-orbitrap-ch3.psm \
    	--spectra data/riptideTrainingData/strict-orbitrap-ch3.ms2 \
    	--output-mean-file dripLearned-ch3.means \
    	--output-covar-file dripLearned-ch3.covars \
    	--mods-spec 'C+57.0214'

    echo "Digesting protein database"
    python -OO digest.py \
    	--min-length 6 \
    	--fasta data/yeast.fasta \
    	--enzyme 'trypsin/p' \
    	--monoisotopic-precursor true \
    	--missed-cleavages 0 \
    	--digestion 'full-digest'

    echo "Searching spectra"
    time python -OO dripSearch.py \
    	--digest-dir 'digest-output' \
    	--precursor-window 3.0 \
    	--learned-means dripLearned-ch3.means \
    	--learned-covars dripLearned-ch3.covars \
    	--charges 3 \
	--num-threads $NUMTHREADS \
    	--top-match 1 \
    	--spectra data/test.ms2 \
    	--output dripSearch-test-output
}

#########################################
######### train, search, and find max PSM
######### over different charge states
#########################################
function trainTestRecalibrate {
    echo "Training DRIP"
    python -OO dripTrain.py \
    	--psm-library data/riptideTrainingData/strict-orbitrap.psm \
    	--spectra data/riptideTrainingData/strict-orbitrap.ms2 \
    	--output-mean-file dripLearned.means \
    	--output-covar-file dripLearned.covars \
    	--mods-spec 'C+57.0214'

    echo "Digesting protein database"
    python -OO digest.py \
    	--recalibrate True \
    	--min-length 6 \
    	--fasta data/yeast.fasta \
    	--enzyme 'trypsin/p' \
    	--monoisotopic-precursor true \
    	--missed-cleavages 0 \
    	--digestion 'full-digest'

    echo "Searching spectra"
    python -OO dripSearch.py \
    	--digest-dir 'digest-output' \
    	--precursor-window 3.0 \
    	--learned-means dripLearned.means \
    	--learned-covars dripLearned.covars \
	--num-threads $NUMTHREADS \
    	--top-match 1 \
    	--spectra data/test.ms2 \
    	--output dripSearch-test-output
}

############################################
######### search high-res MS2 spectra
############################################
function dripSearchHighres {
    # digest directory
    ./digest.py  \
    	--fasta data/plasmo_Pfalciparum3D7_NCBI.fasta \
    	--min-length 7 \
    	--custom-enzyme '[K]|[X]' \
    	--mods-spec 'C+57.0214,K+229.16293' \
    	--nterm-peptide-mods-spec 'X+229.16293' \
    	--monoisotopic-precursor true \
    	--recalibrate True \
    	--peptide-buffer 1000 \
    	--decoys True

    python -OO dripSearch.py \
	--digest-dir 'digest-output' \
	--precursor-window 50 \
	--num-threads $NUMTHREADS \
	--high-res-ms2 true \
	--precursor-window-type 'ppm' \
	--precursor-filter 'True' \
	--spectra data/malariaTest.ms2 \
	--output dripSearch-smallMalariaTest-output
}

############################################
######### search high-res MS2 spectra with 
######### variable mods
############################################
function dripSearchHighresVarMods {
    # digest directory
    ./digest.py  \
    	--fasta data/plasmo_Pfalciparum3D7_NCBI.fasta \
    	--min-length 7 \
    	--custom-enzyme '[K]|[X]' \
    	--mods-spec '3M+15.9949,C+57.0214,K+229.16293' \
    	--nterm-peptide-mods-spec 'X+229.16293' \
    	--monoisotopic-precursor true \
    	--recalibrate True \
    	--decoys True

    python -OO dripSearch.py \
	--digest-dir 'digest-output' \
	--precursor-window 50 \
	--num-threads $NUMTHREADS \
	--high-res-ms2 true \
	--precursor-window-type 'ppm' \
	--precursor-filter 'True' \
	--spectra data/malariaTest.ms2 \
	--output dripSearch-malariaTestVarmods-output
}

############################################
######### extract features for high-res MS2,
######### output of dripSearchHighresVarMods
############################################
function dripExtractHighResVarMods {
    python -OO dripExtract.py \
	--append-to-pin false \
	--high-res-ms2 true \
	--precursor-filter 'True' \
    	--learned-means dripLearned.means \
    	--learned-covars dripLearned.covars \
	--psm-file dripSearch-malariaTestVarmods-output.txt \
	--num-threads $NUMTHREADS \
    	--mods-spec '3M+15.9949,C+57.0214,K+229.16293' \
    	--nterm-peptide-mods-spec 'X+229.16293' \
	--spectra data/malariaTest.ms2 \
	--output dripExtract-malariaTestVarmods-output.txt
}

#############################################
######### Split data for cluster use, run 
######### individual jobs to simulate cluster
######### use, collect and merge results
#############################################
function clusterTest {
    if [ ! $DRIPTOOLKIT ]
    then
	echo "Please set DRIPTOOLKIT environment variable a directory containing the built toolkit."
    else

	echo "Digesting protein database"
	python -OO digest.py \
    	    --min-length 6 \
    	    --fasta data/yeast.fasta \
    	    --enzyme 'trypsin/p' \
    	    --monoisotopic-precursor true \
    	    --missed-cleavages 0 \
    	    --digestion 'full-digest'

	echo "Creating cluster jobs"
	python -OO dripSearch.py \
    	    --digest-dir 'digest-output' \
    	    --precursor-window 3.0 \
    	    --learned-means dripLearned.means \
    	    --learned-covars dripLearned.covars \
	    --num-threads $NUMTHREADS \
	    --num-cluster-jobs 4 \
    	    --top-match 1 \
    	    --spectra data/test.ms2 \
    	    --output dripSearch-clusterTest-output

	echo "Running created jobs (locally)"
	for j in encode/*.sh
	do
    	    echo $j
    	    ./$j
	done

	echo "Merging cluster results"
	python -OO dripSearch.py --merge-cluster-results True \
    	    --logDir log \
    	    --output dripSearch-clusterTest-output
    fi
}

function dideaSearchHighresVarMods {
    # digest directory
    # ./digest.py  \
    # 	--fasta data/plasmo_Pfalciparum3D7_NCBI.fasta \
    # 	--min-length 7 \
    # 	--custom-enzyme '[K]|[X]' \
    # 	--mods-spec '3M+15.9949,C+57.0214,K+229.16293' \
    # 	--nterm-peptide-mods-spec 'X+229.16293' \
    # 	--monoisotopic-precursor true \
    # 	--recalibrate True \
    # 	--decoys True

    CH2="charge2-lambdas.txt"
    CH3="charge3-lambdas.txt"

    python -OO dideaSearch.py \
	--digest-dir 'digest-output' \
	--num-threads 1 \
	--spectra data/malariaTest.ms2 \
	--learned-lambdas-ch2 $CH2 \
	--learned-lambdas-ch3 $CH3 \
	--precursor-window 50 \
	--precursor-window-type 'ppm' \
	--output dideaSearch-malariaTestVarmods-output
	# --high-res-ms2 true \
	# --precursor-filter 'True' \

}

function dripDideaSearchHighres {
    # digest directory
    ./digest.py  \
    	--fasta data/plasmo_Pfalciparum3D7_NCBI.fasta \
    	--min-length 7 \
    	--custom-enzyme '[K]|[X]' \
    	--mods-spec 'C+57.0214,K+229.16293' \
    	--nterm-peptide-mods-spec 'X+229.16293' \
    	--monoisotopic-precursor true \
    	--recalibrate True \
    	--peptide-buffer 1000 \
    	--decoys True

    time python -OO dripSearch.py \
	--digest-dir 'digest-output' \
	--precursor-window 50 \
	--num-threads $NUMTHREADS \
	--high-res-ms2 true \
	--precursor-window-type 'ppm' \
	--precursor-filter 'True' \
	--spectra data/malariaTest.ms2 \
	--output dripSearch-smallMalariaTest-output

    CH2="charge2-lambdas.txt"
    CH3="charge3-lambdas.txt"

    time python -OO dideaSearch.py \
	--digest-dir 'digest-output' \
	--precursor-window 50 \
	--num-threads $NUMTHREADS \
	--high-res-ms2 true \
	--precursor-window-type 'ppm' \
	--spectra data/malariaTest.ms2 \
	--output dideaSearch-smallMalariaTest-output \
	--learned-lambdas-ch2 $CH2 \
	--learned-lambdas-ch3 $CH3 \
	--num-threads 1

}

function testRecalibrate {
    # echo "Digesting protein database"
    # python -OO digest.py \
    # 	--recalibrate True \
    # 	--min-length 6 \
    # 	--fasta data/yeast.fasta \
    # 	--enzyme 'trypsin/p' \
    # 	--monoisotopic-precursor true \
    # 	--missed-cleavages 0 \
    # 	--digestion 'full-digest' \
    # 	--digest-dir 'yeastDigest'

    echo "Searching spectra"
    python -OO dripSearch.py \
    	--digest-dir 'yeastDigest' \
    	--precursor-window 3.0 \
    	--learned-means dripLearned.means \
    	--learned-covars dripLearned.covars \
	--num-threads $NUMTHREADS \
    	--top-match 1 \
    	--spectra data/test.ms2 \
    	--output dripSearch-test-output
}

# available examples (see function above for description): 
# trainTest
# testBeam
# dripExtractLowRes (run trainTest first)
# trainTestCh3
# trainTestRecalibrate
# dripSearchHighres
# dripSearchHighresVarMods
# dripExtractHighResVarMods (run dripSearchHighresVarMods first)
# clusterTest

# # traing and test low-res MS2
# trainTest

# # run several tests
# runTests=( dripSearchHighres \
#                dripSearchHighresVarMods \
#                dripExtractHighResVarMods )

# # loop through array of tests
# for dripTest in ${runTests[@]}
# do
#     echo $dripTest
#     $dripTest
# done

#####  Didea
# trainDidea
# testDidea

# dideaSearchHighresVarMods
# dripDideaSearchHighres
testRecalibrate
