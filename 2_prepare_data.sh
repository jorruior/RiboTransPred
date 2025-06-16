#!/bin/bash
# Author: Jorge Ruiz-Orera
# This script prepares the main RNA-seq and Ribo-seq data for the model.

# Activate the conda environment
source ~/.bashrc
mamba activate translatomer

### CONFIGURATION BLOCK ###
INPUT_DIR="data/primate/"
LENGTH="6000" #Transcript bp to be analyzed from 5'
NBINS="1000" #length of bins (should be divisible by length)
MIN_PSITES_LEN="28" #Minimum ribosome fragment length to consider
MAX_PSITES_LEN="30" #Maximum ribosome fragment length to consider

### INPUT FILES ###

# Explicitly declare all input BAM files
declare -A RNA_BAM_FILES=(
    ["human/heart"]="samples/human_lv_test/rna_pooled.bam"
    ["human/ipsc_cm"]="samples/human_cm_test/rna_pooled.bam"
    ["chimp/heart"]="samples/chimp_lv_test/rna_pooled.bam"
    ["chimp/ipsc_cm"]="samples/chimp_cm_test/rna_pooled.bam"
    ["macaque/ipsc_cm"]="samples/macaque_cm_test/rna_pooled.bam"
    ["macaque/heart"]="samples/macaque_lv_test/rna_pooled.bam"
)

declare -A RIBO_BAM_FILES=(
    ["human/heart"]="samples/human_lv_test/ribo_pooled.bam"
    ["human/ipsc_cm"]="samples/human_cm_test/ribo_pooled.bam"
    ["chimp/heart"]="samples/chimp_lv_test/ribo_pooled.bam"
    ["chimp/ipsc_cm"]="samples/chimp_cm_test/ribo_pooled.bam"
    ["macaque/ipsc_cm"]="samples/macaque_cm_test/ribo_pooled.bam"
    ["macaque/heart"]="samples/macaque_lv_test/ribo_pooled.bam"
)

# Chromosome map files (already generated in step 1)
declare -A CHROM_MAPS=(
    ["human"]="${INPUT_DIR}human_chrom_map.txt"
    ["chimp"]="${INPUT_DIR}chimp_chrom_map.txt"
    ["macaque"]="${INPUT_DIR}macaque_chrom_map.txt"
)

### FUNCTION DEFINITIONS ###
process_bam() {
    local input_bam=$1
    local chrom_map=$2
    local output_prefix=$3
    local min_frag=${4:-0}
    local max_frag=${5:-0}
    
    # Create output directory if it doesn't exist
    mkdir -p $(dirname $output_prefix)
    
    # Process header and reheader BAM
    echo "Processing BAM file: $input_bam"
    if [ ! -f "$input_bam" ]; then
        echo "Error: Input BAM file not found: $input_bam"
        return 1
    fi
    
    samtools view -H $input_bam > ${output_prefix}_header.sam
    sed -f <(awk '{print "s/SN:"$1"\t/SN:"$2"\t/"}' $chrom_map) ${output_prefix}_header.sam > ${output_prefix}_new_header.sam
    samtools reheader ${output_prefix}_new_header.sam $input_bam > ${output_prefix}_reheader.bam
    samtools index ${output_prefix}_reheader.bam
    
    # Generate coverage files
    if [ $min_frag -eq 0 ] && [ $max_frag -eq 0 ]; then
        echo "Creating RNA-seq coverage (no fragment size filtering)"
        bamCoverage -b ${output_prefix}_reheader.bam --normalizeUsing RPKM --binSize 1 -o ${output_prefix}.normalized.bw
    else
        echo "Creating Ribo-seq coverage (fragment size $min_frag-$max_frag)"
        bamCoverage -b ${output_prefix}_reheader.bam --normalizeUsing RPKM --binSize 1 --minFragmentLength $min_frag --maxFragmentLength $max_frag -o ${output_prefix}.normalized.bw
    fi
    
    # Clean up temporary files
    rm ${output_prefix}_header.sam ${output_prefix}_new_header.sam ${output_prefix}_reheader.bam ${output_prefix}_reheader.bam.bai
}

### MAIN PROCESSING ###
for key in "${!RNA_BAM_FILES[@]}"; do
    IFS='/' read -r species tissue <<< "$key"
    chrom_map=${CHROM_MAPS[$species]}
    
    echo "========================================"
    echo "Processing $species $tissue..."
    
    # Process RNA-seq
    if [ -n "${RNA_BAM_FILES[$key]}" ]; then
        echo "Processing RNA-seq data..."
        process_bam "${RNA_BAM_FILES[$key]}" "$chrom_map" "${INPUT_DIR}${tissue}/${species}/rnaseq"
    fi
    
    # Process Ribo-seq
    if [ -n "${RIBO_BAM_FILES[$key]}" ]; then
        echo "Processing Ribo-seq data..."
        # Use different fragment sizes for ribo-seq
        process_bam "${RIBO_BAM_FILES[$key]}" "$chrom_map" "${INPUT_DIR}${tissue}/${species}/riboseq" $MIN_PSITES_LEN $MAX_PSITES_LEN
    fi
done

# Generate file lists for all tissues
echo "Generating file lists..."
ls -lt ${INPUT_DIR}/*/*/riboseq.normalized.bw | \
        awk '{print $9}' | \
        sed "s#${INPUT_DIR}##" | \
        sed "s#/#\t#g" | \
        awk '{print $1"\t"$2}' > ${INPUT_DIR}heart.txt

# Generate features
echo "Generating features..."
if [ -f "${INPUT_DIR}heart.txt" ]; then
    bash features.sh "${INPUT_DIR}heart.txt" $LENGTH $NBINS primate
fi

echo "Processing complete!"
