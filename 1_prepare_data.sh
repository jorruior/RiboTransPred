#!/bin/bash
#SBATCH --job-name=prepare_tracks
#SBATCH --output=logs/prepare_tracks_%j.out
#SBATCH --time=96:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# Author: Jorge Ruiz-Orera
# This script prepares RNA-seq and Ribo-seq tracks from BAM files specified in tracks.txt

set -e  # Exit on error
set -o pipefail  # Catch pipe errors

source ~/.bashrc
mamba activate translatomer

# Configuration
TRACKS_FILE=$1
OUTPUT_ROOT="tracks"
CPU="8"
NORM="RPKM"

# Save tmp files
export TMPDIR="/fast/AG_Huebner/Jorge/tmp/$SLURM_JOB_ID"
mkdir -p $TMPDIR

# Create output directories
mkdir -p "${OUTPUT_ROOT}"
mkdir -p logs

# Function to determine BAM type (rna or ribo) from filename
get_seq_type() {
    local filename=$1
    
    local basename=$(basename "$filename" .bam)
    local filename_lower=$(echo "$basename" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$filename_lower" == *"rna"* ]] || [[ "$filename_lower" == *"rnaseq"* ]]; then
        echo "rna"
    elif [[ "$filename_lower" == *"ribo"* ]] || [[ "$filename_lower" == *"riboseq"* ]]; then
        echo "ribo"
    else
        # Try to guess from common patterns
        if [[ "$filename" == *"RNA"* ]] || [[ "$filename" == *"/rna_"* ]] || [[ "$filename" == *"rna."* ]]; then
            echo "rna"
        elif [[ "$filename" == *"RIBO"* ]] || [[ "$filename" == *"/ribo_"* ]] || [[ "$filename" == *"ribo."* ]]; then
            echo "ribo"
        else
            echo "unknown"
        fi
    fi
}

# Function to generate coverage files from BAM
generate_coverage() {
    local input_bam=$1
    local output_prefix=$2
    local seq_type=$3  # "rna" or "ribo"
    local min_frag=${4:-0}
    local max_frag=${5:-0}
    
    # Create output directory if it doesn't exist
    mkdir -p $(dirname $output_prefix)
    
    echo "  Generating coverage for: $(basename $input_bam)"
    
    if [[ $seq_type == "rna" ]]; then
        # RNA-seq: single coverage track
        echo "    Creating RNA-seq coverage..."
        bamCoverage -p 4 -b $input_bam \
            --normalizeUsing $NORM \
            --binSize 1 \
            -o ${output_prefix}.bw
        
    elif [[ $seq_type == "ribo" ]]; then
        # Ribo-seq: two coverage tracks
        echo "    Creating Ribo-seq coverage..."
        # 1. Ribosome footprint coverage
        bamCoverage -p $CPU -b $input_bam \
            --normalizeUsing $NORM \
            --binSize 1 \
            --minFragmentLength $min_frag \
            --maxFragmentLength $max_frag \
            -o ${output_prefix}.bw
        
        echo "    Creating P-sites coverage..."
        # 2. P-sites coverage (with offset)
        bamCoverage -p $CPU -b $input_bam \
            --normalizeUsing $NORM \
            --Offset 12 \
            --binSize 1 \
            --minFragmentLength $min_frag \
            --maxFragmentLength $max_frag \
            -o ${output_prefix}.psites.bw
    fi
}

# Main processing function
process_entry() {
    local bam_file=$1
    local species=$2
    local tissue=$3
    local dataset=$4
    
    if [[ ! -f "$bam_file" ]]; then
        echo "Error: BAM file not found: $bam_file" >&2
        return 1
    fi
    
    # Determine sequence type
    local seq_type=$(get_seq_type "$bam_file")
    
    if [[ "$seq_type" == "unknown" ]]; then
        echo "Warning: Cannot determine sequence type for $bam_file" >&2
        echo "  Please ensure filename contains 'rna' or 'ribo'" >&2
        return 1
    fi
    
    # Create output directory structure: tracks/species/tissue/
    local output_dir="${OUTPUT_ROOT}/${species}/${tissue}"
    mkdir -p "$output_dir"
    
    # Create output prefix: species_tissue_rna/ribo
    local output_prefix="${output_dir}/${species}_${tissue}_${seq_type}"
    
    echo "Processing: $(basename $bam_file)"
    echo "  Species: $species, Tissue: $tissue, Type: $seq_type"
    echo "  Output directory: $output_dir"
    
    # Generate coverage files
    if [[ "$seq_type" == "rna" ]]; then
        generate_coverage "$bam_file" "$output_prefix" "rna"
        echo "  Created: ${species}_${tissue}_${seq_type}.bw"
    elif [[ "$seq_type" == "ribo" ]]; then
        generate_coverage "$bam_file" "$output_prefix" "ribo" 28 30
        echo "  Created: ${species}_${tissue}_${seq_type}.bw"
        echo "  Created: ${species}_${tissue}_${seq_type}.psites.bw"
    fi
    
    echo "  ✓ Completed"
    return 0
}

# Main processing loop
echo "========================================"
echo "Track Preparation Pipeline"
echo "========================================"
echo "Input file: $TRACKS_FILE"
echo "Output directory: $OUTPUT_ROOT/"
echo "========================================"

if [[ ! -f "$TRACKS_FILE" ]]; then
    echo "Error: $TRACKS_FILE not found!" >&2
    echo "Please create a tracks.txt file with format:" >&2
    echo "  <bam_file> <species> <tissue> <dataset>" >&2
    echo ""
    echo "Example:"
    echo "  /path/to/human_heart_rna.bam human heart training"
    echo "  /path/to/human_heart_ribo.bam human heart training"
    exit 1
fi

# Statistics
total_count=0
success_count=0
error_count=0

# Read and process each line from tracks.txt
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue
    
    # Remove leading/trailing whitespace
    line=$(echo "$line" | xargs)
    
    total_count=$((total_count + 1))
    
    # Split line into columns
    IFS=$' \t' read -r bam_file species tissue dataset <<< "$line"
    
    # Validate required fields
    if [[ -z "$bam_file" || -z "$species" || -z "$tissue" ]]; then
        echo "Warning: Missing required fields in line: $line" >&2
        error_count=$((error_count + 1))
        continue
    fi
    
    echo "----------------------------------------"
    echo "Entry $total_count:"
    
    if process_entry "$bam_file" "$species" "$tissue" "$dataset"; then
        success_count=$((success_count + 1))
    else
        error_count=$((error_count + 1))
    fi
    
done < "$TRACKS_FILE"

echo "========================================"
echo "PROCESSING SUMMARY"
echo "========================================"
echo "Total entries: $total_count"
echo "Successfully processed: $success_count"
echo "Errors: $error_count"
echo ""
echo "Output structure in ${OUTPUT_ROOT}/:"
echo "  species/"
echo "  └── tissue/"
echo "      ├── species_tissue_rna.bw"
echo "      ├── species_tissue_ribo.bw"
echo "      └── species_tissue_ribo.psites.bw"
echo ""

# Show generated files
echo "Generated files:"
find "${OUTPUT_ROOT}" -name "*.bw" -type f | while read file; do
    size=$(du -h "$file" 2>/dev/null | cut -f1)
    rel_path=${file#$OUTPUT_ROOT/}
    echo "  ${rel_path} (${size})"
done | sort

if [[ $success_count -eq 0 ]]; then
    echo ""
    echo "ERROR: No files were processed successfully!"
    exit 1
fi

echo ""
echo "========================================"
echo "Processing complete! ✨"
echo "All tracks saved in: ${OUTPUT_ROOT}/"
echo "========================================"
