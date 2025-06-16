#!/bin/bash
# Author: Jorge Ruiz-Orera
# Process genomes


### CONFIGURATION BLOCK ###

# Base directories
GENOMES_DIR="/fast/AG_Huebner/Jorge/GENOMES"
ANNOTATION_DIR="${GENOMES_DIR}/annotation"
FASTA_DIR="${GENOMES_DIR}/fasta"

# Output directory
OUTPUT_DIR="data/primate"
mkdir -p "${OUTPUT_DIR}"

### INPUT FILES ###

# Annotation files (Ensembl GTF format)
HUMAN_GTF="${ANNOTATION_DIR}/Homo_sapiens.GRCh38.98.sorted.gtf"
CHIMP_GTF="${ANNOTATION_DIR}/Pan_troglodytes.Pan_tro_3.0.98.sorted.gtf"
MACAQUE_GTF="${ANNOTATION_DIR}/Macaca_mulatta.Mmul_10.98.sorted.gtf"

# Transcript files (FASTA format, header should include the same "transcript_id" from the GTF)
HUMAN_TRANS="${ANNOTATION_DIR}/Homo_sapiens.GRCh38.cdna_and_ncrna.ENST.all.fa"
CHIMP_TRANS="${ANNOTATION_DIR}/Pan_troglodytes.Pan_tro_3.0.cdna_and_ncrna.ENST.all.fa"
MACAQUE_TRANS="${ANNOTATION_DIR}/Macaca_mulatta.Mmul_10.cdna_and_ncrna.ENST.all.fa"

# Genome FASTA files (FASTA format)
HUMAN_FASTA="${FASTA_DIR}/Homo_sapiens.GRCh38.dna.toplevel.fa"
CHIMP_FASTA="${FASTA_DIR}/Pan_troglodytes.Pan_tro_3.0.dna.toplevel.fa"
MACAQUE_FASTA="${FASTA_DIR}/Macaca_mulatta.Mmul_10.dna.toplevel.fa"

# FASTA index files (derived from genome FASTA, use samtools index)
HUMAN_FAI="${HUMAN_FASTA}.fai"
CHIMP_FAI="${CHIMP_FASTA}.fai"
MACAQUE_FAI="${MACAQUE_FASTA}.fai"


### FUNCTION DEFINITIONS ###

process_gtf() {
    local gtf_file=$1
    local species=$2
    local output_file=$3
    local prefix="_${species}"
    
    echo "Processing GTF for ${species}..."
    grep -P "\texon\t" "${gtf_file}" \
        | grep -E 'transcript_biotype "protein_coding"' \
        | awk -v prefix="${prefix}" '{print "chr"$1 prefix "\t"$4"\t"$5"\t"$7"\t"$14}' \
        | sed 's/"//; s/";//' >> "${output_file}"
}

process_fasta() {
    local fasta_file=$1
    local species=$2
    local output_file=$3
    
    echo "Processing FASTA for ${species}..."
    sed -e "s/ dna/_${species}\t/" -e 's/>/>chr/' \
        "${fasta_file}" \
        | awk '{print $1}' >> "${output_file}"
}

create_chrom_map() {
    local species=$1
    local input_file=$2
    local output_file=$3
    
    echo "Creating chromosome map for ${species}..."
    grep "${species}" "${input_file}" \
        | sed -e 's/>//' -e "s/_${species}//" -e 's/chr//' \
        | awk -v species="${species}" '{print $0"\tchr"$0"_"species}' > "${output_file}"
}

get_chrom_lengths() {
    local fai_file=$1
    local species=$2
    local output_file=$3
    
    echo "Getting chromosome lengths for ${species}..."
    awk -v species="${species}" '{print "chr"$1"_"species"\t"$2}' \
        "${fai_file}" >> "${output_file}"
}


### MAIN PROCESSING ###

### 1. Process GTF files to create BED file
echo "=== Processing GTF files ==="
> "${OUTPUT_DIR}/exon_region.bed"
process_gtf "${HUMAN_GTF}" "human" "${OUTPUT_DIR}/exon_region.bed"
process_gtf "${CHIMP_GTF}" "chimp" "${OUTPUT_DIR}/exon_region.bed"
process_gtf "${MACAQUE_GTF}" "macaque" "${OUTPUT_DIR}/exon_region.bed"

### 2. Process FASTA files to create combined FASTA
echo "=== Processing FASTA files ==="
> "${OUTPUT_DIR}/primates.fa"
process_fasta "${HUMAN_FASTA}" "human" "${OUTPUT_DIR}/primates.fa"
process_fasta "${CHIMP_FASTA}" "chimp" "${OUTPUT_DIR}/primates.fa"
process_fasta "${MACAQUE_FASTA}" "macaque" "${OUTPUT_DIR}/primates.fa"

### 3. Create chromosome maps
echo "=== Creating chromosome maps ==="
create_chrom_map "human" "${OUTPUT_DIR}/primates.fa" "${OUTPUT_DIR}/human_chrom_map.txt"
create_chrom_map "chimp" "${OUTPUT_DIR}/primates.fa" "${OUTPUT_DIR}/chimp_chrom_map.txt"
create_chrom_map "macaque" "${OUTPUT_DIR}/primates.fa" "${OUTPUT_DIR}/macaque_chrom_map.txt"

# Combined map
cat "${OUTPUT_DIR}"/*_chrom_map.txt > "${OUTPUT_DIR}/primate_chrom_map.txt"

### 4. Create chromosome length file
echo "=== Processing chromosome lengths ==="
> "${OUTPUT_DIR}/primate_chrom_lengths.txt"
get_chrom_lengths "${HUMAN_FAI}" "human" "${OUTPUT_DIR}/primate_chrom_lengths.txt"
get_chrom_lengths "${CHIMP_FAI}" "chimp" "${OUTPUT_DIR}/primate_chrom_lengths.txt"
get_chrom_lengths "${MACAQUE_FAI}" "macaque" "${OUTPUT_DIR}/primate_chrom_lengths.txt"

### 5. Create transcript FASTA file
echo "=== Processing transcript files ==="
cat "${HUMAN_TRANS}" "${CHIMP_TRANS}" "${MACAQUE_TRANS}" > "${OUTPUT_DIR}/transcripts.fa"

echo "=== Processing complete ==="