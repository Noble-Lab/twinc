"""
twinc_utils.py
Author: Anupama Jha <anupamaj@uw.edu>
"""

import torch
import numpy as np
import pandas as pd


def gc_predictor(seq1, seq2, is_torch=False):
    """
    A predictor based on GC content in the first
    and second sequences.
    :param seq1: np.array, one-hot-encoded sequence
    :param seq2: np.array, one-hot-encoded sequence
    :param is_torch: are the inputs torch tensors
    :return: gc_content in sequences, the difference in GC content between sequences
    """
    if is_torch:
        seq1_argmax = torch.argmax(seq1, axis=1)
        seq2_argmax = torch.argmax(seq2, axis=1)

        seq1_c = torch.where(seq1_argmax == 1.0, 1.0, 0.0)
        seq1_g = torch.where(seq1_argmax == 2.0, 1.0, 0.0)

        seq1_gc = seq1_c + seq1_g

        seq2_c = torch.where(seq2_argmax == 1.0, 1.0, 0.0)
        seq2_g = torch.where(seq2_argmax == 2.0, 1.0, 0.0)

        seq2_gc = seq2_c + seq2_g

        gc_count = torch.count_nonzero(seq1_gc, axis=1) + torch.count_nonzero(seq2_gc, axis=1)
        gc_content = gc_count / (seq1_gc.shape[1] + seq2_gc.shape[1])
        gc_content = gc_content[:, None]

        gc_count_diff = torch.abs(torch.count_nonzero(seq1_gc, axis=1) - torch.count_nonzero(seq2_gc, axis=1))

        gc_content_diff = gc_count_diff / (seq1_gc.shape[1])
        gc_content_diff = gc_content_diff[:, None]

    else:
        seq1_argmax = np.argmax(seq1, axis=1)
        seq2_argmax = np.argmax(seq2, axis=1)

        seq1_c = np.where(seq1_argmax == 1.0, 1.0, 0.0)
        seq1_g = np.where(seq1_argmax == 2.0, 1.0, 0.0)

        seq1_gc = seq1_c + seq1_g

        seq1_c = np.where(seq2_argmax == 1.0, 1.0, 0.0)
        seq1_g = np.where(seq2_argmax == 2.0, 1.0, 0.0)

        seq2_gc = seq1_c + seq1_g

        gc_count = np.count_nonzero(seq1_gc, axis=1) + np.count_nonzero(seq2_gc, axis=1)
        gc_content = gc_count / (seq1_gc.shape[1] + seq2_gc.shape[1])

        gc_count_diff = np.abs(np.count_nonzero(seq1_gc, axis=1) - np.count_nonzero(seq2_gc, axis=1))
        gc_content_diff = gc_count_diff / (seq1_gc.shape[1])
        gc_content_diff = gc_content_diff[:, np.newaxis]

    return gc_content, gc_content_diff


def extract_gene_coordinates(gene_coords_file,
                             chrA,
                             chrB,
                             chrA_bins,
                             chrB_bins,
                             resolution,
                             min_gene_vicinity=5000,
                             max_enhancer_distance=50000):
    """
    This function takes a file with gene coordinates
    and converts them to Hi-C bin resolution, each
    bin is 1 if it contains a gene and 0 if it does
    not. We do this for each chromosome pair.
    :param gene_coords_file: str, path to the gene
                                  coordinate file
                                  which contains
                                  gene name, chromosome
                                  start bp, end bp. It was
                                  downloaded from ensembl biomart.
    :param chrA: str, name of first chromosome
    :param chrB: str, name of the second chromosome
    :param chrA_bins: int, number of bins in the first chromosome
    :param chrB_bins: int, number of bins in the second chromosome
    :param resolution: int, Hi-C resolution
    :param min_gene_vicinity: int, avoid ends of chromosome
    :param max_enhancer_distance: int, maximum distance upstream
                                    an anhancer can be.
    :return: gene_bins, np.array, how many Hi-C bins in the chromosome pair
                        are in vicinity of genes.
    """
    # remove chr from chromosome name
    chrA = chrA.replace("chr", "")
    chrB = chrB.replace("chr", "")
    # read the gene coordinate file
    genes = pd.read_csv(gene_coords_file, sep="\t")
    # get gene start, end and strand
    gene_starts = genes["Gene start (bp)"]
    gene_ends = genes["Gene end (bp)"]
    gene_strands = genes["Strand"]

    # get rows corresponding to the ChrA and ChrB
    chrA_rows = genes.index[genes["Chromosome/scaffold name"] == chrA].tolist()
    chrB_rows = genes.index[genes["Chromosome/scaffold name"] == chrB].tolist()

    # For the first chromosome, all bins, upto max_enhancer_distance
    # before the start of the gene, are 1, indicating gene vicinity
    # In the negative strand, the genomic coordinates and gene body
    # are reverse, so add max_enhancer_distance to gene end (larger
    # genomic coordinate).
    gene_bins_chrA = np.zeros((chrA_bins, 1))
    for idx, i in enumerate(chrA_rows):
        if int(gene_strands[idx]) == 1:
            gene_vicinity_start = max(min_gene_vicinity, int(gene_starts[idx]) - max_enhancer_distance)
            gene_vicinity_end = int(gene_ends[idx])
        else:
            gene_vicinity_start = int(gene_starts[idx])
            gene_vicinity_end = min(int(gene_ends[idx]) + max_enhancer_distance, chrA_bins * resolution)
        bin_id_start = int(np.floor(gene_vicinity_start / resolution))
        bin_id_end = int(np.ceil(gene_vicinity_end / resolution))
        gene_bins_chrA[bin_id_start:bin_id_end, 0] = 1

    # print(f"gene_bins_chrA: {gene_bins_chrA}")
    # repeat for the second chromosome
    gene_bins_chrB = np.zeros((1, chrB_bins))
    for idx, i in enumerate(chrB_rows):
        if int(gene_strands[idx]) == 1:
            gene_vicinity_start = max(min_gene_vicinity, int(gene_starts[idx]) - max_enhancer_distance)
            gene_vicinity_end = int(gene_ends[idx])
        else:
            gene_vicinity_start = int(gene_starts[idx])
            gene_vicinity_end = min(int(gene_ends[idx]) + max_enhancer_distance, chrB_bins * resolution)
        bin_id_start = int(np.floor(gene_vicinity_start / resolution))
        bin_id_end = int(np.ceil(gene_vicinity_end / resolution))
        gene_bins_chrB[0, bin_id_start:bin_id_end] = 1

    # multiply the first and second chromosome bins, to make gene vicinity interactions 1.
    gene_bins_matmul = np.matmul(gene_bins_chrA, gene_bins_chrB)
    gene_bins_logical_or = np.logical_or(gene_bins_chrA, gene_bins_chrB)

    return gene_bins_matmul, gene_bins_logical_or


def extract_centromere_coordinates(centromeres_file,
                                   chrA,
                                   chrB,
                                   chrA_bins,
                                   chrB_bins,
                                   resolution,
                                   centromere_wobble_distance=1000000
                                   ):
    """
    This function extacts the centromere coordinates
    from each chromosome and makes all centromere
    bins to zero, so as we can avoid them in our model.
    :param centromeres_file: str, path to centromere coord file.
    :param chrA: str, name of the first chromosome.
    :param chrB: str, name of the second chromosome.
    :param chrA_bins: int, number of hi-c bins in the first
                            chromosome.
    :param chrB_bins: int, number of hi-c bins in the second
                            chromosome.
    :param resolution: int, hi-c bin resolution
    :param centromere_wobble_distance: int, region around centromere to avoid
    :return: centromere_bins: centromere bins
    """
    centromeres = pd.read_csv(centromeres_file, sep="\t")
    centromeres.columns = ['index', 'chromosome', 'start', 'end', 'name']

    centromeres_starts = centromeres["start"]
    centromeres_ends = centromeres["end"]

    chrA_rows = centromeres.index[centromeres["chromosome"] == chrA].tolist()
    chrB_rows = centromeres.index[centromeres["chromosome"] == chrB].tolist()

    centromere_bins_chrA = np.ones((chrA_bins, 1))
    for idx, i in enumerate(chrA_rows):
        centromere_vicinity_start = int(centromeres_starts[idx]) - centromere_wobble_distance
        centromere_vicinity_end = int(centromeres_ends[idx]) + centromere_wobble_distance
        bin_id_start = int(np.floor(centromere_vicinity_start / resolution))
        bin_id_end = int(np.ceil(centromere_vicinity_end / resolution))
        centromere_bins_chrA[bin_id_start:bin_id_end, 0] = 0

    centromere_bins_chrB = np.ones((1, chrB_bins))
    for idx, i in enumerate(chrB_rows):
        centromere_vicinity_start = int(centromeres_starts[idx]) - centromere_wobble_distance
        centromere_vicinity_end = int(centromeres_ends[idx]) + centromere_wobble_distance
        bin_id_start = int(np.floor(centromere_vicinity_start / resolution))
        bin_id_end = int(np.ceil(centromere_vicinity_end / resolution))
        centromere_bins_chrB[0, bin_id_start:bin_id_end] = 0

    centromere_bins = np.matmul(centromere_bins_chrA, centromere_bins_chrB)
    return centromere_bins


def count_pos_neg(labels, set_name=""):
    """
    Count the number of positive
    and negative (everything
    else) examples in our set
    :param labels: np.array, one-hot-encoded
                             label,
    :param set_name: str, training,
                          validation or
                          test set.
    :return: #positives, #negatives
    """
    # First column are positive labels
    m6as = np.where(labels == 1)[0]
    # Second column are negative labels
    nulls = np.where(labels == 0)[0]
    num_pos = len(m6as)
    num_neg = len(nulls)
    print(f"{set_name} has {num_pos}" f" positives and {num_neg} negatives")
    return num_pos, num_neg


def decode_chrome_order_dict(string_val):
    """
    Extract dict from config string
    :param string_val: str, config string
                        for chrom_order dict
    :return: dict
    """
    string_val = string_val.replace("{", "")
    string_val = string_val.replace("}", "")
    string_vals = string_val.split(", ")
    string_dict = dict()
    for val in string_vals:
        kk, vv = val.split(":")
        string_dict[kk] = int(vv)
        # print(kk, vv)
    return string_dict


def decode_chrome_order_inv_dict(string_val):
    """
    Extract dict from config string
    :param string_val: str, config string
                        for chrom_order dict
    :return: dict
    """
    string_val = string_val.replace("{", "")
    string_val = string_val.replace("}", "")
    string_vals = string_val.split(", ")
    string_dict = dict()
    for val in string_vals:
        kk, vv = val.split(":")
        string_dict[int(kk)] = vv
    return string_dict


def decode_list(string_val):
    """
    Extract list from config string
    :param string_val: str, config string
                        for train, val,
                        test chrom list
    :return: list
    """
    string_val = string_val.replace("[", "")
    string_val = string_val.replace("]", "")
    string_vals = string_val.split(", ")
    return string_vals


if __name__ == '__main__':
    input_tensor1 = torch.randn(2, 4, 100000)
    input_tensor2 = torch.randn(2, 4, 100000)
    gc_content, gc_content_diff = gc_predictor(input_tensor1, input_tensor1, is_torch=True)
    print(f"With torch, gc_content: {gc_content}, gc_content_diff: {gc_content_diff}")
    input_tensor1 = np.random.uniform(size=(2, 4, 100000))
    input_tensor2 = np.random.uniform(size=(2, 4, 100000))
    gc_content, gc_content_diff = gc_predictor(input_tensor1, input_tensor1, is_torch=False)
    print(f"No torch, gc_content: {gc_content}, gc_content_diff: {gc_content_diff}")
