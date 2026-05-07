"""Generate FASTA-style AMR sequences and convert them into a modeling dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


DNA_ALPHABET = ("A", "T", "G", "C")
ANTIBIOTIC_CONFIG = {
    "beta-lactam": {
        "prefixes": ["bla", "ctx", "oxa"],
        "class_weight": 0.28,
        "resistant_rate": 0.72,
        "class_motif": "ATGCGTAC",
        "resistant_motif": "GGGTTTAA",
        "susceptible_motif": "AACCGGTT",
    },
    "tetracycline": {
        "prefixes": ["tet", "otr"],
        "class_weight": 0.18,
        "resistant_rate": 0.58,
        "class_motif": "TATGCCGA",
        "resistant_motif": "TTGACCAA",
        "susceptible_motif": "CCGTATTA",
    },
    "aminoglycoside": {
        "prefixes": ["aac", "aph", "aad"],
        "class_weight": 0.17,
        "resistant_rate": 0.61,
        "class_motif": "CGTATGGA",
        "resistant_motif": "GATCGGTA",
        "susceptible_motif": "TACCAAGT",
    },
    "fluoroquinolone": {
        "prefixes": ["qnr", "gyr", "par"],
        "class_weight": 0.12,
        "resistant_rate": 0.42,
        "class_motif": "GGCATATA",
        "resistant_motif": "CTTAGGCA",
        "susceptible_motif": "ATCCGTGA",
    },
    "macrolide": {
        "prefixes": ["erm", "mef", "mph"],
        "class_weight": 0.14,
        "resistant_rate": 0.48,
        "class_motif": "AAGTCCGT",
        "resistant_motif": "GTATCCGA",
        "susceptible_motif": "CCTTAAGG",
    },
    "sulfonamide": {
        "prefixes": ["sul", "dfr", "fol"],
        "class_weight": 0.11,
        "resistant_rate": 0.55,
        "class_motif": "TTCCGGAA",
        "resistant_motif": "AGGCTTCA",
        "susceptible_motif": "CATTAACG",
    },
}


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(DNA_ALPHABET, k=length))


def insert_motif(sequence: str, motif: str, rng: random.Random) -> str:
    if len(sequence) <= len(motif):
        return motif[: len(sequence)]
    start = rng.randint(0, len(sequence) - len(motif))
    return f"{sequence[:start]}{motif}{sequence[start + len(motif):]}"


def mutate_sequence(sequence: str, mutation_rate: float, rng: random.Random) -> str:
    chars = list(sequence)
    n_mutations = max(1, int(len(chars) * mutation_rate))
    for index in rng.sample(range(len(chars)), k=min(n_mutations, len(chars))):
        original = chars[index]
        replacement_pool = [base for base in DNA_ALPHABET if base != original]
        chars[index] = rng.choice(replacement_pool)
    return "".join(chars)


def build_records(num_sequences: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    classes = list(ANTIBIOTIC_CONFIG.keys())
    weights = [ANTIBIOTIC_CONFIG[name]["class_weight"] for name in classes]
    records: list[dict] = []

    for index in range(1, num_sequences + 1):
        antibiotic_class = rng.choices(classes, weights=weights, k=1)[0]
        config = ANTIBIOTIC_CONFIG[antibiotic_class]
        resistance_label = int(rng.random() < config["resistant_rate"])
        amr_identifier = f"{rng.choice(config['prefixes'])}_{rng.randint(1, 24)}"
        seq_length = rng.randint(50, 200)

        gene_sequence = random_sequence(seq_length, rng)
        gene_sequence = insert_motif(gene_sequence, config["class_motif"], rng)
        label_motif = config["resistant_motif"] if resistance_label == 1 else config["susceptible_motif"]
        gene_sequence = insert_motif(gene_sequence, label_motif, rng)
        gene_sequence = mutate_sequence(gene_sequence, mutation_rate=0.03, rng=rng)

        records.append(
            {
                "seq_id": f"seq_{index}",
                "gene_sequence": gene_sequence,
                "antibiotic_class": antibiotic_class,
                "amr_identifier": amr_identifier,
                "resistance_label": resistance_label,
            }
        )

    return records


def write_fasta(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                f">{record['seq_id']}|{record['antibiotic_class']}|"
                f"{record['amr_identifier']}|{record['resistance_label']}\n"
            )
            handle.write(f"{record['gene_sequence']}\n")


def fasta_to_dataframe(fasta_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    header: str | None = None
    sequence_chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and sequence_chunks:
                    rows.append(_fasta_entry_to_row(header, "".join(sequence_chunks)))
                header = line[1:]
                sequence_chunks = []
                continue
            sequence_chunks.append(line.upper())

    if header is not None and sequence_chunks:
        rows.append(_fasta_entry_to_row(header, "".join(sequence_chunks)))

    return pd.DataFrame(rows)


def _fasta_entry_to_row(header: str, sequence: str) -> dict:
    parts = [part.strip() for part in header.split("|")]
    if len(parts) != 4:
        raise ValueError(
            "Expected FASTA header format 'seq_id|antibiotic_class|amr_identifier|resistance_label'. "
            f"Received: {header}"
        )
    _, antibiotic_class, amr_identifier, resistance_label = parts
    return {
        "gene_sequence": sequence,
        "antibiotic_class": antibiotic_class,
        "amr_identifier": amr_identifier,
        "resistance_label": int(resistance_label),
    }


def generate_dataset(
    fasta_path: Path,
    csv_path: Path,
    num_sequences: int = 1400,
    seed: int = 42,
) -> pd.DataFrame:
    records = build_records(num_sequences=num_sequences, seed=seed)
    write_fasta(records, fasta_path)
    dataset = fasta_to_dataframe(fasta_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(csv_path, index=False)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FASTA-formatted AMR gene sequences and a CSV dataset.")
    parser.add_argument("--num-sequences", type=int, default=1400, help="Number of FASTA entries to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--fasta-path", type=Path, default=Path("data/sequences.fasta"), help="Output FASTA path.")
    parser.add_argument("--csv-path", type=Path, default=Path("data/dataset.csv"), help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    dataframe = generate_dataset(
        fasta_path=arguments.fasta_path,
        csv_path=arguments.csv_path,
        num_sequences=arguments.num_sequences,
        seed=arguments.seed,
    )
    print(f"Generated {len(dataframe)} records")
    print(f"FASTA saved to {arguments.fasta_path}")
    print(f"Dataset saved to {arguments.csv_path}")
