"""Core analysis logic for Plasmidsaurus sequencing QC.

Handles FASTA/GenBank parsing, local pairwise alignment against a folder of
reference sequences, and screening of unmatched regions against NCBI for
human-derived contamination.

Depends on Streamlit only for @st.cache_data on the heavy alignment/
restriction-mapping functions -- everything else here stays plain Python so
it's still usable/testable outside of a running app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from Bio import Entrez, SeqIO
from Bio.Align import Alignment, PairwiseAligner, substitution_matrices
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Restriction import Analysis, RestrictionBatch
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

SEQUENCE_FILE_GLOBS = ("*.fasta", "*.fa", "*.fna", "*.gb", "*.gbk", "*.genbank")


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    reference_id: str
    reference_description: str
    score: float
    identity_pct: float
    aligned_length: int
    query_coverage_pct: float
    reference_coverage_pct: float
    reference_length: int
    query_strand: str  # "+" if the match is on the query as given, "-" if
                        # only found by aligning the query's reverse complement
    alignment: Alignment


@dataclass
class HumanGeneHit:
    accession: str
    description: str
    percent_identity: float
    e_value: float
    align_length: int


@dataclass
class GeneIdentification:
    accession: str
    hit_description: str
    percent_identity: float
    e_value: float
    align_length: int
    gene_symbol: str | None
    gene_name: str | None


@dataclass
class Mutation:
    """A substitution, or a contiguous indel block, in reference coordinates.

    A run of adjacent deletion/insertion columns is one physical indel
    event, not N independent point mutations, so such runs are merged into
    a single entry with `length` > 1 rather than reported one base at a
    time. Substitutions are always length 1 and reported individually.
    """
    ref_position: int  # 1-based position in the reference/gene; for an
                        # insertion, the position immediately before it
    ref_base: str       # single base for a substitution; the deleted/
                         # inserted subsequence for an indel block
    query_base: str
    kind: str  # "substitution", "insertion", or "deletion"
    length: int = 1


@dataclass
class GeneScope:
    """The reference-coordinate gene/CDS region(s) mutation reporting is
    scoped to, and where that boundary information came from."""
    intervals: list[tuple[int, int]]  # 0-based half-open reference-coordinate intervals
    source: str  # "reference", "query", or "none" (no gene annotation found anywhere)


@dataclass
class Truncation:
    missing_start_bp: int  # reference bases missing from the 5' end
    missing_end_bp: int    # reference bases missing from the 3' end


@dataclass
class ReferenceFreeMatch:
    """Result of exact-matching a gene's sequence against one query read,
    with no reference plasmid/gene file involved."""
    query_id: str
    found: bool
    start: int | None = None        # 1-based inclusive
    end: int | None = None          # 1-based inclusive
    orientation: str | None = None  # "Forward" or "Reverse"


@dataclass
class QueryOutcome:
    """One query's headline result against its best-matching reference, as
    already classified by pick_best_match/classify_match -- the input
    recommend_clones aggregates across all queries in a run."""
    query_id: str
    reference_id: str
    status: str  # "PASS", "GENE_FOUND", or "FAIL"
    identity_pct: float


@dataclass
class AlignmentSegment:
    """One contiguous block of a pairwise alignment, for visualization.

    Coordinates are 0-based half-open. `ref_*`/`query_*` are in the
    respective sequence's own coordinates (query is in whatever
    orientation the alignment actually used); `col_*` is the shared
    position within the alignment itself (gaps included), which is what
    keeps the reference and query tracks of a plot lined up vertically
    even where an indel has shifted one relative to the other.
    """
    kind: str  # "match", "mismatch", "insertion", or "deletion"
    col_start: int
    col_end: int
    ref_start: int
    ref_end: int
    query_start: int
    query_end: int
    ref_seq: str
    query_seq: str


# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------

def find_sequence_files(folder: str | Path) -> list[Path]:
    """Return sorted paths to standard sequence files (FASTA/GenBank) in a folder.

    Raises ValueError if the path doesn't exist or isn't a directory. Returns
    an empty list (not an error) if the directory has no matching files.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    paths: set[Path] = set()
    for pattern in SEQUENCE_FILE_GLOBS:
        paths.update(folder.glob(pattern))
    return sorted(paths)


def _sequence_format_for_suffix(suffix: str) -> str:
    return "genbank" if suffix.lower() in (".gb", ".gbk", ".genbank") else "fasta"


def _parse_records(handle, fmt: str, source_name: str) -> list[SeqRecord]:
    records = list(SeqIO.parse(handle, fmt))
    if not records:
        raise ValueError(f"No sequences found in {source_name} (parsed as {fmt}).")
    return records


def parse_records_from_path(path: str | Path) -> list[SeqRecord]:
    """Parse a FASTA or GenBank file on disk into SeqRecords."""
    path = Path(path)
    return _parse_records(str(path), _sequence_format_for_suffix(path.suffix), path.name)


def parse_records_from_upload(uploaded_file) -> list[SeqRecord]:
    """Parse a FASTA or GenBank file straight out of an in-memory upload
    (e.g. Streamlit's UploadedFile) rather than a path on disk.

    Streamlit Cloud has no local filesystem for a folder-path input to point
    at, so every reference/query file has to be read directly from the
    browser upload's bytes instead -- wrapped in a StringIO so SeqIO.parse
    sees the same kind of text handle it would get from an open() call.
    """
    name = getattr(uploaded_file, "name", "uploaded_file")
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    fmt = _sequence_format_for_suffix(Path(name).suffix)
    return _parse_records(StringIO(text), fmt, name)


def build_seqrecord_from_pasted_sequence(name: str, sequence: str) -> SeqRecord:
    """Turn a user-pasted oligo name + raw sequence into a SeqRecord, so a
    manually-pasted reference can be handed to the same alignment/plotting
    pipeline as anything parsed from a physical FASTA/GenBank file.

    Cleaning reuses _sanitize_sequence -- the same whitespace/newline/stray-
    digit stripping applied to every other sequence in this module, e.g. for
    position numbers pasted in from a sequence viewer -- so a pasted
    reference isn't held to a stricter standard than an uploaded one.
    Raises ValueError on a blank name or a sequence with no letters left
    after cleaning.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Oligo name is required.")

    clean_seq = _sanitize_sequence(sequence)
    if not clean_seq:
        raise ValueError("Pasted sequence is empty (or has no letters left after cleaning).")

    return SeqRecord(Seq(clean_seq), id=clean_name, name=clean_name, description=clean_name)


GENE_FEATURE_TYPES = ("gene", "CDS")


def extract_gene_records(record: SeqRecord) -> list[SeqRecord]:
    """Pull out named gene/CDS features from an annotated GenBank record as
    standalone SeqRecords.

    This lets each named gene (e.g. AGGF1) be compared to a query on its
    own, independent of the rest of the plasmid backbone, instead of
    requiring a hand-prepared gene-only reference file. Records with no
    features (e.g. plain FASTA) simply yield nothing.
    """
    genes: list[SeqRecord] = []
    for feature in record.features:
        if feature.type not in GENE_FEATURE_TYPES:
            continue
        name = (
            feature.qualifiers.get("gene", [None])[0]
            or feature.qualifiers.get("label", [None])[0]
            or feature.qualifiers.get("locus_tag", [None])[0]
        )
        if not name:
            continue
        try:
            seq = feature.extract(record.seq)
        except Exception:
            continue
        gene_record = SeqRecord(seq, id=name, description=f"{name} ({feature.type}) from {record.id}")
        # Self-spanning feature so gene-scoping logic can treat this
        # extracted record as "entirely gene" without special-casing it --
        # otherwise it looks identical to an unannotated plain-FASTA
        # whole-plasmid reference (both have no .features).
        gene_record.features.append(
            SeqFeature(FeatureLocation(0, len(seq)), type="gene", qualifiers={"gene": [name]})
        )
        genes.append(gene_record)
    return genes


def expand_with_gene_features(references: list[SeqRecord]) -> list[SeqRecord]:
    """Return `references` plus any named gene/CDS features extracted from
    the annotated ones among them, deduplicated by (name, sequence)."""
    expanded = list(references)
    seen = set()
    for record in references:
        for gene in extract_gene_records(record):
            key = (gene.id, str(gene.seq))
            if key in seen:
                continue
            seen.add(key)
            expanded.append(gene)
    return expanded


# --------------------------------------------------------------------------
# Pairwise alignment
# --------------------------------------------------------------------------

def build_aligner(mode: str = "local") -> PairwiseAligner:
    """Build a local (Smith-Waterman-style) nucleotide aligner.

    Gap costs are set well above the EMBOSS-water defaults (open=10,
    extend=0.5): with extend that cheap, the aligner can profitably stitch
    together many short chance-matched fragments across long stretches of
    otherwise-unrelated sequence (e.g. a shared gene sitting in two
    differing plasmid backbones), inflating both score and coverage instead
    of cleanly isolating the true homologous region.
    """
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
    aligner.open_gap_score = -16
    aligner.extend_gap_score = -4
    return aligner


_NON_LETTER_RE = re.compile(r"[^A-Za-z]")


def _sanitize_sequence(seq: str) -> str:
    """Strip whitespace/newlines/non-alphabetical characters (e.g. stray
    position numbers pasted in from a sequence viewer) and uppercase, so
    the aligner's substitution-matrix lookup doesn't choke on them."""
    return _NON_LETTER_RE.sub("", seq).upper()


def _span_coverage_pct(aligned_blocks, seq_len: int) -> float:
    """Percent of a sequence spanned by its first-to-last aligned block.

    Using the alignment's matched span (rather than the gap-inflated column
    count of the alignment string) keeps this capped at 100%, and is what
    lets query- and reference-coverage be computed independently.
    """
    if len(aligned_blocks) == 0 or not seq_len:
        return 0.0
    start, end = int(aligned_blocks[0][0]), int(aligned_blocks[-1][1])
    return 100 * (end - start) / seq_len


def align_pair(query: SeqRecord, reference: SeqRecord, aligner: PairwiseAligner) -> AlignmentResult:
    """Align `query` against `reference`, trying both the query as given and
    its reverse complement, and keeping whichever scores higher.

    An insert can be cloned in either orientation relative to however the
    plasmid's linear sequence happens to be numbered, so checking only the
    forward strand (as a naive local alignment would) can completely miss a
    real, high-identity match that happens to sit on the minus strand.
    """
    query_seq = _sanitize_sequence(str(query.seq))
    ref_seq = _sanitize_sequence(str(reference.seq))
    revcomp_query_seq = str(Seq(query_seq).reverse_complement())

    forward_alignment = aligner.align(query_seq, ref_seq)[0]
    reverse_alignment = aligner.align(revcomp_query_seq, ref_seq)[0]

    if reverse_alignment.score > forward_alignment.score:
        alignment, used_query_seq, strand = reverse_alignment, revcomp_query_seq, "-"
    else:
        alignment, used_query_seq, strand = forward_alignment, query_seq, "+"

    aligned_query, aligned_ref = alignment[0], alignment[1]

    matches = sum(1 for a, b in zip(aligned_query, aligned_ref) if a == b and a != "-")
    aligned_length = len(aligned_query)
    identity_pct = 100 * matches / aligned_length if aligned_length else 0.0
    query_coverage_pct = _span_coverage_pct(alignment.aligned[0], len(used_query_seq))
    reference_coverage_pct = _span_coverage_pct(alignment.aligned[1], len(ref_seq))

    return AlignmentResult(
        reference_id=reference.id,
        reference_description=reference.description,
        score=alignment.score,
        identity_pct=identity_pct,
        aligned_length=aligned_length,
        query_coverage_pct=query_coverage_pct,
        reference_coverage_pct=reference_coverage_pct,
        reference_length=len(ref_seq),
        query_strand=strand,
        alignment=alignment,
    )


@st.cache_data(show_spinner=False, hash_funcs={SeqRecord: lambda r: (r.id, str(r.seq))})
def compare_to_references(
    query: SeqRecord,
    references: list[SeqRecord],
    _aligner: PairwiseAligner,
) -> list[AlignmentResult]:
    """Align `query` locally against every reference, best score first.

    This is the expensive step (a full DP alignment per reference), so it's
    cached on the actual sequence content of `query`/`references` -- a
    SeqRecord isn't hashable by Streamlit's default hasher, hence the
    `hash_funcs` override, and `_aligner`'s leading underscore excludes it
    from hashing entirely (it's a fixed local/PairwiseAligner config with a
    single construction site, so it can't vary the result for a given cache
    key anyway, and Streamlit can't hash it).
    """
    results = [align_pair(query, ref, _aligner) for ref in references]
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def classify_match(
    result: AlignmentResult,
    identity_threshold: float,
    coverage_threshold: float,
) -> str:
    """Classify a best-match result as "PASS", "GENE_FOUND", or "FAIL".

    PASS: the reference matches across (almost) the whole query.
    GENE_FOUND: the reference matches at high identity along its own full
        length (e.g. a gene/insert), but covers only part of the query,
        meaning the rest of the query (e.g. vector backbone) differs.
    FAIL: neither the query nor the reference is well covered — no
        confident match. Includes the case of a completely unrelated
        sequence, where the aligner still forces *some* local alignment
        (Smith-Waterman always returns its best-scoring region even when
        every candidate is bad) -- below-threshold identity/coverage here
        means that alignment isn't a real biological match, so callers must
        not run mutation-counting against it. See build_no_match_message.
    """
    if result.identity_pct < identity_threshold:
        return "FAIL"
    if result.query_coverage_pct >= coverage_threshold:
        return "PASS"
    if result.reference_coverage_pct >= coverage_threshold:
        return "GENE_FOUND"
    return "FAIL"


def build_no_match_message(
    result: AlignmentResult,
    identity_threshold: float,
    coverage_threshold: float,
) -> str:
    """Ready-to-display explanation for a FAIL-classified result.

    Names whichever threshold it actually fell short of -- identity is
    checked first since classify_match short-circuits on it, so a FAIL
    caused by low identity should never be reported as a coverage problem.
    Callers must show this instead of running mutation-counting on what's
    typically just a forced alignment against an unrelated sequence.
    """
    if result.identity_pct < identity_threshold:
        return (
            f"No confident match found. The sequence identity "
            f"({result.identity_pct:.1f}%) falls below the required threshold "
            f"({identity_threshold:.0f}%)."
        )
    return (
        "No confident match found. Neither the query coverage "
        f"({result.query_coverage_pct:.1f}%) nor the reference coverage "
        f"({result.reference_coverage_pct:.1f}%) reaches the required coverage "
        f"threshold ({coverage_threshold:.0f}%)."
    )


_STATUS_RANK = {"PASS": 0, "GENE_FOUND": 1, "FAIL": 2}


def pick_best_match(
    results: list[AlignmentResult],
    identity_threshold: float,
    coverage_threshold: float,
) -> AlignmentResult | None:
    """Pick the most informative result to headline.

    Prefers a confidently classified PASS or GENE_FOUND result over a
    merely higher raw alignment score: a whole-plasmid reference can
    occasionally out-score a cleanly isolated gene-level match by picking
    up a few points of incidental similarity right at the match boundary,
    even though the gene-level match is the more meaningful answer.
    """
    if not results:
        return None
    return min(
        results,
        key=lambda r: (_STATUS_RANK[classify_match(r, identity_threshold, coverage_threshold)], -r.score),
    )


def recommend_clones(outcomes: list[QueryOutcome], reference_ids: list[str]) -> pd.DataFrame:
    """Pick one "first-pass" recommended clone per reference: the first
    query (in the given order -- queries are processed in sorted-filename
    order, so this is simultaneously alphabetical and upload order) whose
    best match against that reference reached PASS status.

    `reference_ids` is the full set of references compared against in the
    run, so a reference nobody happened to pass -- or that was never
    anyone's best match at all -- still gets a row saying so, rather than
    silently disappearing from the summary.
    """
    winners: dict[str, QueryOutcome] = {}
    for outcome in outcomes:
        if outcome.status != "PASS":
            continue
        winners.setdefault(outcome.reference_id, outcome)  # first PASS wins

    rows = []
    for reference_id in reference_ids:
        winner = winners.get(reference_id)
        rows.append({
            "Reference": reference_id,
            "Recommended Clone": winner.query_id if winner else "No passing clones found",
            "Match Identity (%)": round(winner.identity_pct, 1) if winner else None,
        })
    return pd.DataFrame(rows, columns=["Reference", "Recommended Clone", "Match Identity (%)"])


def find_mutations(result: AlignmentResult, max_report: int = 50) -> list[Mutation]:
    """List substitutions and indel blocks within the aligned region, in
    1-based reference coordinates. A contiguous run of deletion (or
    insertion) columns is one physical indel event and is merged into a
    single entry rather than reported one base at a time; adjacent
    substitutions are still independent events and stay one-per-entry.
    Stops after `max_report` entries.
    """
    alignment = result.alignment
    aligned_query, aligned_ref = alignment[0], alignment[1]
    ref_blocks = alignment.aligned[1]
    ref_pos = int(ref_blocks[0][0]) if len(ref_blocks) else 0

    raw: list[tuple[int, str, str, str]] = []  # (ref_pos, kind, ref_base, query_base)
    for q_base, r_base in zip(aligned_query, aligned_ref):
        if r_base != "-":
            ref_pos += 1
        if q_base == r_base:
            continue
        kind = "deletion" if q_base == "-" else "insertion" if r_base == "-" else "substitution"
        raw.append((ref_pos, kind, r_base, q_base))

    mutations: list[Mutation] = []
    i = 0
    while i < len(raw) and len(mutations) < max_report:
        pos, kind, r_base, q_base = raw[i]
        if kind == "substitution":
            mutations.append(Mutation(ref_position=pos, ref_base=r_base, query_base=q_base, kind=kind, length=1))
            i += 1
            continue

        # Merge a contiguous run of the same indel kind into one entry.
        # Deletions consume the reference, so a run has strictly
        # consecutive ref_pos values; insertions don't consume the
        # reference, so every column in a run shares the same ref_pos.
        run_ref, run_query = [r_base], [q_base]
        j = i + 1
        while j < len(raw) and raw[j][1] == kind and (
            (kind == "deletion" and raw[j][0] == raw[j - 1][0] + 1)
            or (kind == "insertion" and raw[j][0] == pos)
        ):
            run_ref.append(raw[j][2])
            run_query.append(raw[j][3])
            j += 1

        ref_str, query_str = "".join(run_ref), "".join(run_query)
        length = len(run_ref) if kind == "deletion" else len(run_query)
        mutations.append(Mutation(ref_position=pos, ref_base=ref_str, query_base=query_str, kind=kind, length=length))
        i = j

    return mutations


# --------------------------------------------------------------------------
# Raw alignment text
# --------------------------------------------------------------------------

_MISMATCH_HTML_COLOR = "#d03b3b"  # matches the mismatch color in the alignment map
_HIDDEN_RUN_HTML_COLOR = "#898781"  # matches the muted/unaligned color elsewhere


def format_alignment_html(
    result: AlignmentResult,
    only_mismatches: bool = False,
    flank_bp: int = 20,
    wrap: int = 60,
) -> str:
    """Render a pairwise alignment as an HTML `<pre>` block: a Reference /
    match-bar / Query triplet per `wrap`-column chunk, with mismatched bases
    highlighted in red.

    Rebuilt rather than using Bio.Align.Alignment.__str__ so it can collapse
    long perfect-match runs -- printing every one of a multi-kb alignment's
    mostly-identical columns makes the page unusably long. When
    `only_mismatches` is set, any column more than `flank_bp` away from a
    mismatch/insertion/deletion is replaced by a single collapsed-run line
    instead of being printed.
    """
    aligned_query, aligned_ref = result.alignment[0], result.alignment[1]
    num_cols = len(aligned_query)

    if only_mismatches:
        visible = bytearray(num_cols)
        for c, (q_base, r_base) in enumerate(zip(aligned_query, aligned_ref)):
            if q_base != r_base:
                lo, hi = max(0, c - flank_bp), min(num_cols, c + flank_bp + 1)
                visible[lo:hi] = b"\x01" * (hi - lo)
    else:
        visible = bytearray(b"\x01" * num_cols)

    blocks: list[str] = []
    ref_pos = query_pos = 0
    col = 0
    while col < num_cols:
        run_visible = visible[col]
        start = col
        while col < num_cols and visible[col] == run_visible:
            col += 1
        q_run, r_run = aligned_query[start:col], aligned_ref[start:col]

        if not run_visible:
            # A hidden run is always pure match (any mismatch/indel column
            # would have been marked visible), so its length is exactly the
            # number of identical bp it represents on both sequences.
            hidden_bp = col - start
            blocks.append(
                f'<div style="color:{_HIDDEN_RUN_HTML_COLOR};font-style:italic;'
                f'text-align:center">... [{hidden_bp} identical bp hidden] ...</div>'
            )
            ref_pos += hidden_bp
            query_pos += hidden_bp
            continue

        for i in range(0, len(q_run), wrap):
            q_chunk, r_chunk = q_run[i:i + wrap], r_run[i:i + wrap]
            r_chunk_len = sum(1 for b in r_chunk if b != "-")
            q_chunk_len = sum(1 for b in q_chunk if b != "-")
            match_bar = "".join("|" if a == b else " " for a, b in zip(q_chunk, r_chunk))
            r_html, q_html = [], []
            for q_base, r_base in zip(q_chunk, r_chunk):
                if q_base != r_base and q_base != "-" and r_base != "-":
                    r_html.append(f'<span style="color:{_MISMATCH_HTML_COLOR};font-weight:700">{r_base}</span>')
                    q_html.append(f'<span style="color:{_MISMATCH_HTML_COLOR};font-weight:700">{q_base}</span>')
                else:
                    r_html.append(r_base)
                    q_html.append(q_base)
            # Label + position + separator width must match exactly across all
            # three lines, or the match-bar's "|" columns drift out from under
            # the bases they're marking.
            blocks.append(
                f"{'Ref':<6}{ref_pos + 1:>7} {''.join(r_html)} {ref_pos + r_chunk_len}\n"
                f"{'':<6}{'':>7} {match_bar}\n"
                f"{'Query':<6}{query_pos + 1:>7} {''.join(q_html)} {query_pos + q_chunk_len}"
            )
            ref_pos += r_chunk_len
            query_pos += q_chunk_len

    # A real <pre> tag gets intercepted and re-styled by Streamlit's markdown
    # renderer (which drops this inline style and collapses whitespace), so
    # whitespace preservation is done via a plain <div> instead.
    style = "white-space:pre;overflow-x:auto;line-height:1.4;font-family:monospace"
    return f'<div style="{style}">' + "\n\n".join(blocks) + "</div>"


# --------------------------------------------------------------------------
# Visual alignment map (interactive Plotly figure)
# --------------------------------------------------------------------------

def build_alignment_segments(result: AlignmentResult) -> list[AlignmentSegment]:
    """Run-length encode a pairwise alignment into match/mismatch/insertion/
    deletion blocks, for visualization.

    Match and indel runs are merged into single contiguous segments (like
    find_mutations merges indel runs); mismatches are kept one column per
    segment so each SNP stays individually addressable on hover, matching
    how find_mutations reports substitutions individually.
    """
    alignment = result.alignment
    aligned_query, aligned_ref = alignment[0], alignment[1]

    segments: list[AlignmentSegment] = []
    ref_pos = 0
    query_pos = 0

    run_kind: str | None = None
    run_col_start = run_ref_start = run_query_start = 0
    run_ref_bases: list[str] = []
    run_query_bases: list[str] = []

    def close_run(end_col: int) -> None:
        segments.append(AlignmentSegment(
            kind=run_kind, col_start=run_col_start, col_end=end_col,
            ref_start=run_ref_start, ref_end=ref_pos,
            query_start=run_query_start, query_end=query_pos,
            ref_seq="".join(run_ref_bases), query_seq="".join(run_query_bases),
        ))

    for col, (q_base, r_base) in enumerate(zip(aligned_query, aligned_ref)):
        if q_base == r_base:
            kind = "match"
        elif r_base == "-":
            kind = "insertion"
        elif q_base == "-":
            kind = "deletion"
        else:
            kind = "mismatch"

        if kind == "mismatch":
            # Never merged with neighbors, even other mismatches -- each SNP
            # is its own event, mirroring find_mutations.
            if run_kind is not None:
                close_run(col)
                run_kind = None
            ref_before, query_before = ref_pos, query_pos
            if r_base != "-":
                ref_pos += 1
            if q_base != "-":
                query_pos += 1
            segments.append(AlignmentSegment(
                kind="mismatch", col_start=col, col_end=col + 1,
                ref_start=ref_before, ref_end=ref_pos,
                query_start=query_before, query_end=query_pos,
                ref_seq=r_base, query_seq=q_base,
            ))
            continue

        if kind != run_kind:
            if run_kind is not None:
                close_run(col)
            run_kind = kind
            run_col_start, run_ref_start, run_query_start = col, ref_pos, query_pos
            run_ref_bases, run_query_bases = [], []

        if r_base != "-":
            ref_pos += 1
        if q_base != "-":
            query_pos += 1
        run_ref_bases.append(r_base)
        run_query_bases.append(q_base)

    if run_kind is not None:
        close_run(len(aligned_query))

    return segments


_SEGMENT_COLORS = {
    "match": "#0ca30c",       # status: good
    "mismatch": "#d03b3b",    # status: critical -- SNP
    "insertion": "#fab219",   # status: warning -- extra query bases
    "deletion": "#ec835a",    # status: serious -- query bases missing
    "unaligned": "#c3c2b7",   # muted -- outside the alignment entirely
}
_SEGMENT_LABELS = {
    "match": "Match",
    "mismatch": "SNP / mismatch",
    "insertion": "Insertion",
    "deletion": "Deletion",
    "unaligned": "Not aligned",
}

COMMON_MOTIFS = {
    "EGFP": {"seq": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATC", "color": "#2196F3"},
    "mCherry": {"seq": "ATGGTGAGCAAGGGCGAGGAGGATAACATGGCCATCATCAAGGAG", "color": "#F44336"},
    "FLAG-tag": {"seq": "GATTACAAGGATGACGACGATAAG", "color": "#9C27B0"},
    "HA-tag": {"seq": "TACCCATACGATGTTCCAGATTACGCT", "color": "#E91E63"},
    "His6-tag": {"seq": "CATCACCATCACCATCAC", "color": "#3F51B5"},
    "CMV Promoter": {"seq": "CGTTACATAACTTACGGTAAATGGCC", "color": "#FF9800"},
    "U6 Promoter": {"seq": "GAGGGCCTATTTCCCATGATTC", "color": "#FFC107"},
    "T7 Promoter": {"seq": "TAATACGACTCACTATAGGG", "color": "#795548"},
}


@dataclass
class MotifHit:
    """One exact motif match against a (oriented, gap-free) query sequence.

    `query_start`/`query_end` are 0-based half-open positions in the same
    oriented-query coordinate space as AlignmentSegment.query_*, so they can
    be projected onto the alignment map without any extra reorientation.
    """
    name: str
    color: str
    query_start: int
    query_end: int
    strand: str  # "+" (motif as given) or "-" (motif's reverse complement)
    matched_seq: str


def find_motifs(
    sequence: str, motifs: dict[str, dict[str, str]] = COMMON_MOTIFS
) -> list[MotifHit]:
    """Scan `sequence` for exact matches to each motif, on both strands.

    `sequence` must already be in the orientation whose coordinates the
    caller wants back (e.g. an oriented, sanitized query). Overlapping hits
    are kept (motifs can legitimately nest or abut); hits from different
    motif names that land on the exact same span are merged into a single
    combined hit instead of being drawn as indistinguishable stacked
    rectangles -- e.g. EGFP and mCherry above share a common N-terminal
    sequence, so a match there is genuinely ambiguous between the two.
    """
    seq = _sanitize_sequence(sequence).upper()
    raw_hits: list[MotifHit] = []
    for name, info in motifs.items():
        motif_seq = _sanitize_sequence(info["seq"]).upper()
        if not motif_seq:
            continue
        color = info["color"]
        rc_motif = str(Seq(motif_seq).reverse_complement())
        for strand, pattern in (("+", motif_seq), ("-", rc_motif)):
            start = 0
            while True:
                idx = seq.find(pattern, start)
                if idx == -1:
                    break
                raw_hits.append(MotifHit(
                    name=name, color=color,
                    query_start=idx, query_end=idx + len(pattern),
                    strand=strand, matched_seq=pattern,
                ))
                start = idx + 1  # allow overlapping hits of other motifs/strands

    grouped: dict[tuple[int, int, str], list[MotifHit]] = {}
    for hit in raw_hits:
        grouped.setdefault((hit.query_start, hit.query_end, hit.strand), []).append(hit)

    merged: list[MotifHit] = []
    for (start, end, strand), group in grouped.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        names = " / ".join(h.name for h in group)
        merged.append(MotifHit(
            name=f"{names} (ambiguous -- identical sequence)",
            color=group[0].color,
            query_start=start, query_end=end, strand=strand,
            matched_seq=group[0].matched_seq,
        ))

    merged.sort(key=lambda h: h.query_start)
    return merged


def _make_axis_col_mapper(
    segments: list[AlignmentSegment],
    axis: str,  # "query" or "ref"
    margin_left: int,
    missing_start: int,
    missing_end: int,
    align_start: int,
    align_end: int,
    num_cols: int,
    axis_len: int,
):
    """Build a function mapping a position on one axis (query or reference,
    in the same coordinate space as the matching AlignmentSegment.*_start/
    *_end fields) to its alignment-map column.

    A deletion consumes zero query length and an insertion consumes zero
    reference length, so each axis only has positions to map from within
    the segment kinds that actually advance along it -- within those, the
    axis and column both advance together one-for-one, so the mapping is a
    simple per-segment offset.
    """
    consuming_kinds = ("match", "mismatch", "insertion") if axis == "query" else ("match", "mismatch", "deletion")
    start_attr, end_attr = f"{axis}_start", f"{axis}_end"
    axis_segments = [s for s in segments if s.kind in consuming_kinds]

    def mapper(pos: int) -> int:
        if pos <= 0:
            return margin_left - missing_start
        if pos >= axis_len:
            return margin_left + num_cols + missing_end
        if pos < align_start:
            return (margin_left - missing_start) + pos
        if pos >= align_end:
            return margin_left + num_cols + (pos - align_end)
        for seg in axis_segments:
            seg_start, seg_end = getattr(seg, start_attr), getattr(seg, end_attr)
            if seg_start <= pos < seg_end:
                return margin_left + seg.col_start + (pos - seg_start)
        # Falls between two segments (e.g. right at an indel on this axis)
        # -- no base on this axis actually lives here, so snap to the
        # nearest segment boundary.
        for seg in axis_segments:
            if pos <= getattr(seg, start_attr):
                return margin_left + seg.col_start
        return margin_left + num_cols

    return mapper


def build_alignment_figure(
    query: SeqRecord,
    result: AlignmentResult,
    max_marked_events: int = 500,
    gene_scope: GeneScope | None = None,
    mutations: list[Mutation] | None = None,
) -> go.Figure:
    """Build an interactive two-track Plotly figure for one query-vs-reference
    alignment: reference on top, query on the bottom, colored blocks for
    matches, and clearly flagged mismatches/insertions/deletions.

    The shared x-axis is alignment-column position rather than raw sequence
    position, so reference and query bases that align to each other stay
    vertically lined up even where an indel has shifted one sequence
    relative to the other. Hovering a block still surfaces true
    reference/query coordinates and, for a mismatch or indel, the specific
    bases involved.

    `gene_scope` and `mutations` (both optional, and best passed already
    scoped to that gene) drive one-click "zoom to" buttons on the figure
    itself -- precise region jumps are what users actually want when
    inspecting a specific gene, which a generic drag-handle rangeslider is
    fiddly for on a long, mostly-irrelevant plasmid backbone.
    """
    segments = build_alignment_segments(result)
    oriented_query = _oriented_query(query, result)
    query_len = len(_sanitize_sequence(str(oriented_query.seq)))

    ref_blocks = result.alignment.aligned[1]
    q_blocks = result.alignment.aligned[0]
    ref_align_start = int(ref_blocks[0][0]) if len(ref_blocks) else 0
    ref_align_end = int(ref_blocks[-1][1]) if len(ref_blocks) else 0
    q_align_start = int(q_blocks[0][0]) if len(q_blocks) else 0
    q_align_end = int(q_blocks[-1][1]) if len(q_blocks) else 0

    missing_ref_start = ref_align_start
    missing_ref_end = result.reference_length - ref_align_end
    missing_query_start = q_align_start
    missing_query_end = query_len - q_align_end

    # Flanking regions outside the alignment are right-justified against the
    # alignment's start (and left-justified against its end), so each
    # track's own flank touches the shared column block even though the two
    # tracks' flank lengths generally differ (they aren't the same sequence).
    margin_left = max(missing_ref_start, missing_query_start)
    num_cols = segments[-1].col_end if segments else 0

    query_col = _make_axis_col_mapper(
        segments, "query", margin_left, missing_query_start, missing_query_end,
        q_align_start, q_align_end, num_cols, query_len,
    )
    ref_col = _make_axis_col_mapper(
        segments, "ref", margin_left, missing_ref_start, missing_ref_end,
        ref_align_start, ref_align_end, num_cols, result.reference_length,
    )
    motif_hits = find_motifs(str(oriented_query.seq))

    # Cap the number of individually-drawn mismatch/indel events so a very
    # heavily-mutated, large sequence doesn't produce tens of thousands of
    # bar marks; long match runs are unaffected since they're already
    # merged into single segments regardless of length.
    event_segments = [s for s in segments if s.kind != "match"]
    truncated_event_count = max(0, len(event_segments) - max_marked_events)
    if truncated_event_count:
        keep = {id(s) for s in event_segments[:max_marked_events]}
        segments = [s for s in segments if s.kind == "match" or id(s) in keep]

    by_kind: dict[str, dict[str, list]] = {
        kind: {"x": [], "base": [], "y": [], "customdata": []} for kind in _SEGMENT_COLORS
    }

    def add_bar(kind: str, x0: int, x1: int, row: str, hover: str) -> None:
        d = by_kind[kind]
        d["x"].append(x1 - x0)
        d["base"].append(x0)
        d["y"].append(row)
        d["customdata"].append(hover)

    if missing_ref_start > 0:
        add_bar(
            "unaligned", margin_left - missing_ref_start, margin_left, "Reference",
            f"Ref 1-{missing_ref_start} bp, not covered",
        )
    if missing_query_start > 0:
        add_bar(
            "unaligned", margin_left - missing_query_start, margin_left, "Query",
            f"Query 1-{missing_query_start} bp, unaligned",
        )

    for seg in segments:
        x0, x1 = margin_left + seg.col_start, margin_left + seg.col_end
        if seg.kind == "mismatch":
            hover = f"SNP @{seg.ref_start + 1}: {seg.ref_seq}→{seg.query_seq}"
            add_bar("mismatch", x0, x1, "Reference", hover)
            add_bar("mismatch", x0, x1, "Query", hover)
        elif seg.kind == "insertion":
            hover = f"+{len(seg.query_seq)} bp insertion @{seg.ref_start}"
            add_bar("insertion", x0, x1, "Query", hover)
        elif seg.kind == "deletion":
            hover = f"-{seg.ref_end - seg.ref_start} bp deletion @{seg.ref_start + 1}-{seg.ref_end}"
            add_bar("deletion", x0, x1, "Reference", hover)
        else:  # match
            length = seg.ref_end - seg.ref_start
            hover = f"{length} bp match @{seg.ref_start + 1}-{seg.ref_end}"
            add_bar("match", x0, x1, "Reference", hover)
            add_bar("match", x0, x1, "Query", hover)

    if missing_ref_end > 0:
        add_bar(
            "unaligned", margin_left + num_cols, margin_left + num_cols + missing_ref_end, "Reference",
            f"Ref {result.reference_length - missing_ref_end + 1}-{result.reference_length} bp, not covered",
        )
    if missing_query_end > 0:
        add_bar(
            "unaligned", margin_left + num_cols, margin_left + num_cols + missing_query_end, "Query",
            f"Query {query_len - missing_query_end + 1}-{query_len} bp, unaligned",
        )

    fig = go.Figure()
    for kind, d in by_kind.items():
        if not d["x"]:
            continue
        fig.add_trace(go.Bar(
            name=_SEGMENT_LABELS[kind],
            x=d["x"],
            base=d["base"],
            y=d["y"],
            orientation="h",
            width=0.38,
            marker_color=_SEGMENT_COLORS[kind],
            marker_line_color="#fcfcfb",
            marker_line_width=0.5,
            customdata=d["customdata"],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    motif_col_ranges: list[tuple[str, int, int]] = []
    if motif_hits:
        motif_x, motif_base, motif_color, motif_hover = [], [], [], []
        for hit in motif_hits:
            col0, col1 = query_col(hit.query_start), query_col(hit.query_end)
            if col1 <= col0:
                continue
            motif_x.append(col1 - col0)
            motif_base.append(col0)
            motif_color.append(hit.color)
            motif_hover.append(f"{hit.name} @{hit.query_start + 1}-{hit.query_end} ({hit.strand})")
            motif_col_ranges.append((hit.name, col0, col1))
        if motif_x:
            # Drawn after (so visually on top of) the match/mismatch/indel
            # traces, semi-transparent so the underlying call is still
            # visible through the motif highlight.
            fig.add_trace(go.Bar(
                name="Motif match",
                x=motif_x,
                base=motif_base,
                y=["Query"] * len(motif_x),
                orientation="h",
                width=0.38,
                marker_color=motif_color,
                opacity=0.55,
                marker_line_color="#0b0b0b",
                marker_line_width=0.5,
                customdata=motif_hover,
                hovertemplate="%{customdata}<extra></extra>",
            ))

    # One-click "zoom to" buttons: a drag-handle rangeslider is fiddly to
    # aim precisely at a small gene inside a long backbone, but the app
    # already knows exactly where the gene, its mutations, and any motif
    # hits are -- so jumping straight there is one relayout call away.
    full_x0 = 0
    full_x1 = margin_left + num_cols + max(missing_ref_end, missing_query_end)

    def _padded(x0: int, x1: int) -> tuple[float, float]:
        pad = max(2.0, (x1 - x0) * 0.08)
        return (max(full_x0, x0 - pad), min(full_x1, x1 + pad))

    zoom_regions: list[tuple[str, float, float]] = [("Full view", full_x0, full_x1)]

    if gene_scope and gene_scope.intervals:
        g0 = min(iv[0] for iv in gene_scope.intervals)
        g1 = max(iv[1] for iv in gene_scope.intervals)
        zoom_regions.append(("Gene of interest", *_padded(ref_col(g0), ref_col(g1))))

    if mutations:
        spans = [_mutation_ref_span(m) for m in mutations]
        m0 = min(s[0] for s in spans)
        m1 = max(s[1] for s in spans)
        zoom_regions.append(("Mutations", *_padded(ref_col(m0), ref_col(m1))))

    motif_button_counts: dict[str, int] = {}
    for name, col0, col1 in motif_col_ranges:
        label = name.split(" (")[0]  # drop the "(ambiguous -- ...)" suffix for the button
        motif_button_counts[label] = motif_button_counts.get(label, 0) + 1
        button_label = label if motif_button_counts[label] == 1 else f"{label} #{motif_button_counts[label]}"
        zoom_regions.append((button_label, *_padded(col0, col1)))

    fig.update_layout(
        barmode="overlay",
        height=190,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b", family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            showactive=False,
            x=1, xanchor="right",
            y=1.32, yanchor="top",
            pad=dict(l=4, r=4, t=2, b=2),
            font=dict(size=11),
            buttons=[
                dict(label=label, method="relayout", args=[{"xaxis.range": [x0, x1]}])
                for label, x0, x1 in zoom_regions
            ],
        )],
    )
    fig.update_xaxes(
        title_text="Alignment position (bp)",
        showgrid=False,
        zeroline=False,
        range=[full_x0, full_x1],
        minallowed=full_x0,
        maxallowed=full_x1,
        color="#898781",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Query", "Reference"],
        showgrid=False,
        zeroline=False,
        color="#0b0b0b",
        fixedrange=True,
    )
    if truncated_event_count:
        fig.add_annotation(
            text=f"({truncated_event_count} additional mismatch/indel event(s) not drawn on the map above)",
            xref="paper", yref="paper", x=0, y=-0.32, showarrow=False, align="left",
            font=dict(size=11, color="#898781"),
        )
    return fig


def _feature_label(feature) -> str:
    return (
        feature.qualifiers.get("gene", [None])[0]
        or feature.qualifiers.get("label", [None])[0]
        or feature.qualifiers.get("locus_tag", [None])[0]
        or ""
    )


def _gene_intervals(record: SeqRecord, gene_name: str | None = None) -> list[tuple[int, int]]:
    """0-based half-open (start, end) intervals for a record's named gene/CDS
    features. If `gene_name` is given, only features whose label contains it
    (case-insensitive) are included -- otherwise a richly-annotated file
    (e.g. one auto-annotated by pLannotate) will surface unrelated vector
    elements like AmpR or mEGFP as if they were the gene of interest.
    """
    matches = []
    for f in record.features:
        if f.type not in GENE_FEATURE_TYPES:
            continue
        if gene_name and gene_name.lower() not in _feature_label(f).lower():
            continue
        matches.append((int(f.location.start), int(f.location.end)))
    return matches


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _mutation_ref_span(m: Mutation) -> tuple[int, int]:
    """0-based half-open reference-coordinate span a mutation occupies.

    A deletion consumes `length` reference bases starting at its position.
    A substitution or insertion anchors to the single reference base at (or
    immediately before, for an insertion) its position.
    """
    pos0 = m.ref_position - 1
    if m.kind == "deletion":
        return pos0, pos0 + m.length
    return pos0, pos0 + 1


def _map_query_pos_to_ref_pos(
    query_pos0_original: int,
    alignment: Alignment,
    query_len: int,
    strand: str,
) -> int | None:
    """Map a 0-based position in the *original* query orientation to a
    0-based reference position, via the alignment's matched blocks.

    Used to project a query-side gene annotation into reference
    coordinates (the inverse of ref->query mapping) so that gene-overlap
    checks can be done in a single, consistent coordinate system -- which
    matters because a deletion large enough to remove an entire gene has no
    single query point left inside that gene to map *from*.
    """
    oriented_pos = query_len - 1 - query_pos0_original if strand == "-" else query_pos0_original
    ref_blocks, query_blocks = alignment.aligned[1], alignment.aligned[0]

    for qb, rb in zip(query_blocks, ref_blocks):
        q_start, q_end = int(qb[0]), int(qb[1])
        if q_start <= oriented_pos < q_end:
            return int(rb[0]) + (oriented_pos - q_start)

    for qb, rb in zip(query_blocks, ref_blocks):
        if oriented_pos <= int(qb[1]):
            return int(rb[1])
    if len(ref_blocks):
        return int(ref_blocks[-1][1])
    return None


def find_gene_scope(
    result: AlignmentResult,
    reference_record: SeqRecord,
    query_record: SeqRecord | None = None,
    gene_name: str | None = None,
) -> GeneScope:
    """Determine the reference-coordinate gene/CDS region(s) to scope
    mutation reporting to, and where that boundary information came from.

    If `gene_name` is given, only features whose label contains it
    (case-insensitive) count -- otherwise a richly-annotated file (e.g. one
    auto-annotated by pLannotate, which tags dozens of standard vector
    elements) would let an unrelated gene like AmpR or mEGFP stand in for
    the gene actually being checked.

    Gene boundaries come from `reference_record`'s own annotations if it
    has matching gene/CDS features (automatically true, and trivial, when
    `reference_record` is itself a gene-only entry from
    expand_with_gene_features, since it carries a self-spanning feature) --
    source="reference". If the reference has no matching annotation at all
    (e.g. a plain FASTA whole-plasmid reference), falls back to projecting
    `query_record`'s matching gene annotation into reference coordinates
    instead, so every check happens in one coordinate system -- this also
    correctly flags a deletion so large it removes the entire gene, since
    that's still an interval overlap even though no single query point
    inside the (now-absent) gene exists to anchor to -- source="query". If
    neither side has a matching gene/CDS annotation, returns an empty scope
    with source="none": this is a meaningfully different state from "found
    the gene and it's clean" and callers should surface it as such, rather
    than silently treating "nothing to filter against" the same as "no
    mutations found in the gene".
    """
    ref_intervals = _gene_intervals(reference_record, gene_name)
    if ref_intervals:
        return GeneScope(intervals=ref_intervals, source="reference")

    if query_record is not None:
        query_intervals = _gene_intervals(query_record, gene_name)
        if query_intervals:
            query_len = len(_sanitize_sequence(str(query_record.seq)))
            mapped: list[tuple[int, int]] = []
            for q_start, q_end in query_intervals:
                a = _map_query_pos_to_ref_pos(q_start, result.alignment, query_len, result.query_strand)
                b = _map_query_pos_to_ref_pos(q_end - 1, result.alignment, query_len, result.query_strand)
                if a is not None and b is not None:
                    mapped.append((min(a, b), max(a, b) + 1))
            if mapped:
                return GeneScope(intervals=mapped, source="query")

    return GeneScope(intervals=[], source="none")


def scope_mutations_to_genes(mutations: list[Mutation], gene_scope: GeneScope) -> list[Mutation]:
    """Keep only mutations overlapping `gene_scope`'s intervals.

    If `gene_scope.source == "none"`, no gene boundaries could be
    established at all, so this returns `mutations` unfiltered as a safe
    default -- but callers should check `gene_scope.source` themselves to
    tell "no mutations found in the gene" (source is "reference"/"query",
    result is empty) apart from "couldn't verify where the gene even is"
    (source is "none"), since an empty result reads as reassuring and would
    be misleading in the latter case.
    """
    if gene_scope.source == "none":
        return mutations
    return [m for m in mutations if any(_spans_overlap(_mutation_ref_span(m), gi) for gi in gene_scope.intervals)]


def find_truncation(result: AlignmentResult, min_missing_bp: int = 1) -> Truncation | None:
    """Detect whether the match falls short of the reference's full length
    at either end (e.g. a gene cut off by a cloning error or partial read).
    Returns None if the reference is covered end-to-end."""
    blocks = result.alignment.aligned[1]
    if len(blocks) == 0:
        return None
    start, end = int(blocks[0][0]), int(blocks[-1][1])
    missing_start = start
    missing_end = result.reference_length - end
    if missing_start < min_missing_bp and missing_end < min_missing_bp:
        return None
    return Truncation(missing_start_bp=missing_start, missing_end_bp=missing_end)


def _oriented_query(query: SeqRecord, result: AlignmentResult) -> SeqRecord:
    """Return `query` in whichever orientation `result`'s alignment actually
    used, so its coordinates (in alignment.aligned) line up correctly."""
    if result.query_strand != "-":
        return query
    oriented = query.reverse_complement()
    oriented.id = query.id
    oriented.description = f"{query.description} (reverse complement)"
    return oriented


def get_unmatched_flanks(
    query: SeqRecord,
    best: AlignmentResult,
    min_fragment_len: int = 50,
) -> list[SeqRecord]:
    """Return query subsequences that fall outside the best reference alignment.

    These flanking, unaligned fragments are the candidates worth screening for
    contamination (e.g. host-cell human DNA) since they're not explained by
    any reference sequence.
    """
    oriented_query = _oriented_query(query, best)
    query_seq = str(oriented_query.seq)
    blocks = best.alignment.aligned[0]
    if len(blocks) == 0:
        start, end = 0, 0
    else:
        start, end = int(blocks[0][0]), int(blocks[-1][1])

    fragments: list[SeqRecord] = []
    if start >= min_fragment_len:
        fragments.append(_subrecord(oriented_query, 0, start, "upstream_unmatched"))
    if len(query_seq) - end >= min_fragment_len:
        fragments.append(_subrecord(oriented_query, end, len(query_seq), "downstream_unmatched"))
    return fragments


def get_matched_query_region(query: SeqRecord, result: AlignmentResult) -> SeqRecord:
    """Return the portion of the query that aligned to the reference in `result`."""
    oriented_query = _oriented_query(query, result)
    blocks = result.alignment.aligned[0]
    if len(blocks) == 0:
        start, end = 0, 0
    else:
        start, end = int(blocks[0][0]), int(blocks[-1][1])
    return _subrecord(oriented_query, start, end, "matched_region")


def _subrecord(query: SeqRecord, start: int, end: int, tag: str) -> SeqRecord:
    sub = query[start:end]
    sub.id = f"{query.id}_{tag}_{start}-{end}"
    sub.description = f"{tag} region of {query.id} ({start}-{end})"
    return sub


# --------------------------------------------------------------------------
# NCBI human-gene screening
#
# Entrez itself is text/ID based, so it can't search by raw sequence. To
# identify a sequence we use NCBI's remote BLAST service (qblast) restricted
# to Homo sapiens via an Entrez query, then use Bio.Entrez.efetch to pull
# back gene annotations for any hits.
# --------------------------------------------------------------------------

def configure_entrez(email: str, api_key: str | None = None) -> None:
    if not email:
        raise ValueError("An email address is required to query NCBI.")
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def screen_for_human_genes(
    sequence: str,
    identity_threshold: float = 95.0,
    max_hits: int = 5,
    hitlist_size: int = 10,
) -> list[HumanGeneHit]:
    """BLAST `sequence` against NCBI nt, restricted to Homo sapiens.

    This is a blocking network call to NCBI and can take from tens of
    seconds up to a few minutes depending on server load.
    """
    if not Entrez.email:
        raise RuntimeError("Call configure_entrez(email) before querying NCBI.")

    result_handle = NCBIWWW.qblast(
        "blastn",
        "nt",
        sequence,
        entrez_query="Homo sapiens[Organism]",
        hitlist_size=hitlist_size,
    )
    try:
        blast_record = NCBIXML.read(result_handle)
    finally:
        result_handle.close()

    hits: list[HumanGeneHit] = []
    for alignment in blast_record.alignments:
        hsp = alignment.hsps[0]
        percent_identity = 100 * hsp.identities / hsp.align_length if hsp.align_length else 0.0
        if percent_identity < identity_threshold:
            continue
        hits.append(
            HumanGeneHit(
                accession=alignment.accession,
                description=alignment.hit_def,
                percent_identity=percent_identity,
                e_value=hsp.expect,
                align_length=hsp.align_length,
            )
        )
        if len(hits) >= max_hits:
            break

    return hits


def fetch_gene_annotation(accession: str) -> str:
    """Fetch a short gene-name summary for an accession via Entrez.efetch."""
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
    try:
        record = SeqIO.read(handle, "genbank")
    finally:
        handle.close()

    gene_names = sorted({
        feature.qualifiers["gene"][0]
        for feature in record.features
        if feature.type == "gene" and "gene" in feature.qualifiers
    })
    genes = ", ".join(gene_names) if gene_names else "n/a"
    return f"{record.description} | genes: {genes}"


def fetch_gene_sequence_by_symbol(gene_symbol: str, organism: str = "Homo sapiens") -> SeqRecord:
    """Resolve a gene symbol/name (e.g. "AGGF1") to an actual nucleotide
    sequence via Entrez, for reference-free searching when no local
    reference file is available to supply one.

    Prefers a RefSeq mRNA record (spliced, exon-only) over the gene's full
    genomic locus -- a cloned insert carries the coding sequence, not the
    introns, so a genomic hit would essentially never exact-match a read.
    Falls back to any nucleotide hit for the symbol if no mRNA record is
    found. Raises ValueError if nothing comes back at all.
    """
    if not Entrez.email:
        raise RuntimeError("Call configure_entrez(email) before querying NCBI.")

    def _search(term: str) -> list[str]:
        handle = Entrez.esearch(db="nucleotide", term=term, retmax=1, sort="relevance")
        try:
            result = Entrez.read(handle)
        finally:
            handle.close()
        return result.get("IdList", [])

    ids = _search(
        f'{gene_symbol}[Gene Name] AND {organism}[Organism] AND biomol_mrna[PROP] AND refseq[Filter]'
    )
    if not ids:
        ids = _search(f'{gene_symbol}[Gene Name] AND {organism}[Organism]')
    if not ids:
        raise ValueError(f"No NCBI nucleotide record found for gene '{gene_symbol}' ({organism}).")

    fetch_handle = Entrez.efetch(db="nucleotide", id=ids[0], rettype="fasta", retmode="text")
    try:
        return SeqIO.read(fetch_handle, "fasta")
    finally:
        fetch_handle.close()


# --------------------------------------------------------------------------
# Reference-free gene search
#
# For when no local reference plasmid/gene file is available: the gene's
# sequence is resolved from NCBI by symbol (fetch_gene_sequence_by_symbol
# above) instead, and matched directly against each query read by exact
# string search rather than alignment.
# --------------------------------------------------------------------------

def find_gene_in_queries(gene_sequence: str, queries: list[SeqRecord]) -> list[ReferenceFreeMatch]:
    """Exact-match `gene_sequence` against each query, checking both the
    forward strand and the reverse complement -- a cloned insert can land in
    either orientation relative to how the plasmid happens to be numbered,
    so checking only the forward strand would miss a real match half the
    time. Returns one ReferenceFreeMatch per query, in the same order.
    """
    probe = _sanitize_sequence(gene_sequence)
    if not probe:
        raise ValueError("Gene of interest sequence is empty.")
    revcomp_probe = str(Seq(probe).reverse_complement())

    matches: list[ReferenceFreeMatch] = []
    for query in queries:
        seq = _sanitize_sequence(str(query.seq))

        idx = seq.find(probe)
        orientation = "Forward"
        if idx == -1:
            idx = seq.find(revcomp_probe)
            orientation = "Reverse"

        if idx == -1:
            matches.append(ReferenceFreeMatch(query_id=query.id, found=False))
        else:
            matches.append(ReferenceFreeMatch(
                query_id=query.id, found=True,
                start=idx + 1, end=idx + len(probe), orientation=orientation,
            ))
    return matches


def _lookup_gene_for_accession(accession: str) -> tuple[str | None, str | None]:
    """Resolve a nucleotide accession to its linked NCBI Gene symbol/name.

    Follows the standard three-step E-utilities chain: nucleotide accession
    -> nucleotide UID (esearch) -> linked Gene UID (elink) -> Gene summary
    (esummary). Returns (None, None) if any step comes up empty (e.g. a
    non-coding hit with no linked gene record) rather than raising, so one
    unresolvable hit doesn't take down the rest of the results.
    """
    try:
        search_handle = Entrez.esearch(db="nucleotide", term=accession)
        try:
            search_result = Entrez.read(search_handle)
        finally:
            search_handle.close()
        ids = search_result.get("IdList", [])
        if not ids:
            return None, None

        link_handle = Entrez.elink(dbfrom="nucleotide", db="gene", id=ids[0])
        try:
            link_result = Entrez.read(link_handle)
        finally:
            link_handle.close()
        linksets = link_result[0].get("LinkSetDb", [])
        if not linksets:
            return None, None
        gene_uid = linksets[0]["Link"][0]["Id"]

        summary_handle = Entrez.esummary(db="gene", id=gene_uid)
        try:
            summary = Entrez.read(summary_handle)
        finally:
            summary_handle.close()
        doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
        return str(doc.get("Name", "")) or None, str(doc.get("Description", "")) or None
    except Exception:
        return None, None


def identify_gene_via_ncbi(
    sequence: str,
    organism: str | None = None,
    identity_threshold: float = 90.0,
    max_hits: int = 3,
    hitlist_size: int = 10,
) -> list[GeneIdentification]:
    """BLAST `sequence` against NCBI nt and resolve top hits to their
    official NCBI Gene symbol/name.

    Useful both to confirm a local reference's gene annotation matches
    NCBI's official naming, and to identify a sequence that isn't in any
    local reference at all. Works best on an isolated, gene-sized sequence
    rather than a whole plasmid backbone, for the same reason described in
    build_aligner: a short, focused sequence gives one unambiguous top hit
    instead of a noisy pile of unrelated vector-backbone hits.

    This is a blocking network call to NCBI and can take from tens of
    seconds up to a few minutes depending on server load.
    """
    if not Entrez.email:
        raise RuntimeError("Call configure_entrez(email) before querying NCBI.")

    qblast_kwargs = {"hitlist_size": hitlist_size}
    if organism:
        qblast_kwargs["entrez_query"] = f"{organism}[Organism]"

    result_handle = NCBIWWW.qblast("blastn", "nt", sequence, **qblast_kwargs)
    try:
        blast_record = NCBIXML.read(result_handle)
    finally:
        result_handle.close()

    identifications: list[GeneIdentification] = []
    for alignment in blast_record.alignments:
        hsp = alignment.hsps[0]
        percent_identity = 100 * hsp.identities / hsp.align_length if hsp.align_length else 0.0
        if percent_identity < identity_threshold:
            continue

        gene_symbol, gene_name = _lookup_gene_for_accession(alignment.accession)

        identifications.append(
            GeneIdentification(
                accession=alignment.accession,
                hit_description=alignment.hit_def,
                percent_identity=percent_identity,
                e_value=hsp.expect,
                align_length=hsp.align_length,
                gene_symbol=gene_symbol,
                gene_name=gene_name,
            )
        )
        if len(identifications) >= max_hits:
            break

    return identifications


# --------------------------------------------------------------------------
# Restriction analysis & virtual gel
# --------------------------------------------------------------------------

# A curated panel of widely-stocked commercial cloning enzymes, rather than
# Bio.Restriction's full ~600-enzyme commercial batch -- that panel is
# comprehensive enough to bury the handful of practically useful cutters
# (unique/rare cutters, standard MCS enzymes) in noise for a several-kb
# plasmid.
COMMON_ENZYMES = [
    "EcoRI", "BamHI", "HindIII", "XhoI", "XbaI", "SalI", "PstI", "SacI", "KpnI",
    "SmaI", "NotI", "NcoI", "NdeI", "SpeI", "ApaI", "ClaI", "EcoRV", "NheI",
    "BglII", "AvrII", "MluI", "NsiI", "PacI", "AscI", "FseI", "SbfI", "BsrGI",
    "AflII", "ApaLI", "BstEII", "DraI", "EagI", "HpaI", "MfeI", "NruI", "PmeI",
    "PvuI", "PvuII", "ScaI", "SphI", "SspI", "StuI", "XmaI",
]

DNA_LADDERS: dict[str, list[int]] = {
    "50 bp Ladder": [1350, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    "100 bp Ladder": [1517, 1200, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100],
    "1 kb Ladder": [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 500],
    "1 kb Plus Ladder": [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1200, 1000, 850, 650, 500, 400, 300, 200, 100],
}


def is_circular_record(record: SeqRecord) -> bool:
    """Whether a parsed record is annotated as circular (set by GenBank's
    topology field; FASTA carries no such annotation and defaults to linear).
    Plasmids are physically circular, which changes fragment counts for a
    restriction digest -- N cut sites yield N fragments, not N+1."""
    return str(record.annotations.get("topology", "linear")).lower() == "circular"


@st.cache_data(show_spinner=False)
def map_restriction_sites(
    sequence: str, enzyme_names: list[str] | None = None, circular: bool = False
) -> pd.DataFrame:
    """Scan `sequence` against a panel of restriction enzymes (COMMON_ENZYMES
    by default) and return a DataFrame of every enzyme that actually cuts it:
    its recognition site, cut count, exact 1-based cut positions, and whether
    it's a unique cutter (exactly one site -- the ones most useful for
    linearizing a plasmid or dropping in a single site for cloning).

    Enzymes with zero sites are omitted rather than listed with a 0, since a
    "which enzymes cut this" table is what's actually useful here.

    Cached: this re-runs on every script rerun for every query/reference in
    the results loop (any unrelated widget interaction elsewhere on the page
    triggers a full rerun), not just when "Run analysis" is clicked, so
    without caching it silently redoes the same ~40-enzyme scan over and
    over. All args here are plain str/list[str]/bool, so no hash_funcs
    override is needed.
    """
    seq = Seq(_sanitize_sequence(sequence))
    batch = RestrictionBatch(enzyme_names or COMMON_ENZYMES)
    analysis = Analysis(batch, seq, linear=not circular)

    rows = []
    for enzyme, sites in analysis.full().items():
        if not sites:
            continue
        rows.append({
            "Enzyme": str(enzyme),
            "Recognition Site": str(enzyme.site),
            "Cut Sites": len(sites),
            "Positions (bp)": ", ".join(str(s) for s in sites),
            "Unique Cutter": len(sites) == 1,
        })

    df = pd.DataFrame(rows, columns=["Enzyme", "Recognition Site", "Cut Sites", "Positions (bp)", "Unique Cutter"])
    if not df.empty:
        df = df.sort_values(
            ["Unique Cutter", "Cut Sites", "Enzyme"], ascending=[False, True, True]
        ).reset_index(drop=True)
    return df


def digest_fragments(sequence: str, enzyme_names: list[str], circular: bool = False) -> list[int]:
    """Simulate a multi-enzyme restriction digest, returning fragment lengths
    (bp) in no particular order (a gel doesn't care about fragment order,
    only size).

    With no enzymes selected, or none of the selected enzymes cutting, the
    "digest" is just the uncut sequence -- one fragment, full length. A
    circular molecule cut at N sites yields N fragments (the cut linearizes
    the loop rather than adding an end piece); a linear one yields N+1.
    """
    seq_len = len(_sanitize_sequence(sequence))
    if not enzyme_names or seq_len == 0:
        return [seq_len]

    seq = Seq(_sanitize_sequence(sequence))
    batch = RestrictionBatch(enzyme_names)
    analysis = Analysis(batch, seq, linear=not circular)
    sites = sorted({pos for positions in analysis.full().values() for pos in positions})
    if not sites:
        return [seq_len]

    if circular:
        return [
            (sites[i + 1] if i + 1 < len(sites) else sites[0] + seq_len) - sites[i]
            for i in range(len(sites))
        ]
    bounds = [0, *sites, seq_len]
    return [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]


_GEL_BAND_COLOR = "#39ff14"  # neon green, UV-transilluminator style

# Explicit tick milestones for the gel's y-axis -- Plotly's automatic log-scale
# ticks pack in a dense, overlapping run of minor-decade numbers (2, 3, 4, ...,
# 20, 30, ...) that read as noise next to actual ladder sizes.
_GEL_YAXIS_TICKVALS = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
_GEL_YAXIS_TICKTEXT = ["100 bp", "250 bp", "500 bp", "1 kb", "1.5 kb", "2 kb", "3 kb", "4 kb", "5 kb", "6 kb", "8 kb", "10 kb"]


def build_virtual_gel_figure(lanes: dict[str, list[int]]) -> go.Figure:
    """Render a simulated agarose gel: one categorical x-axis 'lane' per
    entry in `lanes` (insertion order, so the ladder should be passed first),
    each fragment/band drawn as a thick horizontal marker on a reversed log
    y-axis, glowing neon green against a dark background like a UV
    transilluminator image.
    """
    fig = go.Figure()
    lane_names = list(lanes.keys())

    for lane in lane_names:
        sizes = sorted(lanes[lane], reverse=True)
        fig.add_trace(go.Scatter(
            x=[lane] * len(sizes),
            y=sizes,
            mode="markers",
            marker=dict(
                symbol="line-ew",
                size=34,
                line=dict(width=6, color=_GEL_BAND_COLOR),
                color=_GEL_BAND_COLOR,
            ),
            name=lane,
            showlegend=False,
            hovertext=[f"{s:,} bp" for s in sizes],
            hoverinfo="text",
        ))

    fig.update_xaxes(
        type="category",
        categoryarray=lane_names,
        title_text="Lane",
        showgrid=False,
        color="#cfcfcf",
    )
    fig.update_yaxes(
        type="log",
        autorange="reversed",
        tickmode="array",
        tickvals=_GEL_YAXIS_TICKVALS,
        ticktext=_GEL_YAXIS_TICKTEXT,
        minor=dict(showgrid=False, ticks=""),
        title_text="Fragment size (bp)",
        gridcolor="#2a2a2a",
        color="#cfcfcf",
    )
    fig.update_layout(
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#e0e0e0", family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        height=520,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig
