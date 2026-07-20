"""Core analysis logic for Plasmidsaurus sequencing QC.

Handles FASTA/GenBank parsing, local pairwise alignment against a folder of
reference sequences, and screening of unmatched regions against NCBI for
human-derived contamination.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Align import Alignment, PairwiseAligner, substitution_matrices
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

REFERENCE_GLOBS = {
    "fasta": ("*.fasta", "*.fa", "*.fna"),
    "genbank": ("*.gb", "*.gbk", "*.genbank"),
}


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


# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------

def parse_records_from_bytes(data: bytes, filename: str) -> list[SeqRecord]:
    """Parse an uploaded FASTA or GenBank file's raw bytes into SeqRecords."""
    suffix = Path(filename).suffix.lower()
    fmt = "genbank" if suffix in (".gb", ".gbk", ".genbank") else "fasta"

    text = data.decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(text), fmt))

    if not records:
        raise ValueError(f"No sequences found in {filename} (parsed as {fmt}).")
    return records


def load_reference_records(folder: str | Path) -> list[SeqRecord]:
    """Load all FASTA/GenBank reference sequences from a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Reference folder not found: {folder}")

    records: list[SeqRecord] = []
    for fmt, patterns in REFERENCE_GLOBS.items():
        for pattern in patterns:
            for path in sorted(folder.glob(pattern)):
                records.extend(SeqIO.parse(str(path), fmt))

    if not records:
        raise ValueError(f"No reference FASTA/GenBank files found in {folder}")
    return records


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


def compare_to_references(
    query: SeqRecord,
    references: list[SeqRecord],
    aligner: PairwiseAligner,
) -> list[AlignmentResult]:
    """Align `query` locally against every reference, best score first."""
    results = [align_pair(query, ref, aligner) for ref in references]
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def classify_match(
    result: AlignmentResult,
    identity_threshold: float,
    coverage_threshold: float,
) -> str:
    """Classify a best-match result as "PASS", "GENE_FOUND", or "FLAGGED".

    PASS: the reference matches across (almost) the whole query.
    GENE_FOUND: the reference matches at high identity along its own full
        length (e.g. a gene/insert), but covers only part of the query,
        meaning the rest of the query (e.g. vector backbone) differs.
    FLAGGED: neither the query nor the reference is well covered — no
        confident match.
    """
    if result.identity_pct < identity_threshold:
        return "FLAGGED"
    if result.query_coverage_pct >= coverage_threshold:
        return "PASS"
    if result.reference_coverage_pct >= coverage_threshold:
        return "GENE_FOUND"
    return "FLAGGED"


_STATUS_RANK = {"PASS": 0, "GENE_FOUND": 1, "FLAGGED": 2}


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
