"""Streamlit app for QC-checking Plasmidsaurus sequencing results.

Takes uploaded Plasmidsaurus FASTA/GenBank files, aligns them locally
against uploaded (or pasted) reference sequences, and optionally screens
any unmatched regions against NCBI for human-derived contamination. Built
to run on Streamlit Cloud, so all file input is via st.file_uploader --
no local filesystem access.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import analyzer


def _sticky_expander(label: str, state_key: str):
    """st.expander that stays open across reruns triggered by a widget
    nested inside it.

    st.expander's `expanded` argument is only an *initial* hint that gets
    re-evaluated fresh on every rerun; without tracking it explicitly, any
    nested widget (a toggle, multiselect, ...) triggering a rerun makes the
    expander default back to closed and visibly slam shut mid-interaction.
    Pair this with `_mark_expanded(state_key)` as that widget's `on_change`.
    """
    st.session_state.setdefault(state_key, False)
    return st.expander(label, expanded=st.session_state[state_key])


def _mark_expanded(state_key: str) -> None:
    st.session_state[state_key] = True


def _render_gene_identifications(identifications: list[analyzer.GeneIdentification]) -> None:
    if not identifications:
        st.info("No high-confidence NCBI hits found.")
        return
    st.table(
        [
            {
                "Accession": g.accession,
                "NCBI hit description": g.hit_description,
                "Gene symbol": g.gene_symbol or "(no linked Gene record)",
                "Gene name": g.gene_name or "-",
                "Identity (%)": round(g.percent_identity, 1),
                "E-value": g.e_value,
            }
            for g in identifications
        ]
    )

st.set_page_config(page_title="Plasmidsaurus QC", layout="wide")

# Streamlit's built-in top-right "running" status indicator has no official
# API toggle, so it's hidden via a direct CSS override on its element.
st.markdown(
    "<style>[data-testid='stStatusWidget'] { visibility: hidden; }</style>",
    unsafe_allow_html=True,
)

st.title("Plasmidsaurus Sequencing QC")
st.caption(
    "Upload Plasmidsaurus FASTA/GenBank result and reference files, align them, and "
    "flag any unmatched regions for NCBI human-gene screening."
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    st.subheader("Match thresholds")
    identity_threshold = st.slider("Minimum identity (%)", 0, 100, 95)
    coverage_threshold = st.slider(
        "Minimum coverage (%)",
        0, 100, 90,
        help=(
            "Applied to query coverage (whole-plasmid match) or reference "
            "coverage (a gene/insert found within a differing backbone), "
            "whichever is higher."
        ),
    )

    st.subheader("Mutation reporting")
    gene_name_filter = st.text_input(
        "Gene of interest",
        value="AGGF1",
        help=(
            "With a reference provided: only differences overlapping a gene/CDS feature "
            "whose annotation label contains this name (from the reference or, failing "
            "that, the query) are reported as mutations. Leave blank to report "
            "differences against any annotated gene/CDS, or against the whole reference "
            "if none are annotated.\n\n"
            "Without a reference: this gene's sequence is looked up on NCBI by name and "
            "searched for directly in each query read (Reference-Free Gene Search) -- "
            "requires an Entrez email below."
        ),
    )

    st.subheader("NCBI screening")
    entrez_email = st.text_input("Entrez email (required for NCBI queries)", value="")
    entrez_api_key = st.text_input("Entrez API key (optional)", value="", type="password")
    min_fragment_len = st.number_input("Min. unmatched fragment length (bp)", min_value=10, value=50, step=10)

# ---------------------------------------------------------------------------
# Main: upload + run
# ---------------------------------------------------------------------------

_SEQUENCE_FILE_TYPES = [g.split(".")[-1] for g in analyzer.SEQUENCE_FILE_GLOBS]
_IMAGE_FILE_EXTENSIONS = {"png", "jpg", "jpeg"}


def _sort_sequence_uploads(uploaded_files) -> tuple[list, list]:
    """Split a raw Plasmidsaurus folder upload into sequence files and gel images.

    Plasmidsaurus result folders bundle undigested gel simulation images
    alongside the actual sequence files (and occasionally OS cruft like
    .DS_Store); only the two recognized kinds are kept, everything else is
    silently dropped rather than passed to the analyzer or displayed.
    """
    sequence_files = []
    image_files = []
    for uploaded_file in uploaded_files:
        ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
        if ext in _SEQUENCE_FILE_TYPES:
            sequence_files.append(uploaded_file)
        elif ext in _IMAGE_FILE_EXTENSIONS:
            image_files.append(uploaded_file)
    return sequence_files, image_files

reference_input_method = st.radio(
    "Reference input method",
    ["Upload Reference File", "Paste Oligo/Sequence"],
    horizontal=True,
)

oligo_name = ""
oligo_sequence = ""
if reference_input_method == "Upload Reference File":
    reference_files = st.file_uploader(
        "Reference sequence file(s) (known-good plasmid/gene FASTA/GenBank files) — "
        "optional if a Gene of interest is set in the sidebar. A raw, unedited Plasmidsaurus "
        "folder can be dropped here too -- non-sequence files (gel images, .txt notes, "
        "DS_Store, etc.) are accepted and then ignored automatically.",
        # No `type=` restriction: a raw folder-select mixes in gel images, .txt
        # notes, .DS_Store, etc., and Streamlit's frontend would otherwise
        # reject every one of those with a red error chip before Python ever
        # sees them. Sorting into sequence/image/ignored is handled entirely
        # by _sort_sequence_uploads below instead.
        accept_multiple_files=True,
    )
else:
    reference_files = []
    oligo_name = st.text_input("Oligo name")
    oligo_sequence = st.text_area(
        "DNA sequence",
        help="Pasted freely -- spaces, line breaks, and stray position numbers are stripped automatically.",
    )

reference_sequence_files, _ = _sort_sequence_uploads(reference_files or [])
reference_provided = bool(reference_sequence_files) or bool(oligo_name and oligo_sequence)

query_files = st.file_uploader(
    "Plasmidsaurus result file(s) (query FASTA/GenBank files -- drop a raw, unedited "
    "Plasmidsaurus folder here as-is; gel simulation images are shown separately below "
    "and anything else, like .txt notes or DS_Store, is ignored)",
    # No `type=` restriction -- see the comment on the reference uploader above.
    accept_multiple_files=True,
)
st.caption(
    "Tip: clicking \"Browse files\" above lets you select an entire folder -- every "
    "FASTA/GenBank file inside it (recursively) is uploaded at once. Dragging a folder "
    "directly onto the drop zone isn't supported; drag its individual files instead."
)

query_sequence_files, query_image_files = _sort_sequence_uploads(query_files or [])

# st.file_uploader has no public API for folder selection, but its compiled
# frontend already has an internal `acceptDirectory` prop that isn't wired up
# to anything from Python yet -- when set, it renders the underlying <input>
# with the standard `webkitdirectory` attribute, which is what actually makes
# a browser's file dialog let you pick a folder (and hand back every file
# inside it). This sets that attribute directly on the input via its stable,
# Streamlit-testing-owned `data-testid` (far less likely to change across
# versions than emotion-hashed class names), rather than waiting for a
# Python-level parameter that doesn't exist yet.
#
# Only the "Browse files" click path is affected -- Streamlit's drag-and-drop
# handling is separate and still expects individual files. Re-scans on an
# interval (rather than patching once) because Streamlit can remount this
# input on a rerun, which would otherwise silently drop the attribute; the
# whole thing is wrapped in try/catch so an unrelated future frontend change
# just falls back to normal multi-file selection instead of breaking.
components.html(
    """
    <script>
    (function () {
        function patchDirectoryInputs() {
            try {
                const inputs = window.parent.document.querySelectorAll(
                    'input[data-testid="stFileUploaderDropzoneInput"]:not([data-dir-patch-applied])'
                );
                inputs.forEach((input) => {
                    input.setAttribute("webkitdirectory", "");
                    input.setAttribute("directory", "");
                    input.setAttribute("mozdirectory", "");
                    input.setAttribute("data-dir-patch-applied", "1");
                });
            } catch (err) {
                // Streamlit's internal markup changed or isn't ready yet --
                // fail silently and fall back to normal file selection.
            }
        }
        const interval = setInterval(patchDirectoryInputs, 500);
        setTimeout(() => clearInterval(interval), 3600000);
    })();
    </script>
    """,
    height=0,
)

run_clicked = st.button(
    "Run analysis",
    type="primary",
    disabled=not (query_sequence_files and (reference_provided or gene_name_filter)),
)


def _load_records_from_uploads(uploaded_files, label: str) -> list[analyzer.SeqRecord]:
    """Parse all uploaded FASTA/GenBank files, surfacing problems in the UI.

    Streamlit Cloud has no local filesystem for a folder path to point at,
    so files come in as in-memory UploadedFile objects instead of paths.

    Callers pre-filter to recognized sequence extensions via
    _sort_sequence_uploads, but that's a filename-only check -- it can't
    guarantee the *contents* are well-formed (e.g. a mislabeled or
    corrupted file). Biopython's parsers don't consistently raise
    ValueError for malformed input, so this catches any Exception rather
    than risk one bad file taking down the whole app for every query/
    reference in the batch.
    """
    records = []
    for uploaded_file in uploaded_files:
        try:
            records.extend(analyzer.parse_records_from_upload(uploaded_file))
        except Exception as exc:
            st.error(f"{label} — {uploaded_file.name}: couldn't be parsed ({exc}).")
    return records


if run_clicked and reference_provided:
    progress_bar = st.progress(0.0, text="Starting analysis...")

    if reference_input_method == "Upload Reference File":
        references = _load_records_from_uploads(reference_sequence_files, "Reference files")
    else:
        try:
            references = [analyzer.build_seqrecord_from_pasted_sequence(oligo_name, oligo_sequence)]
        except ValueError as exc:
            progress_bar.empty()
            st.error(f"Pasted reference: {exc}")
            st.stop()

    if not references:
        progress_bar.empty()
        st.error("No valid reference sequences found.")
        st.stop()

    comparison_refs = analyzer.expand_with_gene_features(references)
    extracted_genes = comparison_refs[len(references):]
    if extracted_genes:
        st.info(
            f"Also extracted {len(extracted_genes)} named gene/CDS feature(s) from annotated "
            f"reference(s): {', '.join(g.id for g in extracted_genes)}. Each is compared to "
            "the query individually, so a shared gene can be found even if the rest of the "
            "plasmid backbone differs."
        )

    aligner = analyzer.build_aligner(mode="local")

    # Load every query file up front so the total record count is known
    # before starting alignment -- needed to drive a determinate progress bar.
    pending_queries = _load_records_from_uploads(query_sequence_files, "Query files")

    analysis_results = []  # list of (query_record, results, unmatched_fragments)
    total = len(pending_queries)

    if total == 0:
        progress_bar.empty()
        st.warning("No valid sequences to analyze.")
    else:
        start_time = time.time()

        for i, query in enumerate(pending_queries):
            results = analyzer.compare_to_references(query, comparison_refs, aligner)

            best = analyzer.pick_best_match(results, identity_threshold, coverage_threshold)
            unmatched_fragments = []
            if best:
                status = analyzer.classify_match(best, identity_threshold, coverage_threshold)
                if status != "PASS":
                    unmatched_fragments = analyzer.get_unmatched_flanks(
                        query, best, min_fragment_len=min_fragment_len
                    )

            analysis_results.append((query, results, unmatched_fragments))

            done = i + 1
            elapsed = time.time() - start_time
            eta = elapsed / done * (total - done)
            progress_bar.progress(
                done / total,
                text=(
                    f"Analyzed {done}/{total} ({query.id}) — "
                    + (f"~{eta:.0f}s remaining" if done < total else "done")
                ),
            )

    st.session_state["analysis_results"] = analysis_results
    st.session_state["comparison_refs"] = comparison_refs
    # A prior reference-free run's results shouldn't linger under a fresh
    # reference-based run's output below.
    st.session_state["reference_free_results"] = []

elif run_clicked:
    # No reference provided, but the button is only enabled here because a
    # Gene of interest was set -- resolve its sequence from NCBI by name and
    # search for it directly in the reads instead of aligning to a reference.
    if not entrez_email:
        st.error("Enter an Entrez email in the sidebar to look up the gene of interest on NCBI.")
        st.stop()

    pending_queries = _load_records_from_uploads(query_sequence_files, "Query files")
    if not pending_queries:
        st.warning("No valid sequences to analyze.")
        st.stop()

    analyzer.configure_entrez(entrez_email, entrez_api_key or None)
    try:
        with st.spinner(f"Looking up '{gene_name_filter}' on NCBI..."):
            gene_record = analyzer.fetch_gene_sequence_by_symbol(gene_name_filter)
    except Exception as exc:  # network/NCBI errors surfaced to the user
        st.error(f"Couldn't resolve '{gene_name_filter}' via NCBI: {exc}")
        st.stop()

    reference_free_results = analyzer.find_gene_in_queries(str(gene_record.seq), pending_queries)

    st.session_state["reference_free_results"] = reference_free_results
    st.session_state["reference_free_gene"] = gene_record
    st.session_state["reference_free_gene_name"] = gene_name_filter
    # A prior reference-based run's results shouldn't linger under a fresh
    # reference-free run's output below.
    st.session_state["analysis_results"] = []

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

reference_free_results = st.session_state.get("reference_free_results", [])

if reference_free_results:
    st.divider()
    st.header("Reference-Free Gene Search")
    gene_record = st.session_state.get("reference_free_gene")
    gene_name = st.session_state.get("reference_free_gene_name", gene_name_filter)
    if gene_record is not None:
        st.caption(
            f"Searched for **{gene_name}** using NCBI sequence {gene_record.id} "
            f"({len(gene_record.seq)} bp), forward and reverse complement."
        )

    for match in reference_free_results:
        with st.container(border=True):
            st.subheader(match.query_id)
            if match.found:
                st.success(
                    f"Gene of Interest found in {match.query_id} at bp "
                    f"{match.start}-{match.end} ({match.orientation})"
                )
            else:
                st.error(f"Gene of Interest NOT found in {match.query_id}")

analysis_results = st.session_state.get("analysis_results", [])
comparison_refs = st.session_state.get("comparison_refs", [])

# Global virtual gel configuration -- depends on analysis_results (to scope
# the enzyme dropdown to what's actually found in this run), so it's built
# here rather than up with the rest of the sidebar, but still renders inside
# the sidebar. Configured once for the whole run instead of once per result,
# and cached map_restriction_sites calls make repeating them here (rather
# than passing found-enzyme sets around) essentially free.
all_found_enzymes: set[str] = set()
restriction_expanded_keys = []
for gel_i, (gel_query, gel_results, _) in enumerate(analysis_results):
    gel_best = analyzer.pick_best_match(gel_results, identity_threshold, coverage_threshold)
    if gel_best is None:
        continue
    # gel_i mirrors the enumerate index used for the same key in the main
    # results loop below -- both walk analysis_results in the same order
    # within a single script run, so the indices line up and this list
    # keeps pointing at the right per-result session-state entries even
    # when two results happen to share a query id/reference id pair.
    restriction_expanded_keys.append(f"restriction_expanded_{gel_i}_{gel_query.id}_{gel_best.reference_id}")
    all_found_enzymes |= set(analyzer.map_restriction_sites(
        str(gel_query.seq), circular=analyzer.is_circular_record(gel_query)
    )["Enzyme"])
    gel_ref_record = next((r for r in comparison_refs if r.id == gel_best.reference_id), None)
    if gel_ref_record is not None:
        all_found_enzymes |= set(analyzer.map_restriction_sites(
            str(gel_ref_record.seq), circular=analyzer.is_circular_record(gel_ref_record)
        )["Enzyme"])
found_enzymes = sorted(all_found_enzymes)


def _mark_all_expanded(state_keys: list[str]) -> None:
    for state_key in state_keys:
        st.session_state[state_key] = True


with st.sidebar:
    st.header("Virtual Gel Configuration")
    selected_enzymes = st.multiselect(
        "Enzymes for virtual digest",
        options=found_enzymes,
        default=found_enzymes[:1] if found_enzymes else [],
        help=(
            "Populated from the enzymes found across this run's queries/references that "
            "actually cut them. Applies to every result below."
        ),
        on_change=_mark_all_expanded,
        args=(restriction_expanded_keys,),
    )
    ladder_name = st.selectbox(
        "DNA ladder",
        options=list(analyzer.DNA_LADDERS.keys()),
        on_change=_mark_all_expanded,
        args=(restriction_expanded_keys,),
    )

if query_image_files:
    with st.expander("Original Plasmidsaurus Gel Simulation", expanded=False):
        for image_file in query_image_files:
            st.image(image_file, use_container_width=True)
            st.caption(image_file.name)

for i, (query, results, unmatched_fragments) in enumerate(analysis_results):
    st.divider()

    if not results:
        st.header(f"{query.id} ({len(query.seq)} bp)")
        st.warning("No reference sequences to compare against.")
        continue

    best = analyzer.pick_best_match(results, identity_threshold, coverage_threshold)
    status = analyzer.classify_match(best, identity_threshold, coverage_threshold)

    st.header(f"{query.id} ({len(query.seq)} bp)")
    st.subheader(f"Best match: {best.reference_id}")

    if status == "PASS":
        st.success("🟢 STATUS: PASS")
    elif status == "GENE_FOUND":
        st.warning("🟡 STATUS: GENE FOUND")
    else:
        st.error("🔴 STATUS: FAIL")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Identity", f"{best.identity_pct:.1f}%")
    metric_cols[1].metric("Query Coverage", f"{best.query_coverage_pct:.1f}%")
    metric_cols[2].metric("Reference Coverage", f"{best.reference_coverage_pct:.1f}%")

    ref_record = next((r for r in comparison_refs if r.id == best.reference_id), None)

    # A FAIL result means the aligner forced its best-scoring local alignment
    # against what's below-threshold on identity and/or coverage -- often a
    # genuinely unrelated sequence. Counting "mutations" against that forced
    # alignment produces thousands of meaningless entries and a summary that
    # contradicts the STATUS: FAIL banner above it, so mutation-counting is
    # skipped entirely rather than computed and then downplayed cosmetically.
    if status == "FAIL":
        gene_scope = analyzer.GeneScope(intervals=[], source="none")
        mutations: list[analyzer.Mutation] = []
        truncation = None
        summary_lines = [analyzer.build_no_match_message(best, identity_threshold, coverage_threshold)]
    else:
        raw_mutations = analyzer.find_mutations(best, max_report=1000)
        gene_scope = (
            analyzer.find_gene_scope(best, ref_record, query, gene_name=gene_name_filter or None)
            if ref_record is not None
            else analyzer.GeneScope(intervals=[], source="none")
        )
        mutations = analyzer.scope_mutations_to_genes(raw_mutations, gene_scope)
        truncation = analyzer.find_truncation(best)

        summary_lines = []
        if status == "PASS":
            summary_lines.append(f"Matches **{best.reference_id}** across the full query.")
        else:  # GENE_FOUND
            summary_lines.append(
                f"Gene found: **{best.reference_id}** matches at {best.identity_pct:.1f}% identity "
                f"across {best.reference_coverage_pct:.1f}% of its own length, but only covers "
                f"{best.query_coverage_pct:.1f}% of the query. The rest of the query (vector "
                "backbone) differs from this reference — expected if this is a different construct "
                "that shares this gene/insert."
            )

        if gene_scope.source == "none":
            gene_label = gene_name_filter or "the gene of interest"
            summary_lines.append(
                f"⚠️ Couldn't locate '{gene_label}' in either the reference or this query's "
                "annotations, so mutation reporting is unfiltered (whole-reference differences). "
                "This can mean the gene is entirely absent from this clone — not just mutated, but "
                "undetectable even as a fragment — rather than a clean match."
            )

        if not mutations and not truncation:
            label = gene_name_filter if gene_scope.source != "none" else best.reference_id
            summary_lines.append(f"✅ **{label}** matches exactly, full length, no mutations detected.")
        else:
            if truncation:
                pieces = []
                if truncation.missing_start_bp:
                    pieces.append(f"{truncation.missing_start_bp} bp missing from the start")
                if truncation.missing_end_bp:
                    pieces.append(f"{truncation.missing_end_bp} bp missing from the end")
                summary_lines.append(
                    f"✂️ **{best.reference_id}** looks cut off relative to the reference: "
                    + " and ".join(pieces) + "."
                )
            if mutations:
                total_affected_bp = sum(m.length for m in mutations)
                mutation_label = gene_name_filter if gene_scope.source != "none" else best.reference_id
                summary_lines.append(
                    f"🧬 **{mutation_label}** matches, but with {len(mutations)} distinct mutation "
                    f"event(s) affecting {total_affected_bp} bp total relative to the reference "
                    "(see table below)."
                )

    st.info("\n\n".join(summary_lines))

    st.plotly_chart(
        analyzer.build_alignment_figure(query, best, gene_scope=gene_scope, mutations=mutations),
        use_container_width=True,
        key=f"alignment_map_{i}_{query.id}_{best.reference_id}",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.caption(
        "Scroll or pinch on the map to zoom; drag to pan. Use the buttons above the "
        "map to jump straight to the gene, mutations, or a detected motif."
    )

    if mutations:
        shown = mutations[:20]

        def _fmt_seq(s: str, max_len: int = 30) -> str:
            return s if len(s) <= max_len else f"{s[:max_len]}...({len(s)}bp)"

        st.table(
            [
                {
                    "Reference position": m.ref_position,
                    "Length (bp)": m.length,
                    "Reference": _fmt_seq(m.ref_base),
                    "Query": _fmt_seq(m.query_base),
                    "Type": m.kind,
                }
                for m in shown
            ]
        )
        if len(mutations) > len(shown):
            st.caption(f"...and {len(mutations) - len(shown)} more event(s) not shown.")

    confirm_key = f"confirm_{i}_{query.id}_{best.reference_id}"
    if st.button(f"🔍 Verify {best.reference_id} via NCBI", key=confirm_key):
        if not entrez_email:
            st.error("Enter an Entrez email in the sidebar before querying NCBI.")
        else:
            matched_region = analyzer.get_matched_query_region(query, best)
            try:
                analyzer.configure_entrez(entrez_email, entrez_api_key or None)
                with st.spinner("Querying NCBI BLAST (this can take a minute)..."):
                    identifications = analyzer.identify_gene_via_ncbi(str(matched_region.seq))
            except Exception as exc:  # network/NCBI errors surfaced to the user
                st.error(f"NCBI query failed: {exc}")
            else:
                top_symbol = identifications[0].gene_symbol if identifications else None
                if top_symbol and top_symbol.upper() == best.reference_id.upper():
                    st.success(f"NCBI confirms this region matches **{top_symbol}**.")
                elif top_symbol:
                    st.warning(
                        f"NCBI's top hit resolves to gene **{top_symbol}**, which doesn't "
                        f"match the local reference name '{best.reference_id}' — worth "
                        "double-checking the reference annotation."
                    )
                _render_gene_identifications(identifications)

    with st.expander("📊 All reference alignments (summary table)"):
        st.table(
            [
                {
                    "Reference": r.reference_id,
                    "Description": r.reference_description,
                    "Score": round(r.score, 1),
                    "Identity (%)": round(r.identity_pct, 1),
                    "Query coverage (%)": round(r.query_coverage_pct, 1),
                    "Reference coverage (%)": round(r.reference_coverage_pct, 1),
                    "Strand": r.query_strand,
                }
                for r in results
            ]
        )

    raw_alignment_expanded_key = f"raw_alignment_expanded_{i}_{query.id}_{best.reference_id}"
    with _sticky_expander("📄 Raw alignment text (best match)", raw_alignment_expanded_key):
        strand_note = (
            "Matched on the query's reverse complement strand (the insert is oriented "
            "opposite to how the query sequence is numbered — normal for circular "
            "assemblies, not an error)."
            if best.query_strand == "-"
            else "Matched on the query as given (forward strand)."
        )
        st.caption(strand_note)
        only_mismatches = st.toggle(
            "Only show regions with mismatches",
            key=f"only_mismatches_{i}_{query.id}_{best.reference_id}",
            on_change=_mark_expanded,
            args=(raw_alignment_expanded_key,),
        )
        with st.container(height=500):
            alignment_html = analyzer.format_alignment_html(best, only_mismatches=only_mismatches)
            # st.markdown's markdown pass reinterprets indented-looking lines
            # (the match-bar line's leading spaces) as a markdown code block
            # and drops the inline styling, so this renders in its own
            # unprocessed HTML frame instead. Sized generously off the line
            # count -- the surrounding fixed-height container is what
            # actually clips/scrolls it, so overestimating is harmless.
            frame_height = max(120, alignment_html.count("\n") * 22 + 80)
            components.html(alignment_html, height=frame_height, scrolling=False)

    restriction_expanded_key = f"restriction_expanded_{i}_{query.id}_{best.reference_id}"
    with _sticky_expander("🧬 Restriction Analysis & Virtual Gel", restriction_expanded_key):
        st.caption(
            f"Cut sites from a panel of {len(analyzer.COMMON_ENZYMES)} common commercial "
            "restriction enzymes."
        )

        query_sites_df = analyzer.map_restriction_sites(
            str(query.seq), circular=analyzer.is_circular_record(query)
        )
        st.markdown(f"**Query — {query.id}**")
        if query_sites_df.empty:
            st.info("None of the panel's enzymes cut this sequence.")
        else:
            st.dataframe(
                query_sites_df,
                use_container_width=True,
                hide_index=True,
                key=f"query_sites_{i}_{query.id}_{best.reference_id}",
            )

        ref_sites_df = pd.DataFrame()
        if ref_record is not None:
            ref_sites_df = analyzer.map_restriction_sites(
                str(ref_record.seq), circular=analyzer.is_circular_record(ref_record)
            )
            st.markdown(f"**Reference — {ref_record.id}**")
            if ref_sites_df.empty:
                st.info("None of the panel's enzymes cut this sequence.")
            else:
                st.dataframe(
                    ref_sites_df,
                    use_container_width=True,
                    hide_index=True,
                    key=f"ref_sites_{i}_{query.id}_{best.reference_id}",
                )

        if not selected_enzymes:
            st.info("Select at least one enzyme in the sidebar's Virtual Gel Configuration to simulate a digest.")
        else:
            lanes = {ladder_name: analyzer.DNA_LADDERS[ladder_name]}
            if ref_record is not None:
                lanes[f"Reference — {ref_record.id}"] = analyzer.digest_fragments(
                    str(ref_record.seq), selected_enzymes, circular=analyzer.is_circular_record(ref_record)
                )
            lanes[f"Query — {query.id}"] = analyzer.digest_fragments(
                str(query.seq), selected_enzymes, circular=analyzer.is_circular_record(query)
            )
            st.plotly_chart(
                analyzer.build_virtual_gel_figure(lanes),
                use_container_width=True,
                key=f"gel_{i}_{query.id}_{best.reference_id}",
            )

    if unmatched_fragments:
        label = (
            "Backbone region(s) outside the matched gene — optionally screen for "
            "contamination:"
            if status == "GENE_FOUND"
            else f"{len(unmatched_fragments)} unmatched region(s) found outside the best "
            "reference alignment. These may indicate insert/contamination."
        )
        (st.info if status == "GENE_FOUND" else st.warning)(label)
        for frag in unmatched_fragments:
            st.write(f"**{frag.id}** — {len(frag.seq)} bp")
            frag_cols = st.columns(2)

            screen_key = f"screen_{i}_{query.id}_{frag.id}"
            if frag_cols[0].button("🔍 Screen for human genes via NCBI", key=screen_key):
                if not entrez_email:
                    st.error("Enter an Entrez email in the sidebar before querying NCBI.")
                else:
                    try:
                        analyzer.configure_entrez(entrez_email, entrez_api_key or None)
                        with st.spinner("Querying NCBI BLAST (this can take a minute)..."):
                            hits = analyzer.screen_for_human_genes(str(frag.seq))
                    except Exception as exc:  # network/NCBI errors surfaced to the user
                        st.error(f"NCBI query failed: {exc}")
                    else:
                        if not hits:
                            st.info("No high-confidence human-origin matches found.")
                        else:
                            st.error(f"{len(hits)} human-origin match(es) found:")
                            st.table(
                                [
                                    {
                                        "Accession": h.accession,
                                        "Description": h.description,
                                        "Identity (%)": round(h.percent_identity, 1),
                                        "E-value": h.e_value,
                                    }
                                    for h in hits
                                ]
                            )

            identify_key = f"identify_{i}_{query.id}_{frag.id}"
            if frag_cols[1].button("🔍 Identify via NCBI Gene database", key=identify_key):
                if not entrez_email:
                    st.error("Enter an Entrez email in the sidebar before querying NCBI.")
                else:
                    try:
                        analyzer.configure_entrez(entrez_email, entrez_api_key or None)
                        with st.spinner("Querying NCBI BLAST (this can take a minute)..."):
                            identifications = analyzer.identify_gene_via_ncbi(str(frag.seq))
                    except Exception as exc:  # network/NCBI errors surfaced to the user
                        st.error(f"NCBI query failed: {exc}")
                    else:
                        _render_gene_identifications(identifications)
    elif status == "FAIL":
        st.info("Below threshold, but no unmatched flanking region met the minimum fragment length.")

if analysis_results and comparison_refs:
    st.divider()
    st.header("Recommended Clones")
    st.caption(
        "For each reference, the first query (in upload/alphabetical order) that reached "
        "a PASS status against it -- the tube to proceed with."
    )

    query_outcomes = []
    for query, results, _ in analysis_results:
        best = analyzer.pick_best_match(results, identity_threshold, coverage_threshold)
        if best is None:
            continue
        status = analyzer.classify_match(best, identity_threshold, coverage_threshold)
        query_outcomes.append(analyzer.QueryOutcome(
            query_id=query.id,
            reference_id=best.reference_id,
            status=status,
            identity_pct=best.identity_pct,
        ))

    recommended_clones_df = analyzer.recommend_clones(
        query_outcomes, reference_ids=[r.id for r in comparison_refs]
    )
    st.dataframe(recommended_clones_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download summary as CSV",
        data=recommended_clones_df.to_csv(index=False).encode("utf-8"),
        file_name="clone_summary.csv",
        mime="text/csv",
    )

# The alignment map's x-axis is in bp, so zooming in past a few bp is
# meaningless -- unbounded scroll/box zoom eventually lands on a fractional
# sub-bp range (e.g. "169.236347429"), which reads as a rendering glitch.
# Plotly has no min-zoom-span layout option, so this reaches into the parent
# document (Streamlit's component iframe allows same-origin access) and
# snaps any alignment map's xaxis.range back out to a minimum span on
# relayout -- covers scroll-zoom, box-zoom, and the modebar zoom buttons.
components.html(
    """
    <script>
    (function () {
        const MIN_SPAN_BP = 4;

        function clampZoom(gd) {
            if (gd.__zoomClampAttached) return;
            gd.__zoomClampAttached = true;
            gd.on("plotly_relayout", function (evt) {
                const x0 = evt["xaxis.range[0]"];
                const x1 = evt["xaxis.range[1]"];
                if (x0 === undefined || x1 === undefined) return;
                if (x1 - x0 >= MIN_SPAN_BP) return;
                const mid = (x0 + x1) / 2;
                window.parent.Plotly.relayout(gd, {
                    "xaxis.range": [mid - MIN_SPAN_BP / 2, mid + MIN_SPAN_BP / 2],
                });
            });
        }

        const scan = () =>
            window.parent.document.querySelectorAll(".js-plotly-plot").forEach(clampZoom);
        const interval = setInterval(scan, 400);
        setTimeout(() => clearInterval(interval), 20000);
    })();
    </script>
    """,
    height=0,
)
