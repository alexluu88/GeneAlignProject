"""Streamlit app for QC-checking Plasmidsaurus sequencing results.

Upload Plasmidsaurus FASTA/GenBank files, align them locally against
uploaded reference sequences, and optionally screen any unmatched regions
against NCBI for human-derived contamination.
"""

from __future__ import annotations

import time

import streamlit as st

import analyzer


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
    "Upload Plasmidsaurus FASTA/GenBank results, align them against your reference "
    "sequences, and flag any unmatched regions for NCBI human-gene screening."
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
            "Only differences overlapping a gene/CDS feature whose annotation label "
            "contains this name (from the reference or, failing that, the query) are "
            "reported as mutations. Leave blank to report differences against any "
            "annotated gene/CDS, or against the whole reference if none are annotated."
        ),
    )

    st.subheader("NCBI screening")
    entrez_email = st.text_input("Entrez email (required for NCBI queries)", value="")
    entrez_api_key = st.text_input("Entrez API key (optional)", value="", type="password")
    min_fragment_len = st.number_input("Min. unmatched fragment length (bp)", min_value=10, value=50, step=10)

# ---------------------------------------------------------------------------
# Main: upload + run
# ---------------------------------------------------------------------------

reference_files = st.file_uploader(
    "Upload reference sequence(s) (known-good plasmid/gene)",
    type=["fasta", "fa", "fna", "gb", "gbk", "genbank"],
    accept_multiple_files=True,
)

uploaded_files = st.file_uploader(
    "Upload Plasmidsaurus result files",
    type=["fasta", "fa", "fna", "gb", "gbk", "genbank"],
    accept_multiple_files=True,
)

run_clicked = st.button(
    "Run analysis", type="primary", disabled=not (uploaded_files and reference_files)
)

if run_clicked:
    progress_bar = st.progress(0.0, text="Starting analysis...")

    references = []
    for reference_file in reference_files:
        try:
            references.extend(
                analyzer.parse_records_from_bytes(reference_file.getvalue(), reference_file.name)
            )
        except ValueError as exc:
            st.error(f"{reference_file.name}: {exc}")

    if not references:
        progress_bar.empty()
        st.error("No valid reference sequences uploaded.")
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

    # Parse every uploaded file up front so the total record count is known
    # before starting alignment -- needed to drive a determinate progress bar.
    pending_queries = []  # list of query records across all uploaded files
    for uploaded_file in uploaded_files:
        try:
            query_records = analyzer.parse_records_from_bytes(
                uploaded_file.getvalue(), uploaded_file.name
            )
        except ValueError as exc:
            st.error(f"{uploaded_file.name}: {exc}")
            continue
        pending_queries.extend(query_records)

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

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

analysis_results = st.session_state.get("analysis_results", [])
comparison_refs = st.session_state.get("comparison_refs", [])

for query, results, unmatched_fragments in analysis_results:
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

    raw_mutations = analyzer.find_mutations(best, max_report=1000)
    ref_record = next((r for r in comparison_refs if r.id == best.reference_id), None)
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
    elif status == "GENE_FOUND":
        summary_lines.append(
            f"Gene found: **{best.reference_id}** matches at {best.identity_pct:.1f}% identity "
            f"across {best.reference_coverage_pct:.1f}% of its own length, but only covers "
            f"{best.query_coverage_pct:.1f}% of the query. The rest of the query (vector "
            "backbone) differs from this reference — expected if this is a different construct "
            "that shares this gene/insert."
        )
    else:
        summary_lines.append(f"No confident match found against **{best.reference_id}**.")

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

    confirm_key = f"confirm_{query.id}_{best.reference_id}"
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

    with st.expander("📄 Raw alignment text (best match)"):
        strand_note = (
            "Matched on the query's reverse complement strand (the insert is oriented "
            "opposite to how the query sequence is numbered — normal for circular "
            "assemblies, not an error)."
            if best.query_strand == "-"
            else "Matched on the query as given (forward strand)."
        )
        st.caption(strand_note)
        st.text(str(best.alignment))

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

            screen_key = f"screen_{query.id}_{frag.id}"
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

            identify_key = f"identify_{query.id}_{frag.id}"
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
    elif status == "FLAGGED":
        st.info("Below threshold, but no unmatched flanking region met the minimum fragment length.")
