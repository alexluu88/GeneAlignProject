# GeneAlign

**GeneAlign** is an automated, local Streamlit web app for quality-checking [Plasmidsaurus](https://www.plasmidsaurus.com/) sequencing results against your reference plasmids. Upload your Plasmidsaurus FASTA/GenBank reads alongside a reference sequence, and GeneAlign performs local pairwise alignment, applies Pass/Fail grading logic, and — for any unmatched regions — queries NCBI to help identify unexpected gene inserts or contamination.

Everything runs locally in your browser via Streamlit; sequence data is only sent externally when the optional NCBI lookup step is used.

## Key Features

- **Automated local alignment** — Performs local pairwise alignment (via Biopython's `PairwiseAligner`) between Plasmidsaurus sequencing reads and uploaded reference plasmids, checking both strands and reporting identity and coverage statistics.
- **Pass/Fail grading logic** — Automatically classifies each result as `PASS`, `GENE_FOUND`, or `FLAGGED` based on configurable identity and coverage thresholds, giving you an at-a-glance QC verdict for every sample.
- **NCBI Entrez integration** — Screens unmatched or unexpected regions against NCBI (BLAST + Entrez) to identify unknown gene inserts, linking hits back to gene symbols and names where available.

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd GeneAlignProject
   ```

2. **Set up a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Launch the app with Streamlit:

```bash
streamlit run app.py
```

This will start a local web server and open GeneAlign in your default browser. From there:

1. Upload your Plasmidsaurus sequencing result file(s) (FASTA/GenBank).
2. Upload or select your reference plasmid sequence(s).
3. Adjust the identity and coverage thresholds in the sidebar as needed.
4. Review the Pass/Fail status for each sample, and inspect any NCBI-identified gene hits for unmatched regions.
