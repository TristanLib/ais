# Authorial Polish and De-Template Workflow

Generated: 2026-05-17T02:48:32.132551+00:00

Purpose: add a mandatory manuscript-polishing step that removes generated-report flavour, repetitive phrasing, and repository-facing wording while preserving research integrity and the required AI-use declaration.

This workflow is not a way to hide material assistance or bypass disclosure. The AI-use declaration remains in the manuscript where required. The goal is to make the text read like an authored navigation-science paper: specific, concise, claim-bounded, and consistent with the evidence.

## Step Added to the Full Workflow

Run this after automated manuscript generation and before final language editing:

1. Evidence lock: confirm every number in the abstract, results, tables and conclusion maps to `outputs/audit/`, `outputs/final_multiday/`, `outputs/final_risk/`, or `outputs/final_submission/`.
2. Authorial voice pass: replace generic phrases such as "this paper presents" when overused, vary paragraph openings, remove redundant cautionary sentences, and make transitions reflect the actual argument.
3. Journal-reader pass: replace repository-facing wording with navigation-facing wording. Keep file paths in data/code availability or supplementary notes rather than in the main argument.
4. Claim-boundary pass: keep limitations precise, but avoid defensive repetition. The paper should be conservative without sounding apologetic.
5. Figure/table pass: make sure every figure is introduced by a substantive sentence and followed by an interpretation, not merely a caption restatement.
6. Human metadata pass: replace placeholders for authors, affiliations, funding, acknowledgements and contributions.
7. AI-use integrity pass: keep the AI-use declaration accurate and transparent. Do not remove it just to make the manuscript sound less generated.
8. Final read-aloud pass: read each section aloud and shorten sentences that sound like templated prose.

## Typical Edits to Make Manually

- Replace broad claims with exact claims tied to the current evidence pack.
- Convert list-like paragraphs into a narrative argument.
- Remove repeated phrases such as "under this controlled protocol" if they appear too often in nearby paragraphs.
- Replace "the project provides" with the actual research contribution where possible.
- Check that British English spelling is consistent for the JON version.
- Keep technical terms such as AIS, CPA, TCPA, ADE and FDE stable.

## What Not To Do

- Do not remove limitations that protect the claim boundary.
- Do not add live-AIS, autonomous collision-avoidance, all-day, seasonal, or architecture-superiority claims unless new evidence is generated.
- Do not remove the AI-use declaration if AI tools contributed to drafting, code generation, analysis or figure preparation.
