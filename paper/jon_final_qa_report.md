# JON Final QA Report

Generated: 2026-05-18T03:14:09.191081+00:00

Latest manuscript-path cleanup and AI-review response addendum: 2026-05-18T06:55:26Z

Final content/AI-trace audit addendum: 2026-05-18T08:34:20Z

Overall status: **pass**

## Summary

- Passed checks: 61
- Failed checks: 0
- Main manuscript word count: 6418
- Abstract words: 135
- Main PDF pages: 17
- Supplementary ZIP: 0.31 MB
- Main manuscript/DOCX/PDF path cleanup: no `outputs/`, `scripts/`, `.py`,
  `overall_status`, `readiness_report`, `final_submission`, `final_multiday`,
  `final_risk`, or `audit/` repository paths remain in the main submission
  text.
- AI-review response: incorporated useful suggestions on data
  representativeness, tail-error diagnostics, CPA/TCPA scenario definition,
  neural-baseline wording, lead-time error reporting, and AI-use wording.
- Repository-facing consistency: README entry point now describes the active
  JON/high-quality route rather than the older single-date conservative result.
- Final content/AI-trace audit: no TODO/placeholders, local filesystem paths,
  `outputs/` references, unsupported headline claims, or hype-template phrases
  were found in the main manuscript. Minor wording fixes were applied for the
  risk-scenario description, data-availability timing phrase, and single-author
  cover-letter voice.

## Checks

- PASS: file exists: paper/jon_manuscript.md -- {'size_bytes': 49409}
- PASS: file exists: paper/jon_manuscript.docx -- {'size_bytes': 804580}
- PASS: file exists: paper/jon_manuscript.pdf -- {'size_bytes': 1226245}
- PASS: file exists: paper/jon_manuscript_zh.md -- {'size_bytes': 23610}
- PASS: file exists: paper/jon_manuscript_zh.docx -- {'size_bytes': 796436}
- PASS: file exists: paper/jon_manuscript_zh.pdf -- {'size_bytes': 1448921}
- PASS: file exists: paper/jon_cover_letter.md -- {'size_bytes': 1942}
- PASS: file exists: paper/jon_submission_checklist.md -- {'size_bytes': 3206}
- PASS: file exists: paper/jon_scholarone_metadata.md -- {'size_bytes': 3562}
- PASS: file exists: paper/jon_reference_audit.md -- {'size_bytes': 3900}
- PASS: file exists: paper/jon_submission_guide_zh.md -- {'size_bytes': 3823}
- PASS: file exists: paper/jon_supplementary_materials.md -- {'size_bytes': 4179}
- PASS: file exists: paper/jon_supplementary_materials.zip -- {'size_bytes': 326376}
- PASS: file exists: outputs/final_submission/jon_submission_manifest.json -- {'size_bytes': 2734}
- PASS: file exists: outputs/final_submission/readiness_report.json -- {'size_bytes': 9749}
- PASS: file exists: paper/references.bib -- {'size_bytes': 9851}
- PASS: main manuscript word count within JON target band -- {'word_count': 6418, 'target': '6000-8000'}
- PASS: abstract around 150 words -- {'abstract_words': 135}
- PASS: short title under 40 characters -- {'short_title': 'AIS Benchmark for Risk Warning', 'characters': 30}
- PASS: no obvious author/funding/submission placeholders remain -- []
- PASS: unsupported headline claims absent -- []
- PASS: numeric/text claim present: sample count -- {'value': '186,326', 'scope': 'both'}
- PASS: numeric/text claim present: MMSI count -- {'value': '7,425', 'scope': 'both'}
- PASS: numeric/text claim present: source date Jan 2 -- {'value': '2024-01-02', 'scope': 'manuscript'}
- PASS: numeric/text claim present: source date Jan 9 -- {'value': '2024-01-09', 'scope': 'manuscript'}
- PASS: numeric/text claim present: source date Feb 6 -- {'value': '2024-02-06', 'scope': 'manuscript'}
- PASS: numeric/text claim present: source date Mar 5 -- {'value': '2024-03-05', 'scope': 'manuscript'}
- PASS: metric claim matches evidence: temporal Kalman ADE -- {'value': '1,759.7'}
- PASS: metric claim matches evidence: vessel Kalman ADE -- {'value': '3,109.4'}
- PASS: metric claim matches evidence: temporal CV ADE -- {'value': '2,751.3'}
- PASS: metric claim matches evidence: vessel CV ADE -- {'value': '9,553.5'}
- PASS: metric claim matches evidence: vessel Ridge ADE -- {'value': '3,446.9'}
- PASS: metric claim matches evidence: temporal Transformer ADE -- {'value': '56,310.7'}
- PASS: risk claim matches evidence: Kalman risk precision -- {'value': '0.963', 'scope': 'both'}
- PASS: risk claim matches evidence: Kalman risk recall -- {'value': '0.900', 'scope': 'both'}
- PASS: risk claim matches evidence: Kalman false alarm -- {'value': '0.012', 'scope': 'manuscript'}
- PASS: risk claim matches evidence: Kalman missed warning -- {'value': '0.100', 'scope': 'manuscript'}
- PASS: risk claim matches evidence: risk scenario count -- {'value': '2,000', 'scope': 'both'}
- PASS: run manifest is non-debug -- {'is_debug_run': False}
- PASS: readiness report has no blocking gaps -- {'overall_status': 'submission_ready_candidate', 'blocking_gaps': []}
- PASS: all multiday model rows completed -- {'rows': 18, 'non_ok': []}
- PASS: reference list has 30 entries -- {'entries': 30}
- PASS: reference audit has 30 audited rows -- {'audit_rows': 30}
- PASS: DOIs/URLs added where available -- {'doi_count': 18}
- PASS: ScholarOne metadata contains core submission metadata -- {'required_bits': ['Li Bo', 'li.bo@cmaritime.com.cn', 'jon-submission-v1.3']}
- PASS: cover letter contains core submission metadata -- {'required_bits': ['Li Bo', 'li.bo@cmaritime.com.cn', 'jon-submission-v1.3']}
- PASS: Chinese submission guide contains core submission metadata -- {'required_bits': ['Li Bo', 'li.bo@cmaritime.com.cn', 'jon-submission-v1.3', 'ScholarOne']}
- PASS: submission checklist contains core submission metadata -- {'required_bits': ['Li Bo', 'li.bo@cmaritime.com.cn', 'jon-submission-v1.3']}
- PASS: main PDF readable text extraction -- {'pages': 17, 'sample_chars': 48687}
- PASS: main PDF page count within JON review target -- {'pages': 17}
- PASS: main manuscript/DOCX/PDF repository path cleanup -- {'blocked_patterns': ['outputs/', 'scripts/', '.py', 'overall_status', 'readiness_report', 'final_submission', 'final_multiday', 'final_risk', 'audit/'], 'result': 'no matches in main Markdown, DOCX text, or PDF text'}
- PASS: Chinese PDF readable text extraction -- {'pages': 9, 'sample_chars': 3187}
- PASS: Chinese interpretation PDF readable text extraction -- {'pages': 2, 'sample_chars': 1612}
- PASS: DOCX text readable: paper/jon_manuscript.docx -- {'paragraphs': 137, 'contains_author': True}
- PASS: DOCX text readable: paper/jon_manuscript_zh.docx -- {'paragraphs': 106, 'contains_author': True}
- PASS: DOCX text readable: paper/jon_manuscript_zh_interpretation.docx -- {'paragraphs': 44, 'contains_author': False}
- PASS: supplementary zip below 10 MB -- {'size_mb': 0.31125640869140625, 'entries': ['jon_supplementary_materials.md', 'multiday_data_manifest.json', 'model_metrics.csv', 'generalization_metrics.csv', 'error_summary_by_horizon.csv', 'error_summary_by_group.csv', 'statistical_tests.json', 'neural_tuning_protocol.json', 'neural_tuning_results.csv', 'risk_metrics.json', 'risk_scenarios.csv', 'readiness_report.json']}
- PASS: manifest lists scholarone_metadata_md -- {'value': 'paper/jon_scholarone_metadata.md'}
- PASS: manifest lists reference_audit_md -- {'value': 'paper/jon_reference_audit.md'}
- PASS: manifest lists submission_guide_zh_md -- {'value': 'paper/jon_submission_guide_zh.md'}
- PASS: PDF visual sample render reviewed -- {'rendered_pages': ['main page 1', 'main page 8', 'main page 12', 'main page 17'], 'tool': 'pdftoppm + image inspection', 'result': 'A4 portrait pages render visibly; no obvious clipping, overlap, or missing figures observed in reviewed pages'}
- PASS: DOCX visual renderer availability noted -- {'soffice': 'not found', 'libreoffice': 'not found', 'fallback': 'DOCX text extraction and regenerated PDF visual checks were completed; ScholarOne-generated PDF must still be reviewed after upload'}
- PASS: README aligned with active JON/high-quality route -- {'result': 'removed stale headline numbers and updated active evidence paths, manuscript page count, citation guidance, and claim boundary'}
