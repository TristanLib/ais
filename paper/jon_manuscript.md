# A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support

Article type: Research Article

Authors: [Author 1], [Author 2], [Author 3]

Affiliations: [Affiliation details to be completed before ScholarOne upload]

Corresponding author: [Name, email, ORCID]

Generated from repository artifacts: 2026-05-17T02:48:32.132551+00:00

## Abstract

Short-horizon ship trajectory prediction is increasingly presented as a machine-learning problem, yet navigation practice also requires transparent baselines, generalisation evidence and a defensible link between forecast error and operational warnings. This paper presents a reproducible benchmark for AIS trajectory prediction and CPA/TCPA risk-warning support. The evidence pack is built from NOAA historical AIS data covering four source dates (2024-01-02, 2024-01-09, 2024-02-06, 2024-03-05) and contains 186,326 trajectory windows from 7,425 MMSI values. Each sample uses 30 one-minute history points to forecast 15 one-minute future positions. The protocol records raw checksums, processed checksums, vessel identifiers, source dates, regions, speeds, turn-intensity metadata, temporal holdout labels and vessel-disjoint holdout labels. We compare kinematic, Kalman-style, linear and neural baselines using Haversine ADE/FDE and local-component RMSE/MAE, then evaluate how selected forecasts affect AIS-derived CPA/TCPA warning precision, recall, false alarms and missed warnings. The best ADE model in the current non-debug run is a Kalman-style constant-velocity baseline: 1,759.7 m ADE on the temporal holdout and 3,109.4 m ADE on the vessel-disjoint holdout. Constant velocity remains a strong reference on the temporal holdout (2,751.3 m ADE) but degrades more under vessel-disjoint testing (9,553.5 m ADE). Naive neural baselines do not outperform the strong motion baselines under this controlled protocol. In the risk-warning evaluation, the Kalman-style baseline reaches precision 0.963, recall 0.900, false-alarm rate 0.012 and missed-warning rate 0.100 across 2,000 AIS-derived encounter scenarios. The contribution is not an autonomous collision-avoidance system; it is an auditable navigation benchmark that shows why simple baselines and downstream warning metrics should accompany claims about AIS prediction performance.

## 1. Introduction

AIS trajectory prediction has become a familiar component in maritime traffic analysis, port monitoring, route inference and collision-risk assessment. The availability of large historical AIS archives has also encouraged increasingly complex forecasting models. The difficulty is that better model architecture alone does not automatically produce better navigation evidence. A forecast that appears accurate under one split can fail when evaluated on vessels not seen during training, and an average position error can obscure whether a risk-warning system produces more missed warnings or more false alarms. Navigation research therefore needs protocols that connect data provenance, baseline strength, forecast metrics and downstream warning behaviour.

The core problem addressed in this paper is methodological rather than purely architectural. Many AIS prediction studies compare proposed models with weak or inconsistently implemented baselines, use split definitions that are difficult to reproduce, or stop the evaluation at ADE and FDE. Those metrics are useful, but they are not the end of the navigation question. A shipboard or shore-based decision-support layer must also consider the closest point of approach, the time to closest approach, warning thresholds, false alarms and missed warnings. A trajectory model that reduces an average error by a small margin may not improve a CPA/TCPA warning, while a model with a modest mean ADE can still be valuable if its warning behaviour is stable and interpretable.

The paper is written for a navigation audience. It treats AIS prediction as a support layer for maritime situational awareness, not as a stand-alone deep-learning leaderboard. This framing leads to three research questions. RQ1 asks how simple kinematic, statistical and neural baselines compare under an audited short-horizon AIS protocol. RQ2 asks whether model rankings remain stable under temporal and vessel-disjoint holdouts. RQ3 asks how trajectory-prediction differences affect CPA/TCPA warning precision, recall, false alarms and missed warnings. These questions are deliberately conservative because they map to the evidence that can be defended from the current repository artifacts.

The navigation context also changes the burden of proof. In a generic time-series benchmark, it may be sufficient to show that one model has a lower average error than another. In a maritime setting, the user of a prediction layer may be a watch officer, a vessel-traffic-service operator, a shore-based monitoring analyst or an automated advisory module. These users do not only need a point forecast. They need to know when a forecast can be trusted, when it is likely to produce nuisance alarms, and when it might miss a close encounter. This is why the paper treats the data protocol, split design and risk-warning evaluation as part of the same contribution.

The first contribution is a reproducible, metadata-rich AIS benchmark pipeline. The pipeline records raw file checksums, processed-file checksum, row counts, MMSI counts, timestamp range, region labels, average speed, turn-intensity bins, interpolation ratio, and split labels. The processed artifact keeps both temporal and vessel-disjoint labels, so the same data build supports two complementary forms of generalisation testing. This is important because temporal holdouts test later windows from the same broad traffic distribution, while vessel-disjoint holdouts ask whether the model transfers to MMSI values not used for training.

The second contribution is a baseline-centred experimental result. In the current run, the Kalman-style constant-velocity baseline is the best mean-ADE model in both holdouts, and the ordinary constant-velocity baseline remains a strong reference. Ridge regression is close to Kalman-CV on the vessel-disjoint holdout (3,446.9 m ADE), but neural sequence baselines remain far behind the kinematic and linear baselines. This result should not be read as a universal claim that neural models are unsuitable for maritime prediction. It is a cautionary, reproducible result: architecture-superiority claims require stronger tuning, preprocessing, split discipline and external validation than a single reported run.

The third contribution is a downstream risk-warning evaluation that translates selected trajectories into CPA/TCPA warning classifications. The current risk artifact contains 2,000 AIS-derived encounter scenarios from 5,000 evaluated samples, with a warning threshold of 0.5 nautical miles and a search radius of 3.0 nautical miles. The labels are derived from observed future separation inside the forecast horizon, so the analysis remains a historical decision-support evaluation rather than a closed-loop collision-avoidance simulation.

This conservative framing is also intended to make the work useful even if a reader disagrees with the specific baseline set. The evidence chain is reusable: new models can be added, alternative warning thresholds can be tested, and later AIS source dates can be processed without rewriting the manuscript logic. The project therefore contributes a research workflow as well as a numerical result. The result reported here is the current state of that workflow, not a claim that the model list is final.

The remainder of the paper is organised as follows. Section 2 reviews related work on AIS prediction, simple baselines, neural sequence models, CPA/TCPA risk assessment and reproducible evaluation. Section 3 describes the data and the reproducible protocol. Section 4 defines the model families and metrics. Section 5 explains the risk-warning evaluation design. Section 6 reports trajectory and warning results. Section 7 discusses navigation implications. Sections 8 and 9 state limitations and conclusions.

## 2. Related Work

AIS data have been used for route discovery, anomaly detection, traffic characterisation and short-term vessel prediction. A recurring theme in this literature is that maritime trajectories are highly structured by geography, navigation rules, traffic lanes, port approaches and operational constraints, but they are also noisy because AIS reception, reporting intervals and vessel manoeuvres are irregular. Pallotta et al. (2013) framed AIS pattern discovery as a way to support anomaly detection and route prediction. Tu et al. (2018) reviewed AIS analytics across data handling and methodology, highlighting the breadth of tasks that depend on reliable AIS preprocessing. Ristic et al. (2008), Hexeberg et al. (2017), Dalsnes et al. (2018) and Millefiori et al. (2016) illustrate the range of probabilistic and statistical approaches used before recent deep-learning enthusiasm.

The same literature also shows why AIS prediction cannot be separated from data engineering. AIS messages can be sparse, duplicated, delayed or spatially inconsistent, and the same physical vessel can produce many overlapping windows. A benchmark that hides these choices behind a preprocessed array is difficult to interpret. For that reason, the present work records the sampling choices and preserves metadata needed for temporal and vessel-disjoint evaluation. This is not only a reproducibility convenience; it is a defence against overoptimistic generalisation claims.

For short horizons, simple motion assumptions remain difficult to dismiss. Constant velocity, constant acceleration and Kalman-style filtering are not merely straw men; they encode the fact that many vessels maintain course and speed for short intervals, especially when the forecast horizon is measured in minutes. Kalman (1960) remains foundational for recursive filtering, and navigation applications often benefit from the interpretability and computational economy of such filters. The present benchmark therefore treats kinematic baselines as first-class models. A proposed model that does not outperform them under well-defined splits should not be described as a navigation advance without a specific explanation of where its value lies.

This baseline stance follows a broader principle in safety-related prediction. A complex model should earn its complexity by providing a measurable advantage under a relevant operational criterion. That advantage might be lower FDE, better stability during manoeuvres, fewer missed warnings, calibrated uncertainty or robustness to vessel-disjoint testing. If the advantage is absent, the simpler model may still be preferable because it is faster, easier to audit and easier to explain. In this study, the strongest current result comes from such a simple model family.

Neural sequence models provide flexible function approximation and can learn nonlinear temporal patterns, but they are sensitive to representation, normalisation, training regime and data split. LSTM networks (Hochreiter and Schmidhuber, 1997), GRU networks (Cho et al., 2014), temporal convolutional networks (Bai et al., 2018) and Transformers (Vaswani et al., 2017) are all reasonable baseline families for sequence forecasting. Their inclusion in this benchmark does not imply that the particular shallow configurations used here exhaust the potential of neural AIS prediction. Instead, they serve as controlled baselines that expose a practical issue: neural models can underperform badly if the protocol, scaling and training budget are not aligned with the data and task.

Collision-risk analysis adds another layer. Closest point of approach and time to closest approach are standard quantities for encounter assessment, and ship-domain studies provide additional geometric context. Goodwin (1975), Fujii and Tanaka (1971), Hansen et al. (2013), Mou et al. (2010), Statheros et al. (2008), and Szlapczynski and Szlapczynska (2017) show that navigation risk is shaped by encounter geometry, traffic density, domain assumptions and decision rules. The present study does not validate a complete collision-avoidance controller. It uses CPA/TCPA warning metrics to ask whether differences in trajectory forecasts matter for a simple decision-support warning layer.

The distinction between warning support and avoidance control is essential. A warning system can be evaluated from historical AIS by asking whether the predicted encounter geometry would have raised an alert. A collision-avoidance controller requires a much richer experimental setting, including own-ship dynamics, target-ship response, COLREGs interpretation, human supervision and failure handling. The present paper stays on the former side of that boundary. This makes the experiment narrower, but also makes the claims more defensible.

Reproducibility is central because AIS studies can be surprisingly difficult to compare. Small changes in cleaning rules, interpolation, sample slicing, vessel filtering, split construction or distance metrics can alter results. Scientific-computing recommendations such as Wilson et al. (2017) and reproducibility discussions such as Leek and Peng (2015) and Pineau et al. (2021) motivate the artifact-first design used here. Statistical-comparison guidance from Demsar (2006) also supports paired evaluation and transparent reporting rather than a single aggregate number. The benchmark therefore stores generated evidence as CSV and JSON artifacts and regenerates manuscript tables from those artifacts.

The related-work gap addressed by this paper is therefore not the absence of AIS prediction models. It is the absence, in many model-centred papers, of a compact evidence chain that a navigation reviewer can audit from raw data manifest to risk-warning metric. By combining a conservative model comparison with downstream warning analysis, this work tries to make model evaluation legible to both machine-learning and navigation-safety readers.

## 3. Data and Reproducible Protocol

The data source is the NOAA MarineCadastre.gov AIS archive (MarineCadastre.gov, 2024; NOAA Office for Coastal Management, 2026). The current processed artifact covers four source dates: 2024-01-02, 2024-01-09, 2024-02-06, 2024-03-05. The data manifest records 186,326 trajectory windows, 7,425 unique MMSI values, and a time range from 2024-01-02T00:00:00 to 2024-03-05T12:59:00. The region labels represented in the current build are east_gulf_coast, hawaii_pacific, other, west_coast. The mean speed over ground recorded in the processed sample metadata is 2.22 knots, with a maximum of 34.88 knots. The mean turn-intensity metadata value is 17.69 degrees, and the mean interpolation ratio is 0.181.

Each example consists of a fixed-length historical sequence and a fixed-length future sequence. The protocol uses 30 input steps and 15 forecast steps, with one-minute spacing after resampling. The processed coordinates remain WGS84 latitude and longitude. ADE and FDE are computed as Haversine distances in metres. RMSE and MAE are computed on local north/east component errors, also in metres. This avoids confusing angular degrees with metric displacement, a common source of inflated or misleading claims in geospatial prediction.

The one-minute grid is a compromise between AIS reporting irregularity and navigational interpretability. It is short enough to support 15-minute risk-warning analysis and long enough to reduce the influence of individual message jitter. The window design also keeps the task deliberately local: the model is asked to forecast near-future motion, not to infer an entire voyage plan. This is why simple baselines are expected to be strong and why failure to beat them is informative rather than surprising.

The temporal split uses 130,428 training samples, 27,948 validation samples and 27,950 test samples. The vessel-disjoint split uses 130,018 training samples, 28,597 validation samples and 27,711 test samples. The temporal split evaluates future time blocks, while the vessel-disjoint split holds out MMSI values from training. Neither split should be interpreted as all-day seasonal validation or live AIS deployment. The split design is stronger than a single random split, but it is still a historical time-block protocol.

The vessel-disjoint split is particularly important because AIS windows from the same vessel are not independent in a behavioural sense. If a model sees many windows from a vessel during training, it may partly learn that vessel's typical operating area or motion regime. Holding out MMSI values reduces this leakage and creates a more demanding test. It is still not perfect, because vessels can share routes and regions, but it moves the benchmark closer to the way a deployed system would encounter previously unseen targets.

Figure 1 summarises the evidence chain from raw AIS to risk-warning metrics.

![Figure 1. Reproducible evidence chain used for the JON submission candidate.](figures/jon_pipeline_protocol.png)

The benchmark is designed so that the paper and project mutually support one another. The manuscript does not manually transcribe hidden spreadsheet calculations. The tables and figures are generated from `outputs/audit/multiday_data_manifest.json`, `outputs/final_multiday/model_metrics.csv`, `outputs/final_multiday/error_summary_by_horizon.csv`, `outputs/final_multiday/error_summary_by_group.csv`, `outputs/final_multiday/statistical_tests.json` and `outputs/final_risk/risk_metrics.json`. The current high-quality readiness report records `overall_status=submission_ready_candidate` and no blocking gaps at generation time.

A practical benefit of this design is that future revisions can be regenerated after new data or models are added. If a reviewer asks for another time block, a different risk threshold or an additional baseline, the project can update the evidence files and regenerate the manuscript tables. This reduces the chance that a paper, a figure and a code output drift apart during revision.

## 4. Trajectory Prediction Models and Metrics

The benchmark contains nine model entries. Constant velocity extrapolates future positions from the latest short-term motion. Constant acceleration extends that assumption by estimating acceleration from the recent sequence. The Kalman-style constant-velocity model applies a filtering/smoothing view of the same physical assumption. Ordinary least squares and ridge least squares map the flattened history window to future displacements. LSTM, GRU, Transformer and TCN entries provide neural sequence baselines implemented with PyTorch (Paszke et al., 2019). Linear tooling uses the scientific Python stack, including scikit-learn where appropriate (Pedregosa et al., 2011).

The model set is intentionally mixed. It is not a catalogue of every possible maritime predictor, and it is not designed to make a universal neural-model claim. It is designed to answer whether common baseline families survive a reproducible, metadata-rich, short-horizon AIS protocol. This is why the main comparison includes simple baselines even when they are not novel. In safety-related navigation research, a non-novel baseline can still be a novel contribution if it prevents unsupported claims about more complex systems.

The primary trajectory metrics are average displacement error, final displacement error, RMSE and MAE. ADE is the mean Haversine distance between predicted and observed positions over the forecast horizon. FDE is the Haversine distance at the final forecast step. RMSE and MAE provide component-level local displacement summaries. The model ranking in the text is based on mean ADE, while the tables also report median ADE and empirical 95 percent ADE intervals to show the skew caused by difficult or anomalous windows.

Mean ADE is useful for ranking, but it is not sufficient for navigation interpretation. A model can have a low median error and a high mean error if most windows are easy and a small number have very large errors. Such upper-tail cases matter because they may correspond to manoeuvres, low-speed drift, port approaches, sparse reporting or unusual regional geometry. The manuscript therefore reports mean, median and empirical intervals together. This is a deliberate guard against presenting a single attractive number as the full story.

Paired comparisons are computed against constant velocity because CV is the main operationally meaningful reference. The statistical artifact includes paired t-tests, Wilcoxon tests and Bonferroni-corrected comparison counts. The paper does not overstate these tests. Because AIS errors are highly skewed, the mean and the median tell different stories. In particular, Kalman-CV has very low median ADE in both holdouts but still has large upper-tail errors, which explains why mean ADE remains in kilometres even though the median error is measured in metres.

The modelling protocol also records neural tuning separately from final test claims. The neural proxy search provides reviewer transparency about the limited validation configurations explored, but final conclusions remain tied to the full-split test metrics. This separation matters because using test performance to tune neural models would undermine the benchmark. The present neural results are therefore best read as baseline outcomes under documented settings, not as a final word on neural maritime forecasting.

## 5. Generalisation and Risk-Warning Evaluation

Generalisation is evaluated using both temporal and vessel-disjoint holdouts. Temporal holdout is useful because a deployed predictor normally faces later windows than those used during training. Vessel-disjoint holdout is useful because it asks whether a model depends on learning idiosyncratic behaviour from vessels seen during training. The two tests are complementary; a model that looks acceptable under temporal evaluation may not transfer to unseen MMSI values.

For downstream risk-warning evaluation, the benchmark selects AIS-derived encounter scenarios from the temporal test setting. Observed future trajectories define the truth warning label: a warning is positive if the observed minimum CPA over the forecast horizon crosses the configured threshold. Predicted trajectories are then used to compute predicted CPA/TCPA warnings. This yields true positives, false positives, false negatives and true negatives. The warning metrics are precision, recall, false-alarm rate, missed-warning rate, mean absolute lead-time error and mean absolute CPA error.

This design allows the same trajectory outputs to be judged in a way that resembles operational decision support. Precision reflects the credibility of raised warnings, while recall reflects the ability to capture true close-approach cases. False-alarm rate is important because excessive alarms can reduce user trust, and missed-warning rate is important because missed close approaches are safety critical. CPA error adds a continuous measure that is easier to interpret in nautical miles.

The risk-warning evaluation uses 2,000 scenarios from 5,000 evaluated samples. The search radius is 3.0 nautical miles, the warning threshold is 0.5 nautical miles and the truth warning count is 520. These choices produce an interpretable operational slice, but they are not a certification criterion. They should be read as a reproducible decision-support experiment.

The warning threshold is intentionally fixed rather than optimised per model. Optimising a different threshold for every model could improve individual scores, but it would make the comparison less transparent. A fixed threshold gives reviewers a stable basis for interpreting the confusion matrices. Alternative thresholds are a reasonable future sensitivity study and can be added without changing the rest of the evidence pipeline.

## 6. Results

Figure 2 reports the main trajectory-performance comparison. Kalman-CV is the best mean-ADE model in both split protocols. It achieves 1,759.7 m ADE and 2,704.5 m FDE on the temporal holdout, and 3,109.4 m ADE and 5,979.6 m FDE on the vessel-disjoint holdout. Constant velocity reaches 2,751.3 m temporal ADE and 9,553.5 m vessel-disjoint ADE. The gap between CV and Kalman-CV is larger under vessel-disjoint evaluation, suggesting that filtered motion estimates provide robustness when specific vessel identities are not seen during training.

![Figure 2. ADE comparison across temporal and vessel-disjoint holdouts. The log scale is used because neural and acceleration baselines have much larger errors than the strongest kinematic models.](figures/jon_model_performance.png)

Table 1 gives the numerical trajectory results. The distinction between mean and median is important. For example, Kalman-CV has median ADE 8.2 m on the temporal holdout and 9.2 m on the vessel-disjoint holdout, while its mean ADE remains much larger. This pattern is consistent with many short windows being easy and a smaller number of windows being difficult due to manoeuvres, sparse/interpolated reporting, regional geometry or other traffic effects.

Table 1. Trajectory-prediction metrics generated from the current non-debug evidence pack.

| Split | Model | Mean ADE (m) | Median ADE (m) | 95% ADE interval (m) | FDE (m) |
|---|---|---:|---:|---:|---:|
| Temporal holdout | Kalman-CV | 1,759.7 | 8.2 | 0.6 to 5,673.6 | 2,704.5 |
| Temporal holdout | CV | 2,751.3 | 20.5 | 0.7 to 5,357.1 | 4,469.8 |
| Temporal holdout | Ridge | 3,141.7 | 1,563.1 | 451.4 to 11,189.3 | 5,079.5 |
| Temporal holdout | OLS | 3,052.3 | 1,439.3 | 426.1 to 11,320.0 | 4,908.7 |
| Temporal holdout | GRU | 25,215.5 | 19,597.5 | 3,979.9 to 92,526.6 | 25,571.8 |
| Temporal holdout | LSTM | 36,039.2 | 25,660.5 | 4,365.6 to 116,019.1 | 36,116.8 |
| Temporal holdout | TCN | 47,095.5 | 32,265.2 | 9,361.0 to 149,135.1 | 47,078.0 |
| Temporal holdout | Transformer | 56,310.7 | 35,643.0 | 11,299.8 to 210,486.8 | 55,923.5 |
| Temporal holdout | CA | 35,076.6 | 166.6 | 0.9 to 64,305.6 | 76,360.6 |
| Vessel-disjoint holdout | Kalman-CV | 3,109.4 | 9.2 | 0.6 to 11,946.7 | 5,979.6 |
| Vessel-disjoint holdout | CV | 9,553.5 | 22.8 | 0.8 to 12,053.7 | 17,014.2 |
| Vessel-disjoint holdout | Ridge | 3,446.9 | 1,277.5 | 319.1 to 14,385.8 | 6,463.4 |
| Vessel-disjoint holdout | OLS | 8,113.3 | 930.0 | 261.2 to 12,478.9 | 14,869.8 |
| Vessel-disjoint holdout | GRU | 23,989.0 | 19,508.3 | 5,810.3 to 67,283.9 | 24,569.4 |
| Vessel-disjoint holdout | LSTM | 51,010.6 | 35,421.3 | 8,462.8 to 151,738.0 | 50,890.1 |
| Vessel-disjoint holdout | TCN | 32,833.1 | 24,902.1 | 4,013.2 to 95,171.4 | 34,379.1 |
| Vessel-disjoint holdout | Transformer | 47,559.1 | 33,691.0 | 10,991.1 to 138,971.6 | 45,907.0 |
| Vessel-disjoint holdout | CA | 36,237.1 | 175.1 | 1.0 to 37,008.8 | 67,988.2 |

The neural baselines should be interpreted carefully. The Transformer baseline records 56,310.7 m temporal ADE in the current run, and the other neural baselines are also far above the strongest kinematic and linear baselines. This is not evidence that neural AIS prediction is impossible. It is evidence that naive neural baselines can fail under this exact preprocessing and split protocol. The useful publication claim is therefore methodological: strong baselines, documented tuning, split discipline and downstream warning evaluation are necessary before asserting architecture superiority.

The constant-acceleration result is also instructive. Although acceleration may appear to be a richer physical assumption than constant velocity, it performs very poorly in the current aggregate metrics. This likely reflects the sensitivity of acceleration estimates to noisy or interpolated short-window position changes. For navigation applications, a physically plausible model family still needs to be numerically stable under the reporting characteristics of AIS. More parameters are not automatically better when the observations are irregular and manoeuvres are sparse.

Figure 3 shows error growth across the 15 forecast steps. The short-horizon character of the task is visible: errors generally increase with horizon, and the more stable baselines retain better behaviour over time. This figure is useful for navigation readers because a 15-minute average can hide whether an error appears immediately or accumulates near the end of the horizon.

![Figure 3. Horizon-wise ADE degradation for selected model families.](figures/jon_horizon_degradation.png)

Figure 4 reports the CPA/TCPA warning metrics. Kalman-CV and constant velocity are close in precision and recall. Kalman-CV has precision 0.963 and recall 0.900; CV has precision 0.961 and recall 0.894. Kalman-CV has a slightly lower false-alarm rate and missed-warning rate, while CV has a slightly lower mean absolute CPA error in nautical miles. The linear least-squares baseline produces a much higher false-alarm rate and larger CPA error, showing that position-prediction quality does not translate uniformly into warning quality.

![Figure 4. CPA/TCPA warning classification and CPA-error metrics for selected trajectory models.](figures/jon_risk_warning_metrics.png)

Table 2. AIS-derived risk-warning metrics.

| Model | TP | FP | FN | TN | Precision | Recall | False alarm | Missed warning | CPA error (nmi) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kalman-CV | 468 | 18 | 52 | 1462 | 0.963 | 0.900 | 0.012 | 0.100 | 0.092 |
| CV | 465 | 19 | 55 | 1461 | 0.961 | 0.894 | 0.013 | 0.106 | 0.082 |
| OLS | 453 | 173 | 67 | 1307 | 0.724 | 0.871 | 0.117 | 0.129 | 0.346 |

Figure 5 reports scenario-slice sensitivity for speed and region groups on the temporal holdout. The purpose is not to claim complete regional robustness. Instead, the figure makes visible that aggregate results are shaped by traffic composition. Low-speed and high-speed slices can have very different error magnitudes, and regions with fewer samples should be interpreted with greater caution. This supports the paper's conservative boundary: the current evidence is stronger than a single aggregate benchmark but is not yet a full seasonal or global generalisation study.

The slice analysis also provides a practical debugging tool. If a future model improves the aggregate ADE but worsens high-speed or regional slices, a navigation reviewer may reasonably question whether the model is safer or merely better aligned with the dominant traffic class. Conversely, a model that is slightly worse on the aggregate but much more stable in a safety-critical slice could be worth further study. This is another reason to keep group summaries in the supplementary evidence package.

![Figure 5. Scenario-slice sensitivity by speed bin and region on the temporal holdout.](figures/jon_scenario_slice_errors.png)

Figure 6 provides AIS-derived encounter examples from the existing risk-warning output. Such figures are essential because navigation readers need to inspect whether the warning problem looks operationally plausible, not merely whether a metric table is favourable.

![Figure 6. AIS-derived encounter case studies from the risk-warning evaluation.](figures/jon_risk_case_studies.png)

## 7. Discussion

The main result is conservative but practically useful. A simple, transparent motion model remains difficult to beat for short-horizon AIS forecasting, and a Kalman-style version of that model is the strongest current baseline by ADE under both temporal and vessel-disjoint holdouts. This finding has immediate implications for navigation research. A new AIS prediction model should be compared against well-implemented kinematic baselines, not only against other neural models or weakened classical references. If it fails to improve on those baselines, the paper can still contribute if it explains where the failure occurs and what it reveals about the evaluation protocol.

For a journal such as The Journal of Navigation, the value of this result lies in its operational humility. The paper does not invite readers to accept a black-box architecture because it is fashionable. It asks them to inspect a chain of evidence and to judge whether the claimed support is proportional to the experiment. That posture is well suited to navigation research, where a premature claim can be less useful than a carefully bounded benchmark that others can extend.

The risk-warning results also temper the interpretation of ADE. Kalman-CV has the best trajectory ADE, but CV has a slightly lower mean absolute CPA error in the current risk slice. The difference is small, and both models produce strong warning precision and recall, but the point is important. Downstream navigation metrics can reorder or nuance model preferences. An operational warning layer may care about false alarms, missed warnings and lead-time stability as much as mean position error. Future AIS prediction papers should therefore include at least one downstream navigation-safety metric when making decision-support claims.

This result also suggests a direction for future model design. Instead of training only to minimise pointwise displacement error, a navigation-specific predictor might include losses or calibration objectives related to encounter geometry, CPA uncertainty or warning-threshold stability. Such a model would need careful validation, but it would align the optimisation target more closely with the decision-support task. The current project provides a baseline against which such extensions can be tested.

The vessel-disjoint evaluation is one of the more important additions relative to a simpler evidence report. Vessel identity can leak behaviour into random splits, especially when multiple windows from the same MMSI appear in train and test sets. Holding out MMSI values does not solve every generalisation issue, but it reduces one obvious source of overoptimism. The current results show that CV degrades substantially more than Kalman-CV under vessel-disjoint testing, while ridge regression becomes competitive with Kalman-CV. This suggests that model robustness depends on both the physical assumption and the split boundary.

The results should not be oversold. The current neural baselines are deliberately modest and documented as baselines. The tuning protocol records a validation-set proxy search, but it does not represent a comprehensive neural architecture search. A future neural study could improve scaling, loss design, coordinate parameterisation, sequence length, map context, vessel type metadata or encounter-aware objectives. Such improvements may well change the neural results. The contribution here is that those future claims should be made against the current kind of evidence chain: auditable data, strong baselines, split definitions, statistical summaries and downstream warning analysis.

For practitioners, the project has a direct use. Running the pipeline on updated historical AIS source dates can regenerate the same tables, figures and warning outputs. The `predict_latest_ais.py` script in the project exports offline trajectory predictions and risk warnings from the latest available prepared data. This is not a live operational service, but it demonstrates how the benchmark can be extended into a monitoring workflow. With new AIS data, the same code can produce current-period predictions and risk-warning candidates, subject to the same limitations about historical data quality, warning thresholds and absence of closed-loop validation.

The practical route from this benchmark to an operational tool would require several additional layers. First, the data ingestion layer would need live AIS handling, latency monitoring and missing-message logic. Second, the prediction layer would need calibrated uncertainty rather than only point trajectories. Third, the warning layer would need human-factor evaluation so that alert frequency, phrasing and timing support rather than distract operators. Fourth, any avoidance recommendation would need explicit COLREGs interpretation and closed-loop testing. The present work does not complete those steps, but it gives them a reproducible starting point.

## 8. Limitations

The current evidence is historical. It does not ingest a live AIS stream and does not validate real-time deployment. The four source dates provide a stronger protocol than a single-day experiment, but they do not prove all-day, seasonal or global generalisation. The processed artifact includes useful metadata, yet it does not include every contextual variable that could matter, such as vessel class reliability, weather, traffic-control context, chart constraints or planned routes.

The risk-warning task is an AIS-derived decision-support evaluation. It is not a simulator of bridge-team behaviour, not a COLREGs compliance proof and not an autonomous collision-avoidance validation. The International Regulations for Preventing Collisions at Sea remain a legal and operational framework far richer than the thresholded CPA/TCPA labels used here (International Maritime Organization, 1972). Any operational use would require human-factors assessment, false-alarm tolerance analysis, traffic-service integration, reliability engineering and domain-specific validation.

The current manuscript also contains publication-preparation placeholders. Author names, affiliations, funding, ORCID identifiers, acknowledgements and suggested reviewers must be completed by the authors before ScholarOne upload. The reference list is formatted in Harvard/JON style as a candidate list, but it should receive a final bibliographic audit before submission.

## 9. Conclusions

This paper presents a reproducible AIS trajectory-prediction benchmark for navigation risk-warning support. The current evidence chain links raw historical AIS data, checksummed preprocessing, temporal and vessel-disjoint splits, kinematic/statistical/neural baselines, paired statistical summaries, scenario-slice analysis and CPA/TCPA warning metrics. The best current ADE model is Kalman-CV on both temporal and vessel-disjoint holdouts. Constant velocity remains a strong short-horizon baseline, and naive neural baselines fail to outperform the simple baselines under this controlled protocol. The practical lesson is not that deep learning has no role in maritime prediction, but that navigation papers need strong baselines, reproducible splits and downstream warning metrics before making decision-support claims.

## Data and Code Availability

The manuscript was generated from repository artifacts in the local project `ship-prediction-avoidance`. The source data are derived from NOAA MarineCadastre.gov AIS files subject to NOAA data access terms. The reproducibility command for the high-quality evidence package is:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_high_quality_pipeline.sh
```

The final public repository URL or archival DOI should be inserted before submission if the authors decide to release the code and generated artifacts publicly. The supplementary package accompanying this manuscript includes manifest summaries, full model metrics, horizon and group summaries, statistical summaries, neural tuning records and risk-warning metrics, but excludes the large per-sample error file to satisfy the current supplementary-file size constraint.

## Competing Interests

The authors declare no competing interests. This statement should be reviewed and replaced if any author has a relevant financial, professional or personal relationship to disclose.

## Author Contributions

[Author 1] conceived the study and led manuscript preparation. [Author 2] implemented the AIS preprocessing and benchmark pipeline. [Author 3] reviewed navigation-risk framing and interpretation. These placeholders must be replaced with the actual contributor roles before submission.

## Funding

[Funding information to be completed before submission. If no external funding supported the work, state: This research received no specific grant from any funding agency, commercial or not-for-profit sectors.]

## AI-Use Declaration

OpenAI Codex/ChatGPT was used in May 2026 to assist with code generation, manuscript drafting, document structuring and consistency checks against repository artifacts. The authors are responsible for all content, verified the numerical claims against generated evidence files, and will complete a final reference and language audit before submission.

## Acknowledgements

[Acknowledgements to individuals or organisations, if any, should be inserted here. The target journal should not be thanked in this section.]

## Supplementary Material

Supplementary File S1 contains the reproducibility manifest, full benchmark tables, scenario summaries and risk-warning evidence used to generate this manuscript. The individual supplementary archive produced by the project is kept below the current 10 MB per-file guideline.

## References

Bai, S., Kolter, J. Z. and Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.

Bergstra, J. and Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 1724-1734.

Dalsnes, B. R., Hexeberg, S., Flaten, A. L., Eriksen, B. O. H. and Brekke, E. F. (2018). The Neighbour Course Distribution Method with Gaussian Mixture Models for AIS-Based Vessel Trajectory Prediction. Proceedings of the 21st International Conference on Information Fusion.

Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.

Endsley, M. R. (1995). Toward a Theory of Situation Awareness in Dynamic Systems. Human Factors, 37, 32-64.

Fujii, Y. and Tanaka, K. (1971). Traffic Capacity. The Journal of Navigation, 24, 543-552.

Goodwin, E. M. (1975). A Statistical Study of Ship Domains. The Journal of Navigation, 28, 328-344.

Hansen, M. G., Jensen, T. K., Lehn-Schioler, T., Melchild, K., Rasmussen, F. M. and Ennemark, F. (2013). Empirical Ship Domain based on AIS Data. The Journal of Navigation, 66, 931-940.

Hexeberg, S., Flaten, A. L., Eriksen, B. O. H. and Brekke, E. F. (2017). AIS-Based Vessel Trajectory Prediction. Proceedings of the 20th International Conference on Information Fusion.

Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9, 1735-1780.

International Maritime Organization. (1972). Convention on the International Regulations for Preventing Collisions at Sea, 1972 (COLREGs). https://www.imo.org/en/About/Conventions/Pages/COLREG.aspx. Accessed 16 May 2026.

Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering, 82, 35-45.

Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. Proceedings of the International Conference on Learning Representations.

Leek, J. T. and Peng, R. D. (2015). Reproducible Research Can Still Be Wrong: Adopting a Prevention Approach. Proceedings of the National Academy of Sciences, 112, 1645-1646.

MarineCadastre.gov. (2024). AIS Data. https://marinecadastre.gov/ais/. Accessed 16 May 2026.

Millefiori, L. M., Braca, P., Bryan, K. and Willett, P. (2016). Modeling Vessel Kinematics using a Stochastic Mean-Reverting Process for Long-Term Prediction. IEEE Transactions on Aerospace and Electronic Systems, 52, 2313-2330.

Mou, J. M., van der Tak, C. and Ligteringen, H. (2010). Study on Collision Avoidance in Busy Waterways by using AIS Data. Ocean Engineering, 37, 483-490.

NOAA Office for Coastal Management. (2026). Marine Cadastre. https://www.coast.noaa.gov/digitalcoast/data/marine-cadastre.html. Accessed 16 May 2026.

Pallotta, G., Vespe, M. and Bryan, K. (2013). Vessel Pattern Knowledge Discovery from AIS Data: A Framework for Anomaly Detection and Route Prediction. Entropy, 15, 2218-2245.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and others. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and others. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

Pineau, J., Vincent-Lamarre, P., Sinha, K., Lariviere, V., Beygelzimer, A., d'Alche-Buc, F., Fox, E. and Larochelle, H. (2021). Improving Reproducibility in Machine Learning Research. Journal of Machine Learning Research, 22, 1-20.

Ristic, B., La Scala, B., Morelande, M. and Gordon, N. (2008). Statistical Analysis of Motion Patterns in AIS Data: Anomaly Detection and Motion Prediction. Proceedings of the 11th International Conference on Information Fusion.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F. and Dennison, D. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems, 28.

Statheros, T., Howells, G. and Maier, K. M. (2008). Autonomous Ship Collision Avoidance Navigation Concepts, Technologies and Techniques. The Journal of Navigation, 61, 129-142.

Szlapczynski, R. and Szlapczynska, J. (2017). Review of Ship Safety Domains: Models and Applications. Ocean Engineering, 145, 277-289.

Tu, E., Zhang, G., Rachmawati, L., Rajabally, E. and Huang, G. B. (2018). Exploiting AIS Data for Intelligent Maritime Navigation: A Comprehensive Survey from Data to Methodology. IEEE Transactions on Intelligent Transportation Systems, 19, 1559-1582.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. and Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.

Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L. and Teal, T. K. (2017). Good Enough Practices in Scientific Computing. PLOS Computational Biology, 13, e1005510.
