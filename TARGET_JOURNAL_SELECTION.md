# Target Journal Selection

Updated: 2026-05-16

## Paper Positioning

Active target:

> **The Journal of Navigation**. Use
> `JOURNAL_OF_NAVIGATION_SUBMISSION_ROADMAP.md` as the current execution
> roadmap.

Current manuscript thesis:

> On an audited AIS trajectory-prediction protocol, simple kinematic baselines
> remain strong for short-horizon vessel trajectory prediction; the project
> contributes a reproducible evidence chain and a downstream CPA/TCPA
> risk-warning evaluation, rather than claiming neural architecture superiority
> or autonomous collision avoidance.

Best Chinese title direction:

> 面向风险预警的AIS船舶轨迹预测可复现实证研究

The manuscript should be submitted as a maritime navigation safety, traffic
information, and decision-support paper. It should not be framed as a pure deep
learning algorithm paper.

## Recommended Target Tiers

### Tier 1: Best-Match Academic Targets

| Priority | Journal | Fit | Why it matches | Required revision |
|---:|---|---|---|---|
| 1 | 中国航海 | Very high | Core maritime navigation venue; official site has submission guide and recent issues include ship motion prediction, route/path planning, rescue/safety, and intelligent navigation topics. | Emphasize navigation safety, AIS evidence protocol, CPA/TCPA risk-warning value. Add stronger domestic navigation/AIS related work and journal-style figures. |
| 2 | 大连海事大学学报 | High | Maritime university journal centered on waterway transport and engineering research; recent hot/current papers include electronic chart, maritime logistics, ship systems, and navigation-related engineering topics. | Strengthen engineering reproducibility, data protocol, and method validation. Keep claims conservative and technical. |
| 3 | 上海海事大学学报 | High | Has prior AIS trajectory clustering and AIS/LSTM trajectory-prediction related articles; strong topical fit for maritime data analysis and transport engineering. | Reframe around AIS trajectory data mining, benchmark evidence, and risk-warning support. Use a polished Chinese paper structure. |

### Tier 2: Broader Traffic-Information/Safety Target

| Priority | Journal | Fit | Why it matches | Required revision |
|---:|---|---|---|---|
| 4 | 交通信息与安全 | Medium-high | Official journal scope focuses on traffic information, transportation safety, intelligent transportation, and information technology for safety; official site identifies it as CSCD/CNKI indexed. | Make the maritime specificity legible to a broader traffic-safety audience. Put “信息技术支撑交通安全预警” in the foreground and expand safety-evaluation discussion. |

### Tier 3: Practical/Application Fallbacks

| Priority | Journal | Fit | Why it matches | Required revision |
|---:|---|---|---|---|
| 5 | 航海技术 | Medium | China Institute of Navigation calls for practical navigation technology articles; scope includes navigation support, communication/navigation, maritime management, ship operation, and safety; article length expectation is shorter. | Compress to 3500-5000 Chinese characters, reduce benchmark breadth, focus on deployable AIS prediction and warning workflow. |
| 6 | 中国海事 | Medium-low | Maritime administration and safety-practice orientation; good only if rewritten as a maritime supervision/decision-support application note. | Remove most model-comparison detail; emphasize maritime supervision, VTS/risk-warning use, and practical workflow. |

## English Journal Targets

### Tier E1: Strong International Journal Targets

| Priority | Journal | Fit | Why it matches | Required upgrade |
|---:|---|---|---|---|
| 1 | The Journal of Navigation | Very high | Cambridge/RIN journal covering navigation over sea, land, air, and space. The current paper's AIS trajectory prediction, CPA/TCPA warning, and navigation-safety framing fit naturally. | Expand navigation-science discussion, collision-risk interpretation, and limitations; polish English substantially. |
| 2 | Ocean Engineering | High | Elsevier journal includes maritime safety, risk assessment, situational awareness, and broader ocean/maritime engineering applications. | Add more engineering relevance: operational risk-warning value, scenario case studies, and stronger maritime safety discussion. |
| 3 | IEEE Transactions on Intelligent Transportation Systems | High but difficult | IEEE T-ITS explicitly includes maritime transportation and ports/terminals under intelligent transportation systems. | Needs a stronger ITS framing, more extensive validation, public/reproducible artifacts, and likely stronger uncertainty/risk-warning analysis. |
| 4 | Transportation Research Part C: Emerging Technologies | High but difficult | TR-C values emerging transportation technologies, AI/ML, safety, reliability, open science, and large-scale datasets. | Reframe from "maritime paper" to "transportation systems benchmark"; add stronger transferability, open-data, and system-level implications. |

### Tier E2: Practical/Accessible International Journal Targets

| Priority | Journal | Fit | Why it matches | Required upgrade |
|---:|---|---|---|---|
| 5 | Applied Ocean Research | Medium-high | Scope includes hazards, safety, reliability, risk management, maritime engineering, and practical applicability. | Strengthen practical case studies and reduce pure benchmark tone. Note that it is full open access. |
| 6 | IEEE Open Journal of Intelligent Transportation Systems | Medium-high | Open-access IEEE ITS journal covering theoretical, experimental, and operational ITS, including AI/big-data applications. | Suitable if open-access cost is acceptable; needs stronger English polish and ITS positioning. |
| 7 | Journal of Marine Science and Engineering | Medium | Open-access marine journal with maritime/ship trajectory and route-planning topics; likely accessible for this theme. | Good fallback, but consider publisher/venue expectations and APC. Keep claims conservative. |
| 8 | Expert Systems with Applications | Medium-low | Broad applied intelligent systems journal with risk-assessment and engineering applications. | Current result does not showcase a winning expert/AI system; would need a clearer intelligent decision-support system contribution. |

## English Conference / Top-Conference Targets

### Realistic Strong Conference Targets

| Priority | Venue | Fit | Why it matches | Recommended format |
|---:|---|---|---|---|
| 1 | IEEE ITSC | High | IEEE ITSC is the flagship IEEE ITS conference. It is a good fit if framed as intelligent maritime transportation, risk warning, and reproducible AIS benchmarking. | 6-8 page IEEE paper; target main conference or an intelligent/automated waterway transportation workshop if available. |
| 2 | MTEC/ICMASS | High | Maritime autonomous ships conference connects academia and industry around autonomous shipping, port operations, and maritime technology. | Strong domain fit; emphasize autonomous-navigation validation and risk-warning support. |
| 3 | OCEANS / IEEE OES-MTS | Medium-high | Strong marine technology conference; suitable for maritime data, ocean technology, and operational decision support. | Emphasize maritime technology implementation and field-data workflow, not ML novelty. |
| 4 | IEEE IV workshops | Medium | IEEE IV has high visibility for trajectory prediction and safety, but is road-vehicle centered. Workshops on behavior prediction, safety validation, or open science are more realistic than main track. | Use only if reframed as transferable trajectory-prediction benchmark methodology. |

### Not Recommended Without Major Upgrade

| Venue type | Current fit | Why |
|---|---|---|
| NeurIPS / ICML / ICLR main track | Low | Current contribution is not a new ML model, optimization method, theory, or broadly reusable ML benchmark at top-AI scale. |
| KDD / WWW / SIGSPATIAL main track | Low-medium | Could become possible only if the work is upgraded into a public AIS benchmark dataset/data-mining task with stronger novelty, broad baselines, and reusable evaluation server/code. |
| Robotics top conferences such as ICRA/IROS main track | Low-medium | Needs closed-loop autonomous navigation, robotics experiments, or simulation/real-world autonomy validation beyond offline AIS prediction. |

## Recommended English Submission Order

1. **The Journal of Navigation**
   - Best English journal fit for the current thesis.
   - Recommended if the goal is a serious international maritime/navigation
     paper without pretending to be a pure AI breakthrough.
   - Detailed route: `JOURNAL_OF_NAVIGATION_SUBMISSION_ROADMAP.md`.

2. **Ocean Engineering**
   - Good high-quality engineering target if risk-warning and maritime safety
     discussion are strengthened.
   - More demanding on engineering contribution and practical importance.

3. **IEEE ITSC**
   - Best conference route.
   - Recommended if a shorter 6-8 page IEEE-style version is prepared.
   - A workshop on intelligent/automated waterway transportation would be an
     excellent stepping stone if the main conference feels too broad.

4. **IEEE T-ITS / Transportation Research Part C**
   - Aspirational journal targets.
   - Submit only after stronger external validity, broader protocol sensitivity,
     better uncertainty/risk-warning analysis, and a polished English narrative.

5. **Applied Ocean Research / IEEE Open Journal of ITS / JMSE**
   - More practical fallback options depending on open-access budget and desired
     review speed.

## Upgrade Plan for Top English Venues

To make the manuscript credible for T-ITS, TR-C, or a strong IEEE conference,
add the following before submission:

1. Full-day or more diverse time-block sensitivity experiments.
2. Public reproducibility package: code, configs, processed metadata manifest,
   and exact commands.
3. Uncertainty-aware risk-warning metrics, not only point-trajectory CPA/TCPA.
4. More current related work on maritime trajectory prediction, open AIS
   benchmarks, transportation trajectory prediction, and safety-critical
   prediction.
5. Stronger figures: error-vs-horizon curves, scenario-slice heatmaps,
   risk-warning precision/recall plots, and at least one encounter case study.
6. Clear limitations: historical AIS, no live stream, no autonomous avoidance,
   no neural superiority claim.

## Recommended Submission Order

1. **First choice: 中国航海**
   - Best thematic fit: navigation safety, intelligent navigation, AIS, route
     analysis, ship motion prediction.
   - Strongest path if the manuscript is polished as a maritime navigation
     safety/reproducibility paper.

2. **Second choice: 大连海事大学学报**
   - Strong maritime engineering fit and likely more receptive to a detailed
     evidence pipeline if written rigorously.
   - Good fallback if 中国航海 asks for more navigation-theory contribution.

3. **Third choice: 上海海事大学学报**
   - Good fit for AIS trajectory data analysis; prior related AIS trajectory
     work makes the topic natural.
   - Especially suitable if the manuscript highlights maritime traffic data
     mining and risk-warning support.

4. **Fourth choice: 交通信息与安全**
   - Use only after strengthening the broader traffic safety framing.
   - Better if the paper title/abstract includes “交通信息安全保障” or
     “交通安全预警” style language.

5. **Fallback/application: 航海技术**
   - Good if the goal is a practical Chinese publication faster than a core
     academic submission.
   - Needs a shorter, application-oriented version rather than the current full
     benchmark paper.

## Not Recommended as First Targets

- Pure AI/computer journals: the current contribution is not a novel neural
  architecture and neural models are not the winning result.
- Pure ship-design/ship-mechanics journals: the paper is AIS traffic/navigation
  data analysis, not hull, propulsion, or structural engineering.
- High-impact safety journals without stronger accident/incident validation:
  the current risk evidence is warning/decision-support, not validated accident
  causality or closed-loop avoidance.

## Revision Checklist Before Submitting to the Top 3

1. Expand Chinese related work:
   - AIS trajectory prediction and reconstruction.
   - Ship traffic flow and navigation safety.
   - CPA/TCPA collision-risk warning.
   - Reproducible benchmarking and simple-baseline evaluation.
2. Add journal-ready figures:
   - Model ADE/FDE comparison for temporal and vessel-disjoint splits.
   - Horizon-wise degradation curve.
   - Risk-warning precision/recall/false-alarm comparison.
   - One AIS-derived encounter case study.
3. Convert references to the target journal format, likely GB/T 7714 unless
   specified otherwise.
4. Replace placeholders in `paper/submission_manuscript_zh.md`:
   - authors, affiliations, funding, acknowledgements, corresponding author,
     conflict statement.
5. Keep limitations explicit:
   - historical AIS, not live AIS stream;
   - stratified time-block protocol, not full all-day seasonal generalization;
   - risk-warning decision support, not autonomous collision avoidance;
   - no neural architecture-superiority claim.

## Sources Checked

- 中国航海 official site: https://zghh.cinnet.cn/
- 中国航海 official journal pages, including submission-guide navigation and
  recent issue topics: https://zghh.cinnet.cn/
- 大连海事大学学术期刊中心: https://journal.dlmu.edu.cn/
- 上海海事大学学报 official site: https://www.smujournal.cn/
- 上海海事大学学报 AIS trajectory-clustering example:
  https://www.smujournal.cn/article/doi/10.13340/j.jsmu.2022.04.005
- 交通信息与安全 official site: https://www.jtxa.net/
- 交通信息与安全 fee/contact page: https://www.jtxa.net/news/bmfsfbz.htm
- 航海技术征稿启事, 中国航海学会:
  https://www.cinnet.cn/zh-hans/notes/7249-hang-hai-ji-zhu-zheng-gao-qi-shi.htm
- The Journal of Navigation, Cambridge: https://www.cambridge.org/core/journals/journal-of-navigation/information/about-this-journal
- Ocean Engineering, Elsevier: https://www.sciencedirect.com/journal/ocean-engineering
- IEEE Transactions on Intelligent Transportation Systems, IEEE ITSS:
  https://ieee-itss.org/pub/t-its/
- Transportation Research Part C, Elsevier:
  https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies
- Applied Ocean Research, Elsevier:
  https://www.sciencedirect.com/journal/applied-ocean-research
- IEEE Open Journal of Intelligent Transportation Systems, IEEE ITSS:
  https://ieee-itss.org/pub/oj-its/
- Journal of Marine Science and Engineering:
  https://www.mdpi.com/2077-1312
- IEEE ITSC, IEEE ITSS: https://ieee-itss.org/conf/itsc/
- IEEE ITSC 2025 Intelligent and Automated Waterway Transportation workshop:
  https://iawtworkshopitsc.github.io/2025/
- SMRC x MTEC/ICMASS Conference 2026: https://smrc.sg/
- ICMASS overview: https://autonomous-ship.org/events/icmass/general.html
- OCEANS 2026 Monterey overview:
  https://www.oceansciencetechnology.com/events/oceansconference/
