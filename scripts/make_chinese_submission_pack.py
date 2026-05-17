#!/usr/bin/env python3
"""Generate a Chinese domestic-journal style manuscript from current artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def best_by_split(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        split = row.get("split_policy", "")
        if split not in best or float(row["ade_meters"]) < float(best[split]["ade_meters"]):
            best[split] = row
    return best


def model_note(model: str) -> str:
    notes = {
        "constant_velocity": "经典运动学基线",
        "constant_acceleration": "二阶运动学外推",
        "kalman_filter_cv": "近期速度平滑的CV模型",
        "ridge_lstsq": "正则化线性统计基线",
        "linear_lstsq": "普通最小二乘统计基线",
        "lstm_baseline": "循环神经网络基线",
        "gru_baseline": "门控循环神经网络基线",
        "transformer_baseline": "注意力序列模型基线",
        "tcn_baseline": "时间卷积网络基线",
    }
    return notes.get(model, "")


def model_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| 划分策略 | 模型 | ADE/m | FDE/m | MAE/m | 说明 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        lines.append(
            "| {split} | `{model}` | {ade} | {fde} | {mae} | {note} |".format(
                split=row.get("split_policy", ""),
                model=row.get("model", ""),
                ade=fmt(row.get("ade_meters"), 1),
                fde=fmt(row.get("fde_meters"), 1),
                mae=fmt(row.get("mae_meters"), 1),
                note=model_note(row.get("model", "")),
            )
        )
    return "\n".join(lines)


def risk_table(risk_metrics: dict[str, Any]) -> str:
    lines = [
        "| 模型 | 场景数 | 精确率 | 召回率 | 误报率 | 漏报率 | 平均CPA绝对误差/n mile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in sorted(risk_metrics.get("metrics_by_model", {}).items()):
        lines.append(
            "| `{model}` | {n} | {precision} | {recall} | {far} | {miss} | {cpa} |".format(
                model=model,
                n=row.get("n_scenarios", ""),
                precision=fmt(row.get("precision")),
                recall=fmt(row.get("recall")),
                far=fmt(row.get("false_alarm_rate")),
                miss=fmt(row.get("missed_warning_rate")),
                cpa=fmt(row.get("mean_abs_cpa_error_nmi")),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--final-dir", default="outputs/final_multiday")
    parser.add_argument("--risk-dir", default="outputs/final_risk")
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--submission-dir", default="outputs/final_submission")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    final_dir = Path(args.final_dir)
    risk_dir = Path(args.risk_dir)
    audit_dir = Path(args.audit_dir)
    submission_dir = Path(args.submission_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = read_json(audit_dir / "multiday_data_manifest.json")
    readiness = read_json(submission_dir / "readiness_report.json")
    risk_metrics = read_json(risk_dir / "risk_metrics.json")
    run_manifest = read_json(final_dir / "run_manifest.json")
    model_rows = read_csv(final_dir / "model_metrics.csv")
    best = best_by_split(model_rows)
    dataset = data_manifest.get("dataset_summary", {})
    source_dates = dataset.get("source_dates", [])
    risk_generation = risk_metrics.get("scenario_generation", {})
    temporal_best = best.get("temporal_test", {})
    vessel_best = best.get("vessel_disjoint_test", {})
    generated_at = datetime.now(timezone.utc).isoformat()

    title_cn = "面向风险预警的AIS船舶轨迹预测可复现实证研究"
    title_en = "A Reproducible AIS Trajectory Prediction Benchmark for Risk-Warning Support"
    manuscript = f"""# {title_cn}

作者：［作者姓名］

单位：［作者单位，城市 邮编］

基金项目：［基金项目名称及编号］

中图分类号：U675　文献标识码：A

## 摘要

针对船舶短时轨迹预测研究中复杂神经网络模型常被默认视为优选模型、而数据划分和度量口径不足以复核的问题，本文构建了一个面向AIS历史数据的可复现实证评估流程。该流程保留船舶标识、时间、数据源日期、区域、航速和转向强度等元数据，支持时间外推测试、船舶不相交测试、场景分组误差分析以及基于CPA/TCPA的风险预警评估。当前证据包使用NOAA MarineCadastre AIS数据，形成 {dataset.get('sample_count')} 个轨迹窗口，覆盖 {dataset.get('unique_mmsi_count')} 个MMSI，数据源日期为 {', '.join(source_dates)}。在当前分层时间块协议下，Kalman风格的常速度平滑模型在时间保持测试中取得最低ADE（{fmt(temporal_best.get('ade_meters'), 1)} m），在船舶不相交测试中同样取得最低ADE（{fmt(vessel_best.get('ade_meters'), 1)} m）。风险预警实验基于 {risk_generation.get('scenario_count')} 个AIS衍生会遇场景，评估预警精确率、召回率、误报率、漏报率和CPA误差。结果表明，在该短时预测任务中，简单运动学模型仍是必须严肃对照的强基线；若缺少严格的数据审计、划分协议和调参记录，直接宣称深度模型优越并不稳妥。

关键词：AIS；船舶轨迹预测；碰撞风险预警；CPA/TCPA；可复现实验；基线模型

## Abstract

This paper presents a reproducible AIS trajectory-prediction benchmark for short-horizon maritime risk-warning support. The current evidence package keeps vessel identifiers, timestamps, source dates, region labels, speed statistics and turning-intensity metadata, enabling temporal holdout, vessel-disjoint testing, scenario-slice analysis and CPA/TCPA-based warning evaluation. Under the current stratified time-block protocol, a Kalman-style constant-velocity smoother achieves the best ADE on both temporal and vessel-disjoint tests. The results support a conservative methodological conclusion: simple kinematic baselines remain strong competitors, and neural architecture-superiority claims require auditable preprocessing, split protocols, tuning records and downstream decision-support evidence.

Keywords: AIS; vessel trajectory prediction; collision risk warning; CPA/TCPA; reproducible benchmark

## 1 引言

船舶自动识别系统（AIS）为海上交通态势感知、航行安全评估和短时轨迹预测提供了重要数据基础。近年来，LSTM、GRU、Transformer等深度学习模型被广泛用于船舶轨迹预测任务，但在实际工程研究中，模型性能往往同时受到数据清洗、轨迹切片、坐标度量、训练/测试划分以及异常航迹的影响。如果缺少可复核的证据链，复杂模型相对于简单运动学基线的优势并不容易被可靠确认。

本文的出发点不是证明某一种神经网络结构必然优于传统方法，而是建立一个可审计的AIS轨迹预测与风险预警评估流程。该流程要求每一个核心数值均能追溯至仓库中的数据清单、模型输出、逐样本误差、统计检验和风险评估文件。基于当前项目产物，本文重点回答三个问题：第一，简单运动学基线在短时AIS轨迹预测中是否仍具有竞争力；第二，时间保持和船舶不相交划分下模型泛化表现是否一致；第三，轨迹预测误差如何影响基于CPA/TCPA的会遇风险预警结果。

本文的主要贡献包括：（1）构建保留MMSI、时间、区域、航速和转向强度等元数据的AIS短时轨迹预测评估流程；（2）在统一输入输出协议下比较运动学、统计学习和神经网络模型，避免缺少强基线导致的模型优越性误判；（3）将轨迹预测结果进一步映射到CPA/TCPA风险预警指标，讨论预测误差对航行安全决策支持的影响。

## 2 数据与任务定义

### 2.1 数据来源与预处理

本文使用NOAA MarineCadastre公开AIS历史数据。当前中文稿对应的证据包包含4个数据源日期：{', '.join(source_dates)}。数据构建脚本记录原始文件校验和、记录数量、MMSI数量、时间范围、经纬度范围、处理后文件校验和以及划分策略。处理后的轨迹窗口采用WGS84经纬度坐标，预测误差使用Haversine距离或局部北东分量距离计算，避免将经纬度角度误作平面米制距离。

当前处理后数据包含 {dataset.get('sample_count')} 个样本、{dataset.get('unique_mmsi_count')} 个MMSI。时间保持划分为 {dataset.get('temporal_split_counts')}；船舶不相交划分为 {dataset.get('vessel_split_counts')}。区域标签包括 {', '.join(dataset.get('regions', []))}。该数据协议采用分层时间块采样，适合支持当前证据包中的短时预测和风险预警分析；若要扩展为全天候、季节性或航区级宏观结论，还需要进一步运行全日或更多时间块敏感性实验。

### 2.2 预测任务

每个样本使用 {run_manifest.get('config', {}).get('experiment', {}).get('history_steps')} 个一分钟历史步长，预测未来 {run_manifest.get('config', {}).get('experiment', {}).get('forecast_steps')} 个一分钟位置。输入特征包括纬度、经度、对地航速以及航向角的正余弦表示。主要轨迹预测指标为平均位移误差（ADE）、最终位移误差（FDE）、均方根误差（RMSE）和平均绝对误差（MAE）。

设第 i 个样本在第 t 个预测步的真实位置和预测位置分别为 p_i,t 和 p_hat_i,t，Haversine距离为 d(·,·)，则ADE和FDE可表示为：

ADE = 1/(N T) sum_i sum_t d(p_i,t, p_hat_i,t)

FDE = 1/N sum_i d(p_i,T, p_hat_i,T)

该定义保证了经纬度坐标下的位移误差以米为单位报告，而不是直接在角度坐标上计算欧氏距离。

## 3 方法

### 3.1 轨迹预测模型

为避免只比较复杂模型，本文同时纳入运动学、统计学习和神经网络模型。运动学模型包括常速度外推、常加速度外推以及Kalman风格的常速度平滑模型；统计学习模型包括普通最小二乘和岭回归；神经网络模型包括LSTM、GRU、Transformer和时间卷积网络（TCN）。所有模型使用相同的输入历史长度、预测步长、数据划分和误差指标。

### 3.2 泛化与统计分析

实验同时报告时间保持测试和船舶不相交测试。前者用于观察模型对后续时间片段的外推能力，后者用于考察模型面对未见MMSI船舶时的泛化表现。仓库同时输出逐样本误差、按预测步长的误差、按区域/速度/转向强度分组的误差和配对统计检验结果。

### 3.3 风险预警评估

在风险预警部分，本文不宣称完成自主避碰验证，而是将模型预测轨迹用于CPA/TCPA相关的决策支持评估。系统根据真实未来轨迹构造AIS衍生会遇场景，并在给定搜索半径和预警阈值下比较真实预警标签与预测预警结果，报告精确率、召回率、误报率、漏报率和平均CPA绝对误差。

对于两船相对位置 r 和相对速度 v，若假设短时间内速度近似恒定，则到达最近会遇点的时间可写为 TCPA = - (r·v) / ||v||^2，CPA为该时刻两船相对距离。本文仅将该量作为预警评价指标，不将其解释为完整的船舶避碰控制策略。

## 4 实验结果

### 4.1 轨迹预测结果

表1给出当前证据包中的主要轨迹预测结果。可以看到，Kalman风格的常速度平滑模型在两种划分策略下均取得最低ADE。神经网络模型虽然已经纳入验证集proxy调参和早停记录，但在当前协议下并未支持“神经结构优于简单运动学模型”的结论。

表1 主要轨迹预测结果

{model_table(model_rows)}

### 4.2 风险预警结果

风险预警实验共评估 {risk_generation.get('scenario_count')} 个AIS衍生会遇场景，评估样本数为 {risk_generation.get('evaluated_samples')}，预警阈值为 {risk_generation.get('warning_threshold_nmi')} n mile，搜索半径为 {risk_generation.get('search_radius_nmi')} n mile。结果见表2。

表2 基于预测轨迹的风险预警结果

{risk_table(risk_metrics)}

## 5 讨论

当前结果有两点工程含义。第一，在15分钟短时预测任务中，简单运动学模型尤其是带近期速度平滑的常速度模型仍然具有很强的竞争力，应作为船舶轨迹预测研究中的必要基线。第二，轨迹预测研究若要服务航行风险预警，不能只报告ADE或FDE，还应进一步考察预警召回率、误报率、漏报率和CPA误差等下游指标。

同时，本文结果也需要谨慎解释。当前数据协议为分层时间块协议，并非覆盖所选日期的所有分钟，因此不能直接外推为全天候或季节性结论。神经网络调参目前属于验证集proxy搜索，能够提供审稿所需的可审计调参记录，但尚不能支持强结构优越性结论。风险预警实验属于离线决策支持评估，并不等价于闭环自主避碰验证。

## 6 结论

本文形成了一个从AIS数据审计、轨迹预测、泛化评估到风险预警分析的可复现实证流程。当前证据显示，在所采用的短时预测协议下，Kalman风格常速度平滑模型优于多种神经网络基线；深度学习模型的有效性需要在更严格的调参、更多数据协议和外部验证下重新检验。本文适合以“可复现基准评估”和“面向风险预警的工程证据链”为核心贡献，而不宜表述为深度学习方法优越性的论文。

## 数据与代码可用性

本文所有数值均来自当前仓库产物。关键证据文件包括：

- `outputs/audit/multiday_data_manifest.json`
- `outputs/final_multiday/model_metrics.csv`
- `outputs/final_multiday/neural_tuning_protocol.json`
- `outputs/final_multiday/statistical_tests.json`
- `outputs/final_risk/risk_metrics.json`
- `outputs/final_submission/readiness_report.json`

## 利益冲突声明

作者声明不存在与本文研究相关的利益冲突。正式投稿时请根据作者实际情况修改。

## 参考文献

[1] NOAA Office for Coastal Management and Bureau of Ocean Energy Management. MarineCadastre.gov AIS Data. 2024.

[2] International Maritime Organization. Convention on the International Regulations for Preventing Collisions at Sea, 1972 (COLREGs). 1972.

[3] Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997, 9(8): 1735-1780.

[4] Cho K, van Merrienboer B, Gulcehre C, et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP, 2014.

[5] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.

[6] Bai S, Kolter J Z, Koltun V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271, 2018.

[7] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS, 2019.
"""

    manuscript_path = paper_dir / "submission_manuscript_zh.md"
    manuscript_path.write_text(manuscript, encoding="utf-8")
    manifest = {
        "created_at": generated_at,
        "manuscript": str(manuscript_path),
        "source_artifacts": {
            "data_manifest": str(audit_dir / "multiday_data_manifest.json"),
            "model_metrics": str(final_dir / "model_metrics.csv"),
            "risk_metrics": str(risk_dir / "risk_metrics.json"),
            "readiness_report": str(submission_dir / "readiness_report.json"),
        },
        "readiness_status_at_generation": readiness.get("overall_status"),
        "blocking_gaps_at_generation": readiness.get("blocking_gaps", []),
        "scope_note": "中文稿为国内期刊投稿候选稿，仍需按目标期刊模板进行人工排版和GB/T 7714参考文献整理。",
    }
    (submission_dir / "chinese_submission_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Chinese manuscript written to {manuscript_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
