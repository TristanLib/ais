# The Journal of Navigation 投稿操作说明

这份说明用于正式登录 ScholarOne 投稿时对照填写。最终提交前，请以
ScholarOne 页面中的实时提示为准。

## 1. 进入投稿系统

1. 打开 The Journal of Navigation 的 Cambridge 页面。
2. 点击 Submit your article，进入 ScholarOne Manuscripts。
3. 如果没有 ScholarOne 账号，先用通讯作者邮箱 `li.bo@cmaritime.com.cn` 注册。
4. 选择新投稿，期刊选择 The Journal of Navigation。

## 2. 选择文章类型

- Article type 选择 Research Article。
- 不选择付费 Gold Open Access，除非已经确认有经费或机构协议覆盖 APC。
- 如果系统询问彩色图，选择线上彩色即可，不申请纸质彩色印刷。

## 3. 填标题和摘要

从 `paper/jon_scholarone_metadata.md` 复制：

- Title: A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support
- Short title: AIS Benchmark for Risk Warning
- Abstract: 复制 ScholarOne metadata 文件中的 Abstract。
- Keywords: AIS; maritime navigation; trajectory prediction; risk warning。

注意：JON 要求关键词在系统中选择或填写，不需要放进英文主文正文。

## 4. 填作者信息

- Author: Li Bo
- Affiliation: China Maritime Service Center, China
- Email: li.bo@cmaritime.com.cn
- Corresponding author: Yes
- ORCID: 如果有 ORCID，就填；如果没有，可以先不填或投稿前注册。

## 5. 填声明

Funding statement:

> This research received no specific grant from any funding agency, commercial or not-for-profit sectors.

Competing interests:

> The author declares no competing interests.

Data and code availability:

> The code, configuration files, generated figures, compact evidence artefacts and manuscript-generation workflow are available at https://github.com/TristanLib/ais, archived under tag `jon-submission-v1.3`. No separate archival DOI is available for this release at the time of submission. The source data are derived from public NOAA MarineCadastre.gov historical AIS files subject to NOAA data access terms; the repository does not redistribute raw NOAA AIS files or processed NumPy arrays.

AI-use declaration:

> OpenAI Codex/ChatGPT was used in May 2026 to assist with code generation, manuscript drafting, document structuring and consistency checks against repository evidence files. The author is responsible for all content and has verified the numerical claims against generated evidence files.

## 6. 上传文件

建议上传：

| 文件 | 作用 |
|---|---|
| `paper/jon_manuscript.docx` | 主文稿 |
| `paper/jon_supplementary_materials.zip` | 补充材料 |
| `paper/figures/jon_*.png` | 如果系统要求单独上传图，就上传这些图 |
| `paper/jon_cover_letter.md` | 封面信文本来源，复制到系统或上传为 cover letter |

`paper/jon_manuscript.pdf` 是本地审阅用 PDF。ScholarOne 通常会根据 Word
自动生成评审 PDF，最终以系统生成 PDF 为准。

## 7. 推荐审稿人

如果系统要求推荐审稿人，可以先留空或稍后补。不要推荐以下人员：

- 近期合作者或共同作者；
- 同一单位人员；
- 导师、学生、直接上下级；
- 有项目、经费、商业合作或私人关系的人；
- 明显会因论文结果受益或受损的人。

## 8. 最后检查

提交前必须打开 ScholarOne 自动生成的 PDF，逐页检查：

- 标题、作者、单位、邮箱是否正确；
- 摘要是否完整；
- 图是否显示，图题是否在图下方；
- 表格有没有断裂、乱码、重叠；
- 参考文献有没有明显格式错误；
- Supplementary material 是否上传成功；
- 声明是否和主文一致。

如果系统生成 PDF 没问题，再点击最终提交。
