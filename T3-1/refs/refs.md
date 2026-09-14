# T3-1 (v2) 참고문헌 (검증 완료)

2026-09-14에 Europe PMC REST API / Crossref API / PMC 본문에서 서지 정보를 확인했다(v1에서 확인한 문헌 포함).
번호는 초안 본문 인용 순서와 같다. 여기 없는 문헌은 직접 원문을 확인한 뒤에만 추가할 것.

## 본문에 인용한 문헌

| # | 문헌 | DOI / PMID | 한 줄 메모 (초록 기준) |
|---|---|---|---|
| 1 | Geirhos R, Jacobsen JH, Michaelis C, et al. Shortcut learning in deep neural networks. *Nat Mach Intell*. 2020;2(11):665-673. | DOI 10.1038/s42256-020-00257-z (PubMed 미색인) | 지름길 학습 개념 정의: 분포 안에서는 잘 맞히지만 의도하지 않은 쉬운 단서에 기대어 분포 밖에서 실패 |
| 2 | Zech JR, Badgeley MA, Liu M, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLoS Med*. 2018;15(11):e1002683. | DOI 10.1371/journal.pmed.1002683 · PMID 30399157 · PMC6219764 | 3개 병원 흉부 X선 158,323장. CNN이 병원을 99.95% 이상 식별; 유병률이 다른 병원을 섞어 학습하면 내부 AUC 0.931 → 외부 0.815 |
| 3 | DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. *Nat Mach Intell*. 2021;3(7):610-619. | DOI 10.1038/s42256-021-00338-7 (PubMed 미색인) | COVID-19 흉부 X선 모델이 설명 기법으로 보면 병변이 아닌 데이터 출처 관련 단서를 사용, 외부 데이터에서 성능 저하 |
| 4 | Badgeley MA, Zech JR, Oakden-Rayner L, et al. Deep learning predicts hip fracture using confounding patient and healthcare variables. *NPJ Digit Med*. 2019;2:31. | DOI 10.1038/s41746-019-0105-1 · PMID 31304378 · PMC6550136 | 골반 X선 17,587장. 스캐너 기종 AUC 1.00으로 식별; 골절 AUC 0.78이 환자·병원 과정 변수를 균형 맞춘 시험 세트에서 0.52 |
| 5 | Nguyen B, Fong C, Luthra A, et al. Genomic characterization of metastatic patterns from prospective clinical sequencing of 25,000 patients. *Cell*. 2022;185(3):563-575.e11. | DOI 10.1016/j.cell.2022.01.003 · PMID 35120664 · PMC9147702 | MSK-MET 원 논문. 전이는 병리 보고서 + ICD 청구 코드를 21개 장기로 매핑, 부담은 "throughout a patient's clinical course", 차트 검토 대비 민감도 중앙값 77%. 패널 341(n=1,801)/410(n=6,372)/468(n=17,602), 시퀀싱 2013-11-18 ~ 2020-01-06 |
| 6 | Cheng DT, Mitchell TN, Zehir A, et al. Memorial Sloan Kettering-Integrated Mutation Profiling of Actionable Cancer Targets (MSK-IMPACT): a hybridization capture-based next-generation sequencing clinical assay for solid tumor molecular oncology. *J Mol Diagn*. 2015;17(3):251-264. | DOI 10.1016/j.jmoldx.2014.12.006 · PMID 25801821 · PMC5808190 | MSK-IMPACT 341 유전자 표적 패널 검사법 (이후 410/468로 확장) |
| 7 | Jiang B, Mu Q, Qiu F, et al. Machine learning of genomic features in organotropic metastases stratifies progression risk of primary tumors. *Nat Commun*. 2021;12(1):6692. | DOI 10.1038/s41467-021-27017-w · PMID 34795255 · PMC8602327 | MSK-IMPACT 등 표적 패널 기반 전이/장기 예측 선행 연구 (v1 확인) |
| 8 | Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nat Rev Genet*. 2010;11(10):733-739. | DOI 10.1038/nrg2825 · PMID 20838408 · PMC3880143 | 고처리량 유전체 데이터의 배치 효과(처리 시기·시약·인력)가 관심 결과와 얽히면 거짓 결론 |
| 9 | Howard FM, Dolezal J, Kochanny S, et al. The impact of site-specific digital histology signatures on deep learning model accuracy and bias. *Nat Commun*. 2021;12(1):4423. | DOI 10.1038/s41467-021-24698-1 · PMID 34285218 · PMC8292530 | TCGA 3,000명 이상: 조직 제출 기관을 조직 영상에서 식별 가능(색 정규화 후에도) → 생존·변이 예측 성능 과대추정; 기관이 겹치지 않게 분할 제안 |
| 10 | Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. *BMJ*. 2018;361:k1479. | DOI 10.1136/bmj.k1479 · PMID 29712648 · PMC5925441 | EHR 669,452명: 검사 결과가 아니라 검사를 "했다는 사실·시각"이 생존과 연관(86%의 검사) — 의료 과정 흔적 |
| 11 | Cerami E, Gao J, Dogrusoz U, et al. The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. *Cancer Discov*. 2012;2(5):401-404. | DOI 10.1158/2159-8290.CD-12-0095 · PMID 22588877 · PMC3956037 | cBioPortal (데이터 배포처) |
| 12 | Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. | https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree | 사용 모델 (쪽수 미확인이라 생략) |
| 13 | Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell*. 2020;2(1):56-67. | DOI 10.1038/s42256-019-0138-9 · PMID 32607472 · PMC7326367 | TreeSHAP |

## 예비 후보 (본문 미인용)

| 문헌 | DOI / PMID | 메모 |
|---|---|---|
| Gao J, Aksoy BA, Dogrusoz U, et al. Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal. *Sci Signal*. 2013;6(269):pl1. | DOI 10.1126/scisignal.2004088 · PMID 23550210 | cBioPortal 동반 인용 (분량 때문에 뺌) |
| Zehir A, Benayed R, Shah RH, et al. Mutational landscape of metastatic cancer revealed from prospective clinical sequencing of 10,000 patients. *Nat Med*. 2017;23(6):703-713. | DOI 10.1038/nm.4333 · PMID 28481359 · PMC5461196 | 341(2,809건)/410(8,136건) 패널, 2014-01 ~ 2016-05 등록. **468 패널은 이 논문에 없음** |
| Oakden-Rayner L, Dunnmon J, Carneiro G, Ré C. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *Proc ACM CHIL*. 2020:151-159. | DOI 10.1145/3368555.3384468 · PMID 33196064 · PMC7665161 | 숨은 하위집단에서의 성능 실패 |
| Roberts M, Driggs D, Thorpe M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. *Nat Mach Intell*. 2021;3(3):199-217. | DOI 10.1038/s42256-021-00307-0 | 2,212편 체계적 문헌고찰; 교란된 코호트·외부 검증 부재 |
| Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NIPS 2017*. | proceedings.neurips.cc (v1 확인) | SHAP 원 논문 |

## 라벨·데이터 수집 관련 원문 확인 (Nguyen 2022, PMC9147702)
- 전이 사건: 시퀀싱 검체의 병리 보고서 + EHR ICD 청구 코드를 21개 장기로 체계적으로 매핑.
- 전이 부담: "number of distinct organs ... affected by metastases throughout a patient's clinical course".
- 수기 차트 검토 대비 민감도 중앙값 77%.
- 환자당 1검체 선택 규칙: FACETS QC 통과 > 순도 최고 > 커버리지 최고 > 최신 패널.
- 패널 버전별 검체 수와 시퀀싱 기간(2013-11-18 ~ 2020-01-06) — 이 논문 전체 25,775명 기준.
→ 라벨은 진단 시점 전이가 아니라 **전체 경과 동안 기록된 전이**. 일찍 검사한 환자일수록 기록될 시간이 길다.
