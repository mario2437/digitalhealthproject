# T1 참고문헌 목록 (검증: PubMed/DOI/출판사 페이지, 2026-09-14)

## 본문 인용 (인용 순서)
1. Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial intelligence. N Engl J Med. 2021;385(3):283-286. DOI 10.1056/NEJMc2104626, PMID 34260843. — 서신(Letter). 의료 AI 데이터셋 이동의 원인 유형(기술·인구·행태 변화 등) 정리, 정량 분석 없음.
2. Subbaswamy A, Saria S. From development to deployment: dataset shift, causality, and shift-stable models in health AI. Biostatistics. 2020;21(2):345-352. DOI 10.1093/biostatistics/kxz041, PMID 31742354. — 인과 그래프로 이동을 설명하는 개념 논문, 이동에 안정적인 모형 제안.
3. Van Calster B, McLernon DJ, van Smeden M, et al.; STRATOS Topic Group. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17(1):230. DOI 10.1186/s12916-019-1466-7, PMID 31842878. — 보정도 수준·지표(절편, 기울기, 곡선)와 재보정 필요성 정리.
4. Davis SE, Lasko TA, Chen G, Siew ED, Matheny ME. Calibration drift in regression and machine learning models for acute kidney injury. J Am Med Inform Assoc. 2017;24(6):1052-1061. DOI 10.1093/jamia/ocx030, PMID 28379439. — 미국 VA EHR에서 AKI 모형의 시간에 따른 보정 이동 정량화, 원인 귀속 없음.
5. Guo LL, Pfohl SR, Fries J, et al. Evaluation of domain generalization and adaptation on improving model robustness to temporal dataset shift in clinical medicine. Sci Rep. 2022;12(1):2726. DOI 10.1038/s41598-022-06484-1, PMID 35177653. — MIMIC-IV 연도 그룹별 성능 저하, 도메인 일반화·적응 효과 제한적.
6. 질병관리청. 국민건강영양조사 제8기(2019-2021)·제9기(2022-2024) 원시자료 이용지침서. — 혈압계(2020·2021)·허리둘레 줄자(2019)·검사기관(2022) 변경과 연구자용 전환식(HbA1c, 혈압, 중성지방) 출처. 제7기 지침서 부록1(임상검사 분석기관·방법)도 참고.
7. Kim SM, Cho JM. Development and temporal external validation of a high-specificity XGBoost rule-in model for diabetes in middle-aged and older Korean adults. Diabetol Metab Syndr. 2026;18:168. DOI 10.1186/s13098-026-02176-2, PMID 42226262. — KNHANES 2020-2022 개발, 2023 시간적 검증(단일 연도). 가장 가까운 국내 선행연구.
8. Austin PC, Steyerberg EW. The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. Stat Med. 2019;38(21):4051-4065. DOI 10.1002/sim.8281, PMID 31270850. — ICI 정의(loess 기반).
9. van Klaveren D, Gönen M, Steyerberg EW, Vergouwe Y. A new concordance measure for risk prediction models in external validation settings. Stat Med. 2016;35(23):4136-4152. DOI 10.1002/sim.6997, PMID 27251001. — 모형기반 c-통계량(mbc): 판별력 변화가 사례 구성 때문인지 구분.
10. Shimodaira H. Improving predictive inference under covariate shift by weighting the log-likelihood function. J Stat Plan Inference. 2000;90(2):227-244. DOI 10.1016/S0378-3758(00)00115-4 (PubMed 미수록). — 공변량 이동 개념과 밀도비 가중 우도.

## 검증했으나 본문 미인용 (분량 조절용 후보)
- Janssen KJ, et al. Updating methods improved the performance of a clinical prediction model in new patients. J Clin Epidemiol. 2008;61(1):76-86. DOI 10.1016/j.jclinepi.2007.04.018, PMID 18083464. — 절편 재보정 등 모형 갱신 방법 비교.
- Steyerberg EW, et al. Validation and updating of predictive logistic regression models. Stat Med. 2004;23(16):2567-2586. DOI 10.1002/sim.1844, PMID 15287085. — 표본 크기별 갱신 방법.
- Guo LL, et al. Systematic review of approaches to preserve ML performance in the presence of temporal dataset shift in clinical medicine. Appl Clin Inform. 2021;12(4):808-815. DOI 10.1055/s-0041-1735184, PMID 34470057.
- Rabanser S, Günnemann S, Lipton ZC. Failing loudly: an empirical study of methods for detecting dataset shift. NeurIPS 2019:1396-1408. arXiv:1810.11953. — 도메인 분류기 등 이동 탐지 방법 비교.
- Sahiner B, et al. Data drift in medical machine learning. Br J Radiol. 2023;96(1150):20220878. DOI 10.1259/bjr.20220878, PMID 36971405. — 측정·장비 변화 포함 드리프트 리뷰.
- Kweon S, et al. Data resource profile: KNHANES. Int J Epidemiol. 2014;43(1):69-77. DOI 10.1093/ije/dyt228, PMID 24585853.
- Lee YH, et al. A simple screening score for diabetes for the Korean population. Diabetes Care. 2012;35(8):1723-1730. DOI 10.2337/dc11-2347, PMID 22688547. — KNHANES 2001·2005 개발, 2007-2008 검증.
- Ha KH, Kim DJ. Changes in the epidemiological landscape of diabetes in South Korea. Endocrinol Metab. 2024;39(5):669-677. DOI 10.3803/EnM.2024.2073, PMID 39322188. — 2022 KNHANES HbA1c 검사기관 변경을 추이 비교 주의점으로 언급(정량화 없음).
- Casacchia NJ, et al. Development, validation and recalibration of a prediction model for prediabetes: an EHR and NHANES-based study. BMC Med Inform Decis Mak. 2024;24(1):387. DOI 10.1186/s12911-024-02803-w, PMID 39696411.
- Selvin E, et al. Trends in prevalence and control of diabetes in the United States, 1988-1994 and 1999-2010. Ann Intern Med. 2014;160(8):517-525. DOI 10.7326/M13-2411, PMID 24733192. — NHANES 주기 간 HbA1c 보정 선례.

주의: 2022년 HbA1c 변경의 영향을 정량화한 동료심사 논문은 찾지 못함(2026-09 기준).
