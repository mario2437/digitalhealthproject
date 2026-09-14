# T2(v2) 참고문헌 목록 (확인 완료)

모든 문헌은 PubMed/Europe PMC 기록, Crossref DOI 조회, arXiv, 학술지·Hugging Face 페이지로 서지사항을 확인했다(2026-09-14 기준).
대괄호 번호는 초안(`draft/초안_심사용.md`)의 인용 순서다. "후보"는 확인했지만 초안에 인용하지 않은 문헌이다.
수치 메모 중 "초록 기준"은 원문 전체가 아니라 초록·데이터베이스 기록으로 확인한 값이다. 원고에 수치를 더 넣을 때는 원문을 다시 확인한다.

## 초안 인용 문헌 (인용 순서)

[1] Monteiro CA, Cannon G, Levy RB, Moubarac JC, Louzada ML, Rauber F, Khandpur N, Cediel G, Neri D, Martinez-Steele E, Baraldi LG, Jaime PC. Ultra-processed foods: what they are and how to identify them. *Public Health Nutr* 2019;22(5):936-941.
- doi:10.1017/S1368980018003762 · PMID 30744710
- 메모: NOVA 4개 군 정의와 초가공식품 판별 요령. 증류주를 4군으로 둔 근거. 정답 배정 프로토콜과 LLM 프롬프트의 정의 출처.

[2] 박해진, 박소현, 김지영. 한국형 NOVA 식품분류체계의 개발 및 한국 성인의 초가공식품 섭취량 추정: 2018년 국민건강영양조사 자료를 활용하여. *대한지역사회영양학회지* 2022;27(6):455-467.
- doi:10.5720/kjcn.2022.27.6.455 · KCI
- 메모: 국민건강영양조사 식품코드를 전문가가 수작업으로 NOVA에 배정. 한국 연구는 수작업 분류가 표준이라는 근거.

[3] Braesco V, Souchon I, Sauvant P, Haurogné T, Maillot M, Féart C, Darmon N. Ultra-processed foods: how functional is the NOVA system? *Eur J Clin Nutr* 2022;76(9):1245-1253.
- doi:10.1038/s41430-022-01099-1 · PMID 35314769
- 메모(초록 기준): 프랑스 전문가들의 NOVA 배정 일치도가 낮음. Fleiss kappa 0.32(원재료 정보가 있는 시판식품 120개, 159명), 0.34(일반식품 111개, 177명).

[4] Sneed NM, Ukwuani S, Sommer EC, et al. (저자 9명) Reliability and validity of assigning ultraprocessed food categories to 24-h dietary recall data. *Am J Clin Nutr* 2023;117(1):182-190.
- doi:10.1016/j.ajcnut.2022.10.016 · PMID 36789937
- 메모(초록 기준): 24시간 회상 식품 3,099개를 두 코더가 배정. 일치율 88.3%, kappa 0.75. 사람 배정의 기준선으로 쓴다.

[5] Menichetti G, Ravandi B, Mozaffarian D, Barabási AL. Machine learning prediction of the degree of food processing. *Nat Commun* 2023;14(1):2312.
- doi:10.1038/s41467-023-37457-1 · PMID 37085506
- 메모: FoodProX. 입력은 영양소 함량(식품명 아님), 미국 FNDDS 2009-2010의 수작업 NOVA 라벨 2,484개로 랜덤포레스트 학습. 군별 AUC 약 0.96-0.98.

[6] Hu G, Flexner N, Tiscornia MV, L'Abbé MR. Accelerating the classification of NOVA food processing levels using a fine-tuned language model: a multi-country study. *Nutrients* 2023;15(19):4167.
- doi:10.3390/nu15194167 · PMID 37836451
- 메모: 캐나다 포장식품 원재료명(영어) 텍스트로 BERT 계열 모델 미세조정. 최고 F1 약 0.98, 타국 자료 F1 0.89-0.95.

[7] Cattem MVO, Machado FO, da Silva MEG, Canella DS, Koury JC. Comparing ChatGPT and DeepSeek for ultra-processed food classification. *Nutrition* 2026;144:113066.
- doi:10.1016/j.nut.2025.113066 · PMID 41505813
- 메모(초록 기준): 브라질 식품성분표(TBCA) 식품명 1,168개를 상용 LLM으로 분류. 정확도 ChatGPT o1 98.0%, DeepSeek-R1 92.6%. 이진/4범주 여부는 원문 확인 필요.

[8] Ase A, Borowicz J, Rakocy K, Piekarska B. Large language models for real-world nutrition assessment: structured prompts, multi-model validation and expert oversight. *Nutrients* 2026;18(1):23.
- doi:10.3390/nu18010023 · PMID 41515141
- 메모(초록 기준): 폴란드 식품 1,992개를 상용 LLM 3종(Claude, Gemini, GPT)으로 NOVA 분류, 전문가와 일치 90.3-94.2%. 비영어 식품명 사례.

[9] 질병관리청. 국민건강영양조사 제8기(2019-2021) 원시자료 이용지침서; 제9기(2022-2024) 원시자료 이용지침서. 청주: 질병관리청.
- https://knhanes.kdca.go.kr
- 메모: 자료원. 24시간 회상 변수(N_FCODE, N_FNAME, N_PRODUCT, N_CNAME, N_KINDG1), 통합 가중치(wt_tot, 기수 간 통합 규칙), 식품성분표 제10개정판.

[10] Qwen Team (Yang A, Yang B, Zhang B, et al.). Qwen2.5 technical report. arXiv:2412.15115, 2024. 모델 카드: Qwen/Qwen2.5-7B-Instruct, https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- 메모: Apache-2.0 라이선스, 7.61B 파라미터. 모델 카드에 한국어를 포함한 29개 이상 언어 지원 명시. 14B(Qwen2.5-14B-Instruct)도 Apache-2.0.

[11] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res* 2011;12:2825-2830.
- https://jmlr.org/papers/v12/pedregosa11a.html
- 메모: TF-IDF, 로지스틱 회귀, 선형 SVM, 교차검증 구현.

## 후보 문헌 (확인 완료, 미인용)

- Martinez-Steele E, Khandpur N, Batis C, et al. Best practices for applying the Nova food classification system. *Nat Food* 2023;4(6):445-448. doi:10.1038/s43016-023-00779-w · PMID 37264165. 메모: NOVA 적용 3단계 권고(수치 없음). 프로토콜 근거로 쓸 수 있다.
- Arora N, Bhagat S, Dhama R, Bagler G. Machine learning and natural language processing models to predict the extent of food processing. *J Food Compos Anal* 2025;146:107938. doi:10.1016/j.jfca.2025.107938 · arXiv 2412.17217. 메모: FNDDS 영양소로 LightGBM F1 약 0.94.
- Campbell JL, Schofield G, Tiedt HR, Zinn C. Artificial intelligence applications for assessing ultra-processed food consumption: a scoping review. *Br J Nutr* 2026;135(4):463-473. doi:10.1017/S000711452510593X · PMID 41423345. 메모: AI 기반 UPF 평가 연구 9편, 모두 NOVA, F1 0.86-0.98.
- Ravandi B, Ispirova G, Sebek M, Mehler P, Barabási AL, Menichetti G. Prevalence of processed foods in major US grocery stores. *Nat Food* 2025;6(3):296-308. doi:10.1038/s43016-024-01095-7 · PMID 39806219. 메모: GroceryDB, 5만여 제품에 FoodProX 계열 점수.
- Cavero-Redondo I, et al. Validity and reliability of tools to measure ultraprocessed food intake under the NOVA system: a systematic review. *Am J Clin Nutr* 2026;123(5):101263. doi:10.1016/j.ajcnut.2026.101263 · PMID 41796966. 메모: 도구별 kappa 약 0.36-0.84.
- Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas* 1960;20(1):37-46. doi:10.1177/001316446002000104.
- Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics* 1977;33(1):159-174. doi:10.2307/2529310 · PMID 843571.
- Wang L, Yang N, Huang X, Yang L, Majumder R, Wei F. Multilingual E5 text embeddings: a technical report. arXiv:2402.05672, 2024. 메모: 임베딩 기반 분류를 추가할 때 인용.

## 신규성 서술 근거

- 한국 식품이나 국민건강영양조사 식품코드를 NOVA로 자동 분류한 연구는 찾지 못했다(2026-09 PubMed·Europe PMC·한국어 웹·KCI 키워드 검색). KCI·DBpia 전문 검색은 제한적이었으므로 "저자들이 아는 한"으로 쓴다.
- 국외 자동화는 영양소 입력[5], 영어 원재료명[6], 상용 LLM[7,8]이 중심이다. 한국어 식품명·제품명만 쓴 비교와, 분류 오류가 모집단 UPF% 추정에 주는 영향을 함께 본 연구는 확인되지 않았다.
- 상용 LLM 연구[7,8]는 외부 API로 자료를 보낸다. 본 연구는 로컬 공개 모델을 쓴다.
