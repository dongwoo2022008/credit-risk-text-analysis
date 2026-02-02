# 코드-논문 매핑 테이블

## 논문 구조와 코드 매핑

### Phase 0: Structured-Only Baseline (구조화 변수만 사용)
**논문 섹션**: 4.1 Structured-Only Models (Phase 0) Performance  
**관련 테이블/그림**: Table 4-1  
**필요한 코드**:
- `phase0_structured_baseline.py` 또는 `phase0_structured_baseline.ipynb`
  - 데이터 로딩 및 전처리
  - 11개 구조화 변수 준비 (Table 2-2 참조)
  - 8개 모델 학습 (LR, SVM, KNN, DT, NB, RF, GB, XGB)
  - 성능 평가 (ROC-AUC, Recall, F1-score)
  - Table 4-1 결과 생성

**입력 데이터**: `sentiment_scoring.25.12.30.xlsx`  
**출력 결과**: 
- `results/phase0/table_4_1_structured_only_performance.csv`
- `results/phase0/phase0_model_comparison.png`

---

### Phase 1: Text-Only Models (텍스트만 사용)
**논문 섹션**: 4.2 Text-only Model Performance (Stages 1–4)  
**관련 테이블/그림**: Table 4-2  
**필요한 코드**:
- `phase1_text_only.py` 또는 `phase1_text_only.ipynb`
  - Stage 1: TF-IDF 기반 텍스트 특징 추출
  - Stage 2: Subword 기반 텍스트 특징 추출
  - Stage 3: MiniLM 임베딩
  - Stage 4: KoSimCSE 임베딩
  - 각 Stage별 8개 모델 학습 및 평가
  - Table 4-2 결과 생성

**입력 데이터**: `sentiment_scoring.25.12.30.xlsx` (텍스트 필드: title, loan purpose, repayment plan)  
**출력 결과**: 
- `results/phase1/table_4_2_text_only_performance.csv`
- `results/phase1/stage_wise_comparison.png`

---

### Phase 2: Merged Models (구조화 + 텍스트 통합)
**논문 섹션**: 4.3 Merged Model Performance (Phase 2)  
**관련 테이블/그림**: Table 4-3  
**필요한 코드**:
- `phase2_merged_models.py` 또는 `phase2_merged_models.ipynb`
  - Stage 1-4 텍스트 특징 + 11개 구조화 변수 결합
  - 각 Stage별 8개 모델 학습 및 평가
  - LR 및 GB 벤치마크와 비교
  - Table 4-3 결과 생성

**입력 데이터**: `sentiment_scoring.25.12.30.xlsx`  
**출력 결과**: 
- `results/phase2/table_4_3_merged_performance.csv`
- `results/phase2/stage_comparison_vs_baseline.png`

---

### Phase 3: Hyperparameter Tuning
**논문 섹션**: 4.4 Hyperparameter Tuning Results (Phase 3)  
**관련 테이블/그림**: Table 4-4  
**필요한 코드**:
- `phase3_hyperparameter_tuning.py` 또는 `phase3_hyperparameter_tuning.ipynb`
  - RF, GB, XGB 하이퍼파라미터 그리드 서치
  - 교차 검증 (Cross-validation)
  - 튜닝 전후 성능 비교
  - Table 4-4 결과 생성

**입력 데이터**: Phase 2에서 준비된 merged features  
**출력 결과**: 
- `results/phase3/table_4_4_tuning_comparison.csv`
- `results/phase3/tuning_results.json`

---

### Phase 4: Ensemble Models
**논문 섹션**: 4.5 Ensemble Model Performance (Phase 4)  
**관련 테이블/그림**: Table 4-5  
**필요한 코드**:
- `phase4_ensemble_models.py` 또는 `phase4_ensemble_models.ipynb`
  - Hard Voting (Voting-H)
  - Soft Voting (Voting-S)
  - Weighted Voting (Voting-W)
  - Blending (BLD)
  - Stacking (STK)
  - Table 4-5 결과 생성

**입력 데이터**: Phase 2/3에서 학습된 base models  
**출력 결과**: 
- `results/phase4/table_4_5_ensemble_performance.csv`
- `results/phase4/ensemble_comparison.png`

---

### Phase 5: Conditional Effect Analysis (조건부 효과 분석)

#### 5.1 Uncertainty-Based Marginal Case Analysis
**논문 섹션**: 4.6.1 Uncertainty-Based Marginal Case Analysis  
**관련 테이블/그림**: Table 4-6, Table 4-7, Figure 4-2  
**필요한 코드**:
- `phase5_1_uncertainty_analysis.py` 또는 `phase5_1_uncertainty_analysis.ipynb`
  - Marginal cases (0.3 ≤ p ≤ 0.7) 추출
  - Clear cases (p < 0.3 or p > 0.7) 추출
  - 각 케이스별 성능 비교
  - Table 4-6, 4-7, Figure 4-2 생성

**출력 결과**: 
- `results/phase5/table_4_6_marginal_cases.csv`
- `results/phase5/table_4_7_clear_cases.csv`
- `results/phase5/figure_4_2_marginal_vs_clear.png`

#### 5.2 Error Case Analysis (FN Recovery)
**논문 섹션**: 4.6.2 Error Case Analysis  
**관련 테이블/그림**: Table 4-8, Table 4-9, Figure 4-3  
**필요한 코드**:
- `phase5_2_error_case_analysis.py` 또는 `phase5_2_error_case_analysis.ipynb`
  - Confusion matrix 비교
  - FN Recovery Rate 계산
  - 고위험/저위험 그룹별 FN 회복률 분석
  - Table 4-8, 4-9, Figure 4-3 생성

**출력 결과**: 
- `results/phase5/table_4_8_confusion_matrix.csv`
- `results/phase5/table_4_9_fn_recovery.csv`
- `results/phase5/figure_4_3_fn_recovery_rate.png`

#### 5.3 Threshold Sensitivity Analysis
**논문 섹션**: 4.6.3 Threshold Sensitivity Analysis  
**관련 테이블/그림**: Table 4-10, Figure 4-4, Figure 4-5  
**필요한 코드**:
- `phase5_3_threshold_sensitivity.py` 또는 `phase5_3_threshold_sensitivity.ipynb`
  - 임계값 변화 (0.3, 0.4, 0.5, 0.6)
  - 고위험/저위험 그룹별 성능 격차 분석
  - Table 4-10, Figure 4-4, 4-5 생성

**출력 결과**: 
- `results/phase5/table_4_10_threshold_sensitivity.csv`
- `results/phase5/figure_4_4_f1_gap.png`
- `results/phase5/figure_4_5_recall_gap.png`

#### 5.4 Text Length/Richness Interaction Analysis
**논문 섹션**: 4.6.4 Text Length as Information Richness  
**관련 테이블/그림**: Table 4-11, Table 4-12, Table 4-13, Figure 4-6, Figure 4-7  
**필요한 코드**:
- `phase5_4_text_length_analysis.py` 또는 `phase5_4_text_length_analysis.ipynb`
  - 텍스트 길이별 decile 분석
  - 길이-디폴트 상관관계 분석
  - Long vs Short 그룹 비교
  - Table 4-11, 4-12, 4-13, Figure 4-6, 4-7 생성

**출력 결과**: 
- `results/phase5/table_4_11_length_decile.csv`
- `results/phase5/table_4_12_length_correlation.csv`
- `results/phase5/table_4_13_long_vs_short.csv`
- `results/phase5/figure_4_6_default_by_length.png`
- `results/phase5/figure_4_7_length_sensitivity.png`

---

## 공통 유틸리티 코드

### 데이터 전처리
- `utils/data_preprocessing.py`
  - 데이터 로딩
  - 결측치 처리
  - 변수 스케일링 (standardization/normalization)
  - Train-test split (80/20, stratified)

### 텍스트 전처리
- `utils/text_preprocessing.py`
  - 텍스트 정규화
  - 토큰화
  - 불용어 제거
  - Rare token 필터링

### 텍스트 특징 추출
- `utils/text_features.py`
  - Stage 1: TF-IDF
  - Stage 2: Subword features
  - Stage 3: MiniLM embeddings
  - Stage 4: KoSimCSE embeddings

### 모델 학습 및 평가
- `utils/model_training.py`
  - 8개 분류 모델 정의 및 학습
  - 성능 평가 함수 (ROC-AUC, Recall, F1-score)

### 시각화
- `utils/visualization.py`
  - 성능 비교 그래프
  - Confusion matrix 시각화
  - Threshold sensitivity 그래프

---

## 재현성을 위한 요구사항

### 환경 설정
- Python 3.8+
- 주요 패키지:
  - pandas, numpy
  - scikit-learn
  - xgboost
  - transformers (for MiniLM, KoSimCSE)
  - matplotlib, seaborn
  - openpyxl (for Excel reading)

### 실행 순서
1. 데이터 준비: `data/sentiment_scoring.25.12.30.xlsx`
2. Phase 0 실행: `python phase0_structured_baseline.py`
3. Phase 1 실행: `python phase1_text_only.py`
4. Phase 2 실행: `python phase2_merged_models.py`
5. Phase 3 실행: `python phase3_hyperparameter_tuning.py`
6. Phase 4 실행: `python phase4_ensemble_models.py`
7. Phase 5 실행: 
   - `python phase5_1_uncertainty_analysis.py`
   - `python phase5_2_error_case_analysis.py`
   - `python phase5_3_threshold_sensitivity.py`
   - `python phase5_4_text_length_analysis.py`

### 재현성 보장
- 모든 실험에서 동일한 random seed 사용
- 동일한 train-test split 유지
- 데이터 전처리 과정 명확히 문서화
