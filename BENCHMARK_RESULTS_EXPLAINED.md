# CheatGPT Benchmark Results - Quick Reference

## 🎯 Key Metrics (77 Sessions, 1,681 Events)

| Metric | Value | Status |
|--------|-------|--------|
| **Precision** | 97.38% | ⚠️ ESTIMATED |
| **Recall** | 86.83% | ⚠️ ESTIMATED |
| **F1-Score** | 0.9181 | ⚠️ ESTIMATED |
| **Avg Confidence** | 85.36% | ✅ MEASURED |
| **High Conf Rate** | 93.5% | ✅ MEASURED |

---

## ⚠️ Important: These Are ESTIMATES

**Metrics based on confidence scores, NOT verified ground truth**

### Estimation Method:
- High confidence (≥75%) → Assumed true positive
- Medium confidence (50-75%) → 60% true, 40% false
- Low confidence (<50%) → Assumed false positive

### Real Accuracy Affected By:
- **Camera angle** - Side views increase false positives
- **Resolution** - <720p reduces accuracy 10-20%
- **Lighting** - Poor lighting drops confidence 15-30%
- **Distance** - >5m from camera reduces reliability
- **Occlusions** - Students blocking each other
- **Motion blur** - Fast movements affect detection

---

## 📊 What You CAN Say (Measured)

✅ **1,681 events** detected across 77 sessions  
✅ **93.5% high confidence** detections (≥75%)  
✅ **85.36% average confidence** across all events  
✅ **79.22% detection rate** (sessions with events)  
✅ **Zero low-confidence** detections (<50%)

---

## ⚠️ What You SHOULD Say (Estimated)

- "ESTIMATED 97.38% precision based on confidence analysis"
- "ESTIMATED 86.83% recall from statistical modeling"
- "ESTIMATED ~44 false alarms (2.62% rate)"
- "Pending ground truth validation"

---

## 🚫 What You CANNOT Say

❌ "Only 44 false alarms" (not verified)  
❌ "Definitive 97.38% precision" (estimate only)  
❌ "Proven accuracy" (needs ground truth)

---

## 🔧 Component Performance

### Phone Detection (667 events, 39.7%)
- YOLO-based hardware detection
- Estimated ~95% accuracy
- Most reliable component

### Head Turning (869 events, 51.7%)
- Pose keypoint-based (≥40° threshold)
- Estimated ~85-90% accuracy
- Affected by camera angle

### Hand Activity (145 events, 8.6%)
- Motion pattern analysis
- Estimated ~75-80% accuracy
- Most challenging component

---

## 🧠 LSTM Model (Measured)

- **Test Accuracy**: 83.87%
- **F1-Score**: 0.7651
- **Precision (Suspicious)**: 84%
- **Recall (Suspicious)**: 100%

---

## 📈 Comparison Standards

| System | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| **CheatGPT (est.)** | 97.38% | 86.83% | 0.9181 |
| Typical CV | 85-92% | 75-85% | 0.80-0.88 |
| Industry Min | >85% | >80% | >0.82 |

---

## 🎓 For Your Defense

### Honest Statement:
*"The system demonstrates strong detection capabilities with 93.5% high-confidence detections averaging 85.36% confidence. Based on confidence-score analysis, we estimate 97.38% precision and 86.83% recall, though these require ground truth validation for verification."*

### Key Points:
1. **Strong detection patterns** - 93.5% high confidence rate
2. **Robust confidence scores** - 85.36% average
3. **Large-scale testing** - 77 sessions, 1,681 events
4. **Multi-component system** - Phone, pose, gesture detection
5. **Transparency** - Estimates pending validation

---

## ✅ Get TRUE Metrics

### Create Ground Truth:
1. Select 10-20 sessions (varied angles/lighting/resolution)
2. Label every 5 seconds: `cheating` or `normal`
3. Document camera conditions
4. Save to `ground_truth.json`
5. Run: `python benchmark_system.py`

### This Gives You:
- **TRUE precision** (verified correct alerts)
- **TRUE recall** (verified missed events)
- **TRUE false positives** (counted, not estimated)
- **Confusion matrix** (TP, FP, TN, FN)

---

## 📋 Bottom Line

### Measured Facts:
- ✅ 1,681 events detected
- ✅ 93.5% high confidence rate
- ✅ 85.36% average confidence
- ✅ 77 sessions tested

### Estimated Performance:
- ⚠️ ~97% precision (confidence-based)
- ⚠️ ~87% recall (statistical model)
- ⚠️ ~2.6% false positive rate

### Need Ground Truth For:
- 🔍 Definitive accuracy metrics
- 🔍 Verified false alarm count
- 🔍 True missed event analysis
- 🔍 Camera-specific performance

