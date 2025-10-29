# CheatGPT - Performance Summary
**Quick Reference for Defense**

---

## 📊 Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Precision** | 97.38% | ⚠️ Estimated |
| **Recall** | 86.83% | ⚠️ Estimated |
| **F1-Score** | 0.9181 | ⚠️ Estimated |
| **Avg Confidence** | 85.36% | ✅ Measured |
| **High Conf Rate** | 93.5% | ✅ Measured |
| **Sessions** | 77 | ✅ Measured |
| **Events** | 1,681 | ✅ Measured |

---

## ✅ What You CAN Say (Verified)

- **1,681 events** detected across 77 sessions
- **93.5% high confidence** rate (≥75%)
- **85.36% average** confidence score
- **79.22% detection** rate (sessions with events)
- **Zero low-confidence** (<50%) detections

---

## ⚠️ What to Say Carefully (Estimated)

- "ESTIMATED 97.38% precision" (confidence-based)
- "ESTIMATED 86.83% recall" (statistical model)
- "Pending ground truth validation"
- "Performance varies with camera setup"

---

## 🚫 Don't Say

- ❌ "Only 44 false alarms" (not verified)
- ❌ "Definitive 97.38% precision"
- ❌ "99.97% accuracy" (misleading)
- ❌ Ignore camera/environmental factors

---

## 🔧 Components

| Component | Events | Est. Accuracy |
|-----------|--------|---------------|
| Phone Detection | 667 (39.7%) | ~95% |
| Head Turning | 869 (51.7%) | ~85% |
| Hand Activity | 145 (8.6%) | ~78% |

---

## 📈 Standards Comparison

| System | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| **CheatGPT (est.)** | 97% | 87% | 0.92 |
| Typical CV | 85-92% | 75-85% | 0.80-0.88 |
| Industry Min | >85% | >80% | >0.82 |

---

## 🎓 Defense Statement

*"The system demonstrates strong detection capabilities with 93.5% high-confidence detections averaging 85.36%. Based on confidence analysis, we estimate 97.38% precision and 86.83% recall, pending ground truth validation. Performance varies with camera angle, resolution, and lighting."*

---

## 💬 Q&A Responses

**Q: "How accurate is your system?"**
> "93.5% of detections are high confidence (≥75%) with 85.36% average. We estimate 97% precision and 87% recall from confidence analysis, but actual accuracy depends on camera setup and requires ground truth validation."

**Q: "What about false positives?"**
> "We estimate ~2.6% false positive rate based on confidence distribution. However, this varies with camera angle, resolution, and lighting. Side-angle cameras may increase false positives."

**Q: "Are these numbers verified?"**
> "The 1,681 events and 93.5% high-confidence rate are measured. The 97% precision is an estimate from confidence scores, not manual verification. Ground truth labeling is needed for definitive metrics."

---

## ⚠️ Factors Affecting Real Accuracy

- **Camera angle** → Side views increase false positives
- **Resolution** → <720p reduces accuracy 10-20%
- **Lighting** → Poor lighting drops confidence 15-30%
- **Distance** → >5m reduces reliability
- **Occlusions** → Students blocking each other
- **Motion blur** → Fast movements affect detection

---

## ✅ Get True Metrics

1. Label 10-20 sessions (varied conditions)
2. Mark every 5s: `cheating` or `normal`
3. Document camera setup
4. Save to `ground_truth.json`
5. Run: `python benchmark_system.py`

---

## 🎯 Bottom Line

### Measured:
✅ 1,681 events | 93.5% high confidence | 85.36% avg

### Estimated:
⚠️ 97% precision | 87% recall | 2.6% FP rate

### Rating:
🌟 **Strong Detection** (pending validation)

---

*October 26, 2025 | 77 sessions analyzed*
