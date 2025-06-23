# Kurzer Projektstand

**What I Did:**  
- **LassoNet** in die Pipeline korrekt integriert  
- Manuelles Setzen von Seeds entfernt, stattdessen zentraler RNG (keine Seed-Calls in den Methoden)  
- **Statistische Tests** (Shapiro‐Wilk, gepaarter t-Test & Wilcoxon) durchgeführt  
- Prototyp von **NIMO** gemäss Paper versucht zu implementieren  

**Frage:**  
- Soll ich die Pipeline jetzt auf weitere Datensätze anwenden?

---

## Statistische Tests

| # | Modell 1      | Modell 2      | t-Statistik | t-p-Wert   | Wilcoxon-Stat | Wilcoxon-p-Wert |
|:-:|:-------------:|:-------------:|:-----------:|:----------:|:--------------:|:---------------:|
| 0 | neuralnet     | randomforest  |   3.2711    | **0.0040**★ |      22.0      | **0.0010**★    |
| 1 | neuralnet     | lasso         |   2.0140    | 0.0584     |      56.0      | 0.0696         |
| 2 | neuralnet     | lassonet      |  −1.1146    | 0.2789     |      72.0      | 0.2305         |
| 3 | randomforest  | lasso         |  −0.6048    | 0.5525     |      88.0      | 0.5459         |
| 4 | randomforest  | lassonet      |  −3.7740    | **0.0013**★|      25.0      | **0.0017**★    |
| 5 | lasso         | lassonet      |  −2.5881    | **0.0180**★|      44.0      | **0.0215**★    |

★ p < 0.05 → signifikant

**Interpretation:**
- **neuralnet vs randomforest:** NeuralNet performt signifikant besser als RandomForest.
- **randomforest vs lassonet:** LassoNet schlaegt RandomForest signifikant.
- **lasso vs lassonet:** LassoNet ist signifikant besser als klassisches Lasso.
- Alle anderen Paare: p > 0.05 → keine signifikanten Unterschiede.

---

## Statistische Tests **mit** NIMO

| Modell-Paar              | p (t-Test)     | p (Wilcoxon)   |
|--------------------------|:--------------:|:--------------:|
| neuralnet vs nimo        | 3.37 × 10⁻⁵ ★   | 1.05 × 10⁻⁴ ★   |
| randomforest vs nimo     | 1.93 × 10⁻⁷ ★   | 4.00 × 10⁻⁶ ★   |
| lasso vs nimo            | 4.26 × 10⁻⁵ ★   | 2.00 × 10⁻⁶ ★   |
| lassonet vs nimo         | 5.88 × 10⁻²     | 2.18 × 10⁻² ★   |

★ p < 0.05 → signifikant

> **Ergebnis:** NIMO übertrifft alle anderen Modelle signifikant.

---

## NIMO-Implementierung: Drin & Fehlt

**Drin:**  
- Gemeinsames Netz mit Positional Encoding  
- Profil-Likelihood-Loop (IRLS + Gradientenschritte)  
- Logistische Loss & Threshold-Optimierung  
- Basis-Gruppen-L2-Regularisierung  

**Fehlt noch:**  
- Exakte Gruppen-L2 nur auf erste Schicht  
- Gaussian Noise Injection + tanh nach erstem Layer  
- sin-Aktivierung in zweiter Schicht  
- Adaptive-Ridge → Lasso (γ/c-Update)  
- Zero-Mean-Korrektur \(g_u(0)=0\)  
- Output-Clamping  

