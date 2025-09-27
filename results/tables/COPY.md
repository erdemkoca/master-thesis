- write mail to volker: i can come on site. i am available the whole day, but i asusme 13.15 is fine for you too. not needed to have a link for meeeting for us


What i did:
- did include lassonet
- seed deleted, random generator now. seeds not set manually with number
  - inside the methods not seed setting
- statistial test
- tried to code nimo based on paper



Fragen:
- soll ich meine pipeline nun an neuen anderen datensätzen testen was denkst du? 



# Statistischer Vergleich der Modell-Performances

## Einleitung
Im Rahmen der Masterarbeit wurden vier Modelle zur binären Klassifikation kardiovaskulärer Risiken auf demselben Datensatz verglichen:
- **lasso** (Logistische Regression mit L1-Regularisierung)
- **randomforest** (Random Forest)
- **neuralnet** (Feed-Forward Neural Network)
- **lassonet** (LassoNetClassifierCV)

Für jede Iteration (0–19) wurde das Trainingsset durch Downsampling der Mehrheitsklasse balanciert, alle Modelle erhielten identische Trainings- und Testdaten. Anschliessend wurden die besten F1-Scores pro Modell je Iteration ermittelt.

Zur formalen Absicherung wurden folgende Schritte durchgeführt:
1. **Shapiro-Wilk-Test** auf Normalität der paarweisen F1-Differenzen
2. **Paarweise Tests** (gepaarter t-Test & Wilcoxon signed-rank Test)

---

## 1. Shapiro-Wilk-Test auf Normalität der F1-Differenzen
| Modell-Paar               | W-Statistik | p-Wert  |
|---------------------------|:-----------:|:-------:|
| neuralnet vs randomforest |    0.9343   | 0.1866  |
| neuralnet vs lasso        |    0.9455   | 0.3036  |
| neuralnet vs lassonet     |    0.9408   | 0.2482  |
| randomforest vs lasso     |    0.9760   | 0.8734  |
| randomforest vs lassonet  |    0.9607   | 0.5586  |
| lasso vs lassonet         |    0.9604   | 0.5527  |

**Interpretation:** Alle p-Werte liegen oberhalb α = 0.05. Die Nullhypothese „Differenzen normalverteilt“ kann nicht verworfen werden. Der gepaarte t-Test ist damit gerechtfertigt.

---

## 2. Paarweise Tests: t-Test & Wilcoxon signed-rank Test
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

## 3. Schlussfolgerungen
1. **Normalität gegeben:** F1-Differenzen sind annähernd normalverteilt.
2. **Beste Modelle:**  
   - **LassoNet** punktet deutlich gegen RandomForest und Lasso.  
   - **NeuralNet** schlägt RandomForest.
3. **Ähnliche Performance:**  
   - NeuralNet vs Lasso und vs LassoNet zeigen keine signifikanten Unterschiede.  
   - RandomForest vs Lasso ebenfalls nicht signifikant.

**Empfehlung:** Fokus auf LassoNet und NeuralNet; RandomForest liefert tendenziell schlechtere Ergebnisse. Klassisches Lasso bleibt interpretierbar, LassoNet vereint Interpretierbarkeit und Flexibilität.


---

Test für nimo:

Und neu hinzugekommener NIMO-Vergleich zeigt:

NeuralNet vs NIMO: p ≈ 3.4e-05 → NIMO übertrifft NN

RandomForest vs NIMO: p ≈ 1.9e-07 → NIMO übertrifft RF

lasso vs NIMO: p ≈ 4.3e-05 → NIMO übertrifft Lasso

lassonet vs NIMO: p ≈ 0.0218 → NIMO übertrifft LassoNet (gerade noch signifikant)


---
---
# NIMO
## Drin:

- Gemeinsames Netz mit Positional Encoding
- Profil-Likelihood-Loop (IRLS + Gradientenschritte)
- Logistische Modellierung 
- Grundlegende Gruppen-L2-Regularisierung

## Fehlt noch:

- Exakte Gruppen-L2 nur auf erste Schicht 
- Rausch-Injektion + tanh nach erstem Layer 
- sin-Aktivierung 
- Adaptive-Ridge ↔ Lasso (γ/c-Update)
- g_u(0)=0-Korrektur 
- Output-Clamping