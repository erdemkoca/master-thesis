Volker 11.Aug:
- Laaso not working good.
- Testing and Training should have same alpphas/betas
- udnerstand the points of code


 
- Write Introduction and methodology of master thesis
- Make the code much better. 
  - dont use chart lines, more these boxes
  - find out what the problem is with lasso F1
  - dont use F1, use accuracy
  - understand how synthetic data is generated, understand rho
  - understand every code that he could ask (based on previous and potential questions)
  - Statistical Pair testing needed?
  - Random forest volker said something
  - do the 2-feature illustration (for intuition)
  - more grouplasso sparsity how?

TODO:
1. plots verstehen
2. nimo verbessern 
3. 2-feature illustration
4. make hiher test set

Nimo verbessern:
NIMO sparsity verbessern (wenn Interpretierbarkeit Ziel ist):

staerker regularisieren: lam erhoehen; explizites L1/L0-Aequivalent auf beta pruefen (oder magnitude-penalty erhoehen), group_reg aktivieren/erhoehen.

Hard-Thresholding nach dem Training (z. B. |beta| < 1e-2 auf 0 setzen) + diesen Schwellenwert per Validation waehlen.

ggf. gamma in IRLS/Adaptive-Weights erhoehen (aggressiveres Shrinking).

TODO:
- Write down the formulas for synthetic datasets


- Ask what the roadmap could be? when and with wich working on real data?
- What should be the goals to achieve? since not that much time anymore




# GPT after mp4:
Hier ist eine kompakte, vollstaendige Liste der Punkte aus dem Gespraech: Tipps, To-dos und Professor-Bemerkungen.

# Kern-Probleme (Blocker – zuerst klaeren)

- [ ]  **Lasso liefert ueber Iterationen exakt gleiche F1-Werte** → statistisch unmoeglich. Vermutung: Split/CV/Thresholding oder Plotting fehlerhaft (z. B. immer gleiches Testset, gecachter Score, falscher Vektor geplottet).
- [ ]  **Diskrete F1-Werte (z. B. genau drei Stufen) bei Random Forest** → deutet auf fehlerhafte Randomness/Splitting hin (oder identische Seeds, zu kleines/konstantes Testset).
- [ ]  **Verwechslung OOB vs. Test-Eval** → OOB nicht parallel zum Hyperparameter-Tuning verwenden. Einheitlich: Hyperparameter per CV, finale Bewertung **nur** auf Testset.

# Daten & Szenarien (synthetische Generierung)

- [ ]  **Formel fuer jedes Szenario klar definieren und notieren.** Professor will eine exakte Zielfunktion yyy (oder Logit f(x)f(x)f(x)):
    
    Beispiel (Klassifikation): p(y=1∣x)=σ(β⊤x+∑mγm gm(xj1,xj2))p(y=1|x)=\sigma\big(\beta^\top x + \sum_m \gamma_m\, g_m(x_{j_1},x_{j_2})\big)p(y=1∣x)=σ(β⊤x+∑mγmgm(xj1,xj2)).
    
    Bedingungen fuer ggg: g(0)=0g(0)=0g(0)=0, Werte sinnvoll begrenzt (z. B. [−1,1][-1,1][−1,1]).
    
- [ ]  **Interaktion vs. Korrelation sauber trennen.**
    
    *Korrelation* betrifft XXX; *Interaktion/Nichtlinearitaet* betrifft yyy via gm(⋅)g_m(\cdot)gm(⋅).
    
    Szenario B aktuell unklar: war es nur Korrelation in XXX oder wirklich ein Term wie x1x2x_1 x_2x1x2 im **Ziel**?
    
- [ ]  **Nichtlinearitaetsstaerke erhoehen.** Sinus um 0 ist nahezu linear. Skaliere Argument oder Amplitude (z. B. g(x)=sin⁡(kx)g(x)=\sin(kx)g(x)=sin(kx) mit groesserem kkk / γ\gammaγ), sonst sieht Lasso „zu gut“ aus.
- [ ]  **Testset bei synthetischen Daten gross ziehen** (du kannst beliebig viele Samples generieren). 80/20-Split ist nicht noetig; ziehe pro Iteration neues grosses Testset.

# NEMO/NIMO (Modell & Sparsity)

- [ ]  **Sparsity-Quelle klaeren:** Professor: Sparsity haengt hier „nur“ an deiner Adaptive-Ridge-Penalty im linearen Teil. Hoehere Penalty → sparsamer. Penalty muss **modellselektiert** werden (wie λ\lambdaλ beim Lasso).
- [ ]  **Group-Lasso auch im linearen Teil pruefen**, nicht nur in nichtlinearen Termen/Shared-Net.
- [ ]  **NEMO kann bestimmte Formen nicht lernen**, z. B. sin⁡(∑jαjxj)\sin(\sum_j \alpha_j x_j)sin(∑jαjxj) als Ganzes. Baue die **synthetische Zielfunktion passend zur NEMO-Struktur** (Summe linearer Teile + Summe einfacher gmg_mgm auf einzelnen/kleinen Feature-Gruppen).
- [ ]  **Vergleiche/Ablationen**: Linear-only, NN-only, Linear+NN ohne Sparsity, +Sparsity; Penalty-Pfad (Sparsity vs. Performance) auswerten.

# Metriken, Schwellen & CV

- [ ]  **Zum Debuggen zunaechst Accuracy** (balancierte synthetische Sets) statt F1 verwenden.
- [ ]  **F1-Diskretisierung pruefen:** Ist die Schwellenwahl grob gesampelt? Nutze feines Grid oder direkte Kurven (precision_recall_curve) und optimiere F1 auf Val-Set.
- [ ]  **Ein sauberes Schema festziehen:** Train/Val/Test pro Iteration; Schwelle **nur** auf Val bestimmen; final **fix** auf Test anwenden.

# Randomness, Splits, Seeds

- [ ]  **Seeds fuer RF nicht fixieren** bzw. pro Iteration variieren; sicherstellen, dass Bootstrap/Splits wirklich neu sind.
- [ ]  **Pruefen, dass pro Iteration wirklich neue Daten/Splits verwendet werden** (keine stillen Ueberschreibungen/Caching, keine globalen States).
- [ ]  **Testset-Groesse hoch**, um Quantisierungsartefakte bei Metriken zu vermeiden.

# Plausibilitaets-Checks/Debug-Plots

- [ ]  **Minimalpipeline bauen:** Nur Lasso auf einem einfachen Szenario A. Schritt fuer Schritt verifizieren: neue Daten → neue Splits → neue Koeffizienten → andere Scores.
- [ ]  **Lasso-Koeffizienten ueber Iterationen als Boxplots** (und Anzahl selektierter Features) plotten. Sollten spuerbar streuen.
- [ ]  **2D-Fall plotten:** mit 2 Features die gelernte Entscheidungsflaeche/-gerade visualisieren; sollte je Split leicht variieren.
- [ ]  **Logging ausbauen:** fixe Ausgabe pro Iteration (Seed, Split-Indices/Hashes, Val-Schwelle, Hyperparameter, Score-Zerlegung).

# Visualisierung

- [ ]  **Keine Linien zwischen unabhängigen Iterationspunkten.** Stattdessen **Boxplots/Violinplots** ueber Iterationen; pro Methode ein Kasten.
- [ ]  **Klarere Legende und Achsenbeschriftungen** (was ist exakt geplottet: Test-F1, Val-F1, OOB?).

# NN & RF Baselines

- [ ]  **NN ueberarbeiten:** Regularisierung (Dropout, L2/Weight Decay), Lernrate/ES, einfache Hyperparameter-CV; ein kleines, vernuenftig regularisiertes Netz sollte bei linearen Faellen nicht „sehr schlecht“ sein.
- [ ]  **RF-Tuning nachvollziehbar:** n_estimators, max_depth, max_features, min_samples_* per CV; OOB nur fuer Schnelldiagnose, nicht fuer finale Vergleichbarkeit.

# Dokumentation (Professor-Tipp)

- [ ]  **Experiment-Tagebuch fuehren**: Direkt beim Implementieren jede Szenario-Formel XXX-Erzeugung **und** yyy-Erzeugung notieren (gern als Kommentar im Code + separate Notiz).
- [ ]  **Pseudocode fuer deine CV/Train/Eval-Schleifen** aufschreiben und in der Arbeit/Anhang erwaehnen. Spart spaeteres „Reverse Engineering“.

---

Wenn du moechtest, erstelle ich dir daraus eine kurze „Debug-Checkliste“ (Markdown mit Checkboxen) und ein Schema fuer die Szenario-Definitionen (Template mit Platzen fuer XXX-Erzeugung, yyy-Formel, Parameter). Soll ich das als Canvas anlegen?

Gefällt dir diese Persönlichkeit?