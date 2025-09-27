TODO Zu hasue:
- 1. Prio NIMO lernen: alles!
- 2. Prio Synthetic Data generation lernen
- 3. Zusammenfassung beenden bei Plots
- erkläre in allen synthetic data weshalb viel features weshalb wenig linearity etc
- variant und baseline differences

TODO Aktuell:
- mache neu Neuralnet mit diesen sachen, mit notes updaten.
- paralle beide NN laufen lassen um unterschiedmerken ob es besser ist als alte.
- nach scenario update nochmals anschauen ob scenraien besser sind, ebssere outcome
- dann nimo besser verstehen
- interpretatio für alle scenarios machen

Frage:
- non linearity und rho sind auf alle features oder random auf gewisse? wie wird data generiert? wichtig zu wissen



# Roadmap
- different approaches of NIMO
- snythetic data more (how to create synthetic data), see how they did in nimo_Official)
  - test with different synthetic data
- learn about theory, about the steps, search for roadmap in AIchat: Logistic Regression & the Forward‑Pass
- difference in NIMO and MyVersion
- make some experiments with different implementations (include original nimo)
- find out why nimo_official went not good
- rename nimo_official or our nimo
- listen to meeting with assistants
- find out differences between nimoErdem and nimoOfficial, like one hot and not on-hot (for table search for: Offizielles Repo<br/>(AdaptiveRidgeLogisticRegression))
- volker and assitanat asked if i have one nn or multiple
  - since it makes sense because of time consumption
  - what did i do?
  - are both ways possible or not?
- make a summary of inmportant stuff volker gave till now:
  - like pair tsting, synthetc dataset, subsampling, normalization etc. maybe look into the voice recordings again
- Check the audio with volker, try to answer all questions
- for volker create a sketch / entwurf of the algorithmus, how we do masking etc
  - Vmap‑Trick for masking
  - verstehe im code adaptive ridge + sparsity
- Can I also make Lasso implementation better? additional steps to have full potential of lasso?


TOStudy:
- IRLS & Working Response lernen, Theory
- Adaptive Ridge Theory lernen
- in NIMO haben wir ja designmatrix.
  - Design‐Matrix genau die richtigen Interaktions‑ und Nichtlinearitäts­terme (oder gar die wahren Koeffizienten) vorliegen hätte
  - also erlernt man dort nichtlinearitäten und interaktionen zwischen den features?
  - NIMONet‑Variant (grün gepunktet) hält sich deutlich stabiler bei F1 ≈ 0.93–0.95. Es nutzt offenbar seinen sparsamen Maskierungs‑Prior
  - lernen marskierungs prior? was meint man damit


TOOD Improtant:
- look for synthetic data: how it is generated? find appropriate ways to do it
- understand the code of nimo
- implement multiple version of nimo

TODONEW:

Prio 1:
- learn different synthetic dataset what are they making, try to understand and explain why which makes sense
- understand nimo
- make snythetic data really step by step. first just rho, then just supports, than combination etc. to really see which scenraio for which case

Prio 2:
- chekc if baseline is the same like Official
- new version of nimos that makes sense

understand NIMO:
- Nonlinear structure: 
  - Standard sparse linear models (like Lasso) can only pick linear main effects. 
  - NIMONet injects a small neural “correction” network so that once features are masked in or out, 
  - it can still learn smooth nonlinear transformations of those selected inputs.
- Interactions:
  - Beyond just sinusoids or quadratics, you often want to capture feature–feature interactions (e.g. X₀·X₁, or more complex patterns). 
  - NIMONet uses a group‑sparse mask over groups of neurons, effectively learning which combinations (or higher‑order interactions) 
  - to include, while still enforcing overall sparsity for interpretability.



Synthetic Data:
- if there is multicollinearity we cannot distinguish between features coefficient well. So we cannot do sparsity right. selected features not good
  - there is a way for diagnosis: setting up regression model, take the features as dependent value and try to  predict the feature based on other features. thereby we dont need the feature anymore
  - it is overtermined
  - with tolerance and maybe VIF variance inflation vector better to select the features

TODO snythetic:
- Lasso under high correlation:
  - Signal‐to‐noise: If the true β’s are strong and noise is mild, lasso will reliably pick a handful of highly predictive features—
  - even if they’re somewhat collinear.
  - so we need to add more noise maybe then it will not be able to get it. but just rho ist not enough
- try to adapt senrario findo out which makes sense "lasso here high yes makes sense, nimo not why? we have to adapt nimo?"
- make matrix with true not true selected


Questions to ask to volker:
- can we use multicollinearity technique to have better sparsity? if the data is multicollinear?
- i wanted to plot with synthetic data each scenrarios where lasso is not performing good. but somehow it is almosty everytime performing good. Why?
- can we make nimo so good that it can also act like normal NN? for isntance in cases with nonlinear datasets where NN performs quite good
- "If y≈sin(X₀), then plotting X₀ vs. y shows a wave. Unless your model “knows” to include sin(X₀) or to curve‐fit, it will only try straight lines."
  - do we add these nonlinear cases in nimo? like these beginning where we have activation like tanh and sinus is it this?
- we just have RBF custom, does this make sense? should we include other customs? what makes sense?
  - can we even have nice performance by rbf with lasso or nimo, is it even possible?
  - should we not have with NN in rbf settings very good performance?
  - for instance applying this: 
    - "Lasso or plain logistic on raw X sees zero signal—they’d predict at chance. A kernel method (e.g. RBF‐kernel SVM) or a net that can construct those bump features can succeed."
- What to do for nimo to suceed in these situations:
  - **Correlated** features demand either grouping (elastic net) or more robust selection (stability selection) if you care about support.
  - **Interactions** need either manual feature crossing, tree methods (RF/GBM), or nets.
  - **Heavy‐tailed** noise calls for robust losses (Huber, quantile) or outlier‑resistant modeling.
  - **RBF‐like structure** calls for kernel machines or architectures able to learn localized features.
  
Main Question Volker Roth:
- are my scenarios good? if so i would focus now on Neural Net and NIMO to adapt and make it better?
- why do i have always same F1 for lasso?
  - should i always include statistical pair testing also later? or at the end?
- for IRLS step should we apply conjugate gradient descent


TODO General:
- NeuralNet kann prinzipiell Interaktionen lernen, ist hier aber zu schwach regularisiert und liefert nur mässige Leistung.
  - NN verbessern damit es besser performed 
  - handful of hidden units (and no explicit regularization toward those shapes) it only partially recovers the two nonlinear terms
  - Weil das Netz ohne explizite Periodizitäts‑Features auf rohe X-Werte trainiert, muss es erst Die Sinus‐Beziehung aus rohen Merkmalen lernen, 
  - was bei so hoher Frequenz schwierig ist und zu Instabilität führt.
  - Für Sägezahn:
    - piecewise‐ReLU/Maxout‐Layer im NN verwenden, um Kanten besser scharf darzustellen.
- Baseline NIMO, Prior problem eventuell?
  - NIMO‑Baseline ohne sparsity‐induzierenden Prior muss sich in dieser Aufgabe geschlagen geben.

- Nimo_Variant, oder lieber bei NimoNew dmait man einen Unterschied hat:
  - NIMO‑Baseline ohne sparsity‐induzierenden Prior muss sich in dieser Aufgabe geschlagen geben.
  - Architektur anpassen 
    - Erhöhe die Tiefe oder Breite deines Netzes (z. B. 3–4 Hidden‑Layer mit je 64–128 Units statt 2×32). 
    - Füge Dropout oder BatchNorm hinzu, damit es auch bei hoher Korrelation stabil lernt. 
  - Explizite Basis‑Erweiterung im NIMO‑Mask‑Layer 
    - Bau in deinem Mask‑ing‑Mechanismus nicht nur Raw‑X ein, sondern auch X² und X³ als separate Kanäle. 
    - Dann lernt NIMO direkt, welche dieser Polynome wichtig sind. 
  - Loss‑Term für Nichtlinearitäts‑Detektion 
    - Ergänze im NIMO‑Objective einen kleinen Penalty‑Term, der gezielt hohe Gewichte auf quadratische Kanäle fördert, wenn deren Aktivierung die F1 verbessert. 
  - Hyperparameter‑Suche 
    - Justiere Learning‑Rate, Regularisierung (L2), und sparsity‑Gewicht für das Mask‑Prior – manchmal hilft schon ein stärkerer Sparsity‑Term, damit NIMO sauber die wahren Polynom‑Kanäle isoliert. 
  - Feature‑Normalisierung 
    - Gerade bei x³ kann der Wertebereich stark abweichen – min–max oder Standard-Scaling auf jede Basisfunktion, bevor sie ins Netz gehen.
- für RBF, weiss nicht ob nötig ist. ( Szen. I)
  - Kernel‑Trick/NIMOnet‑Erweiterung: Ergänze im Mask‑Layer nicht nur Raw‑X, sondern auch solche Gauß‑Bumps. 
  - So lernt NIMO direkt, welche Center relevant sind, und könnte die konstant besseren Basis‑Features auswählen.
- Nimo Szen K 
  - erweitere dein Netz um multiplicative Layers.
- Nimo Szen L
  - Für NeuralNet und NIMO müsstest du entweder die Sparsity‑Prior stärker gewichten oder architektonisch Multiplikations‑ bzw. Sinus‑Netzwerke (Fourier‑Features) einbauen.



Presentation on Monday to Volker:
- Make one full everything results and one just nimo and neuralt net. or change the code that you can de-seleect some methods
- explain what i changed in nimo_official, which file i took and which stuff i changed (gpu etc)
- Main goal more exact feature selection higher F1
  - ich versuche NErual Net und Nimo beides parallel zu verbessern. Assitant stutzig gemacht gesagt NN immer gleich gut wie Nimo
  - haputunterschiede variant und baseline
- GOAL:
  - A: wie erwartet in Ordnung
  - B: Nimo variant should choose less features since it has the NN included
    - Kein Modell kommt hier auf exakt 5 ausgewählte Features (die beiden Interaktionsterme plus die 3 linearen Hauptterme). 
    - Das ist aber erwartbar, weil wir den Design‑Matrix nicht um explizite Produkt‑Spalten erweitert haben.
  - C: 
  - D:
  - E:
    - If you wanted to push this even harder, you could drop the degrees of freedom of the Student‑t even lower (say df=1 or 2), or throw in a larger variance Gaussian on top — 
    - that will create more catastrophic outliers and widen the spread in your NN’s F1 curve even further, while lasso stays rock‑solid.
    - Noise problem for NN and NIMO
    - **Lasso** bleibt stabil, weil es ohne grosse Anpassung an einzelne Ausreisser selektiert. 
    - **NeuralNet** schwankt stark, weil die Heavy‑Tails die Logits herumreissen und es keine explizite Robustheits‑Regularisierung gibt. 
    - **NIMONet** ist dazwischen: Es hält sich besser als das reine NN, aber kann den Ausreissern nicht völlig entkommen.
    - Fazit: „Ausreisser‑lastiger, heavy‑tailed Noise bricht unregulierte Hochkapazitäts‑Modelle, während sparsere Priors sie stabilisieren.“
  - G:
    - Dieses Szenario deckt genau die Schwäche linearer/Sparsity‑Methoden ab und zeigt den Nutzen von Modellen, die echte Interaktions‑Terms lernen oder approximieren können.
  - H:
    - Er demonstriert, dass ohne Feature‑Augmentation auch starke Regularisierung und maskierte Netze nicht perfekt den wahren quadratisch/kubisch‑getriebenen Support zurückfinden. 
    - Wenn du aber zeigen willst, dass NIMO gegenüber Lasso einen klaren Vorteil hat, müsstest du dem Netzwerk mehr Kapazität oder — noch besser — explizit polynomielle Basisfunktionen in den Input geben.
  - I:
    - Szenario I demonstriert sehr schön, dass du ohne die richtigen nicht‑linearen Basisfunktionen (hier RBF‑Glocken) an die Performance‑Grenze stosst.
  - J:
    - Dieses Szenario ist ein guter „Stress‑Test“ für Nicht‑lineare mit kniffligen Kanten
    - Zwar keine Methode komplett versagt, aber auch keine perfekt die harten Nicht‑Linearitäten meistert
  - K:
    - ohne explizite Design‑Matrix sind für lineare Methoden (Lasso) ein totales No‑Go, daher ihr konstantes, aber niedriges Niveau.
    - ohne explizite Interaktionsterms alle Verfahren – linear, NN und NIMO – deutliche Schwierigkeiten haben, 
    - und nur Lasso stabil ein mittleres Niveau erreicht, weil es einfach konstant das Korrelationsniveau abschöpft
  - M:
    - NIMO dort am stärksten punktet, wo **Interaktion + Nichtlinearität + Korrelation + Ausreißer** zusammenkommen – 
    - klassische Fälle, in denen weder reines Lasso noch ein undifferenziertes NeuralNet ideal sind.



To learn for volker:
- rho is the correlation hyperparameter

Steps:
1. get results for interpretability/sparsity
2. learn theory
3. see difference from Nimo official in code
4. make nice experiemnts test (differnt nimos, different datasets/synthetic)
5. find out why nimo_Official is not performing good
   - maybe because no gpu? maybe because some files missing the are in pipeline of nimo_official

# Notes:
- What i did for changes in nimo official:
   - deleted cuda stuff, gpu stuff 
   - the if None also changed

Stuff to learn:
- forward‐pass, the IRLS + adaptive‐ridge updates, the nonlinearity, the zero‐mean correction or the group‐Lasso proximal step 

# NIMO Aktuell

- In deiner aktuellen NIMO-Implementierung führst du im IRLS-Schritt zwar eine Ridge-Regression durch
   - das ist aber eine klassische, nicht-adaptive Ridge mit festem λ für alle Koeffizienten.
   - ich nutze eine einheitliche Ridge (𝜆*𝐼), also keine adaptive Gewichtung. 
   - Lösung: Um das einzubauen, müsstest du nach jedem IRLS-Schritt die w_j (zB 1/beta_j^gamma) neu berechnen und in Matrix A als diag (𝜆*w) statt (𝜆*𝐼 einsetzen)


# Questions
- do i need to have GPU or something to run NIMO?
- i have always 0.4 f1 score. waht do you think is it because of dataset, all emthdos aournd 0.4
- How is nimo differentiating between  logistic and continuous target variables?


# TODO Code
- Mini-Batches
- automate Hyparparameter selection
- Early-Stopping & Convergence-Criterion
  - find out if it makes sense to have an early stopping. make debugs and see the trend of loss, if at some point it converges or even gets worse
- Tanh/Sinus-Activations & Dropout:
  - Du hast schon Tanh() und SinAct(). Du könntest testen, ob es hilft, das Sinus-Layer vor oder nach der zweiten FC zu setzen, oder mit verschiedenen Skalen (z.B. sin(α x)) zu spielen. 
  - Ebenfalls könntest du experimentieren, ob ein zusätzliches Dropout in der ersten Schicht (statt nur einmal) Überanpassung weiter vermeidet.
- Adaptive-Ridge (Adaptive-Lasso) weiter verfeinern:
  - Im Paper empfehlen sie, das γ-Update (w = 1/|β|^γ) auch innerhalb des IRLS-Loops etwas anzupassen (z.B. γ ← γ·c oder mit kleinem Learning-Rate), um die Sparsity schärfer zu kontrollieren.
- Logging von Training und β-Werten:
  - Sammle nach jeder Iteration den Wert von ∥β∥₀ (Anzahl non-zero β) und ∥β∥₂, und plotte den Verlauf, um zu sehen, wie schnell deine adaptive-Lasso-Sparsity einsetzt.

# Future topcis to study

- Meerkat Statistics videos: logit/link-function,bernoulli, binomial, glm videos, NN vs Logistic Regresion  

# Future, reduce scenarios:

Du hast jetzt 13 sehr unterschiedliche Settings – von rein linearen (A) über Interaktionen (B, G, K), Nichtlinearitäten (C, F, J), Korrelation (H, D), RBF‑Features (I), Heavy‑tailed Noise (E, M) bis hin zu High‑Dim/Low‑N (L). Das deckt de facto den gesamten Spektrum ab.


## Was gut ist

- Jeder Haupt‑Effekt (Lineare vs. Interaktive vs. Nichtlineare vs. Kernel‑Manifold) ist vertreten. 
- Du testest robuste (Student‑t), schwere Ausreißer‑ und Standard‑Gaussian‑Rauschen. 
- Du hast Szenarien mit und ohne Feature‑Korrelation (ρ von 0 bis 0.95). 
- Der Extremfall n<p (“L”) greift typische Genom‑/Text‑Daten ab.

## Wo sich Dinge ähneln
- E vs. M: Beide heavy‑tailed Noise, M hat zusätzlich Interaktionen + Polynom. Du könntest E (nur heavy‑tailed) streichen, wenn dir M schon zeigt, dass Ausreißer zum Problem werden. 
- H vs. D: Beide korrelierte Polynom‑Terms + D hat noch eine Sinus‑Komponente und stärkere Korrelation + Interaktionen + heavy noise. D ist der superset, H also redundant? Wenn dir H nur einen monotoneren Einstieg erlaubt, könnte man es beibehalten – ansonsten reicht D. 
- C vs. F vs. J: Alle drei sind „reine“ Nichtlinearitäten. C kombiniert Sin/Square/Cube; F nur hochfrequente Sinus‑“Verstecker” (X selbst bleibt linear); J eine Sägezahn‑Funktion. Technisch deckt C schon alles ab (mehr Basisfunktionen), F und J sind also eher Edge‑Cases. Wenn es dir primär um die Herausforderung „lassoniveaus“ geht, könntest du F und/oder J rausnehmen.

## Empfehlung
- Core‑Gruppen beibehalten: A, B, C, D, G, I, L, M 
- Optional (je nach Fokus): E, H, F, J, K 
  - Wenn du Zeit/Plot‑Budget sparen willst, streiche E (nur Noise), H (nur Polynom+Corr), F (sin‑highfreq) und J (sawtooth) – C deckt den nichtlinearen Teil ab, D deckt Korrelation+Noise ab. 
  - Falls du jeden Effekt isoliert zeigen möchtest, behalte alle.

Kurz: Für eine schlanke Präsentation reichen etwa 8–9 Szenarien. Alles darüber hinaus ist zwar didaktisch interessant, wirkt aber schnell redundant.