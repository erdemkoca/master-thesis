Goal wiht Volker:
- Finished Pipeline (with all methods and plots). Showing to get feedback
  - with differnt synthetic and real dataset
  - show beta coefficient etc all plottings necessary
- Final Leitfragen if its okay
- How much time left for writing & cna i Send draft version of thesis
- Outline of thesis, show what you plan to write


TODO:
1. Pipeline mit allen methoden korrekt laufen
   - make whole nimo without Net, just adaptive ridge regression: NIMO without NN
   - pipeline with 2 synthetic scenarios and few real dataset scenarios
   - plot korretur of nimo how big it is
2. Leitfragen spezifizieren können
3. Masterarbeit schreiben können ahnhand von Methoden und Leitfragen


- plot korrekturen g(x_-j)
- do stuff volker said
- write on thesis, write my questions
- make working stage of pipeline
- auf accuracy umstellen von F1-Score
- make lasso better, since there we can have better results too. like group lasso etc all we can do there too
- investigate statistical rank testing
- clean all files: like methods give to much returns
- make different nimo implementation: with group, harder adaptive less adaptive etc
  - try to show different result on these
  - like make 5 nimo implementation
- guassian noise is implemented or not? in synthetic dataset
- should we also have NN and Random Forest in pipeline?
- parameters in lasso and nimo should be on the same spectrum scaled


- Könnte man Lasso auch noch verbessern? Eventuell mit mehr Sparsity, Group-Lasso etc?


Volker Roth Nachricht:
- Leitfragen gut so?
- Ich bin gerade meine Nimo am verbessern und habe noch zu tun.
- Ich versuche bis zum nächsten Montag meinen Code komplett zu finalisieren und wäre dann bereit alles zu vorstellen.
- Ich würde am nächsten Montag gerne die finalen Resultate anschauen und die Leitfragen besprechen,
  - ob diese so momentan in Ordnung wären.

- Weitere Fragen:
  - Auf welche realeDatensätze soll ich Lasso und Nimo auch anwenden um es näcshte woche zu zeigen?


TO Study and for writing:
- Profil Likelihood
- beta Koeffizienten interpretierbar -> Populatiobsebene
- Adaptive Ridge ==? Lasso
- F1 accuracy vs Accuracy
- soft–threshold on β, a prox step each IRLS round
- IRLS what is it?
- logit, log odd etc



To write in Thesis:
- Cross validation in Nimo
  - SE Rule
"🎯 What is the One-SE Rule?
The One-SE (One Standard Error) rule is a model selection technique that helps choose a simpler model when multiple models perform similarly well.
The Problem it Solves:
Cross-validation gives you the "best" C value based on highest CV score
But often, several C values perform statistically indistinguishable from the best
The One-SE rule picks the simplest model (highest regularization) within the "uncertainty band"
How it Works:
Find the C with highest CV score
Calculate the standard error of that score
Find all C values within 1 standard error of the best
Pick the smallest C (most regularized) from that band"



- pipeline how it works: all methods alogrithms wirtie
  - write the loops: outer inner loops how I implemented it. what it does

- the effect of scaling how faster it gets, like in lasso for instance
- Accuracy not good better choosing F1 since we are looking 



Friday Volker Presentation:
- Write synthetic formulas down so that he sees them.
  - synthetic dataset: centeered squares E[n]=b_0
  - how is noise?
- what i am saving? probabilities? logits? whar are the results? how is the data saved?
- in NIMO: binary_cross_entropy_with_logits
  - weshalb wie funktionierts?
- always trained on standardized features