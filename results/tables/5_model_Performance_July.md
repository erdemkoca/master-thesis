# Roadmap
- different approaches of NIMO
- snythetic data more (how to create synthetic data)
- learn about theory, about the steps
- difference in NIMO and MyVersion
- make some experiments with different implementations (include original nimo)

# Notes:
- What i did for changes in nimo official:
   - deleted cuda stuff, gpu stuff 

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