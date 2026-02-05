# 🧬 Systematic Alpha Lab

Two independent alpha strategies with industry-grade backtesting  
and Barra-style performance attribution.

Focus: discover signals that survive out-of-sample and explain where returns come from.

---

## ⚙️ Strategies

### 🧬 Strategy A — Genetic Alpha Mining (Walk-Forward)
• Genetic Algorithm / GP factor search  
• rolling train → gap → holdout schedule  
• strict t-1 execution (no look-ahead)  
• fitness = IC t-stat (Newey–West adjusted)  
• complexity control for robustness  
• full walk-forward backtest  

• Barra risk attribution (post-trade)

→ automatically discovers statistically significant alphas
<img width="1641" height="615" alt="image" src="https://github.com/user-attachments/assets/5c0c1451-fc19-4068-9d1a-33ae18049e8b" />
---

### 📐 Strategy B — 101 Formulaic Alpha Strategy
• replication of classic 101 formulaic alphas  
• daily cross-sectional IC testing  
• factor surfacing & stability filtering  
• multi-factor portfolio construction  
• single-window production backtest  

• Barra risk attribution (post-trade)

→ interpretable, research-driven baseline factor strategy
![strategy_return](https://github.com/user-attachments/assets/b234465f-86c3-4d62-8a04-28aed3971306)

---

## 🧠 Philosophy

Out-of-sample first  
No leakage  
Statistical significance required  
Returns must be attributable, not accidental  

If you can’t explain the PnL, you don’t own the alpha.


<img width="1487" height="541" alt="image" src="https://github.com/user-attachments/assets/fbacc089-d0ae-4a7f-ae75-58fa7430c0a3" />


---

## 🛠 Stack

Python · NumPy · Pandas · Numba · scikit-learn  
Vectorized research pipeline · walk-forward backtester

---

## 📌 Applications

Alpha mining · factor research · portfolio construction · performance attribution

---

## 👤 Author

JP Peng  
Quantitative Research · Financial Mathematics




## Disclaimer
This repository is for research and educational purposes only.
Nothing here is financial advice, and performance in backtests does not guarantee future results.

UNAUTHORIZED USAGE OF THE CONTENTS IS PROHIBITED AND MAY RESULT IN LEGAL ACTIONS
