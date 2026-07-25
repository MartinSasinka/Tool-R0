# REWARD_AND_CREDIT_AUDIT — reward sémantika a credit assignment

Zdroje: `analysis/a07_counterfactual_rescore.json`, `analysis/a08_credit_audit.json`,
`lib/reward_ablation_registry.py`, `grpo_train._turn_returns`, `group_stats.py`.

## Counterfactual re-score (skutečný, na uložených eval trajektoriích)

Eval `_traj` obsahuje plné trajektorie → bylo možné přeskórovat **stejné
trajektorie** všemi intended politikami (500 úloh × 6 zdrojů C0/A0–A4).

- Průměrný reward stejných C0 trajektorií: A0-proxy 0.756, A1 0.710,
  A2/A3/A4 0.618. Gap `official_success` vs `executable_wrong_result`:
  A0 0.25, A1 0.43, A2–A4 0.77.
- **Závěr:** intended politiky by na identickém chování dávaly výrazně různé
  signály. Ablace byla smysluplně navržená — jen se (kvůli dispatch bugu)
  nikdy nespustila.
- **NENÍ kauzální** pro trénink: eval trajektorie jsou temp-0 single rollouty
  bez GRPO groups. Train-side counterfactual je
  `UNTESTABLE_WITH_CURRENT_LOGS` (rollout texty se při tréninku neukládají).

## Ověření reward sémantiky

- Pásma v3_2_dense i ablation skaláry respektují uspořádání
  success > partial > executable-wrong > failure; „loss překoná success"
  nenastává (a07: gap kladný ve všech politikách).
- Kratší validní cesty: oficiální scorer je akceptuje
  (`alternative_valid_solution_pass` u 254/500 C0 úloh) — reward na evalu
  nepenalizuje alternativní validní cestu.
- **Oprava metriky:** `SYNTHETIC_SUCCESS_REWARD` byl v offline auditu 0.92 a
  vydáván za „official success". Pásmo `fully_correct` v3_2_dense začíná na
  **0.90**; metrika přejmenována na *reward-threshold success proxy* a
  threshold opraven (`lib/offline_audit/__init__.py`). `episode_reward ≥
  threshold` NENÍ path-invariant terminal success — je to proxy; správně
  přeznačeno v ON_POLICY_METRICS výstupu.

## Credit assignment (G_t, advantage) — rekonstruováno z raw train logů

Mechanika: `G_t = f(turn_rewards[t:], episode_reward; γ, λ)` dle
`_turn_returns` (kopie funkce v audit `common.py` má testovanou paritu),
advantage = group-normalizace per turn pozice (`group_stats.py`); dead groups
detekované a přeskočené — `dead_flag_parity = true` ve všech armech
(stored == recomputed, 33–50/160 dead groups).

### Testy „podezřelého" credit assignmentu

| Metrika | A0–A4 rozsah | Interpretace |
|---|---|---|
| good-turn s negativní advantage | 27–33 % | **97–99 % z nich je v epizodách pod group mean** → očekávané chování outcome-based GRPO, ne bug |
| executable-wrong s pozitivní advantage | 26–29 % | **96–100 % z nich je v all-failure groups** → GRPO musí nejméně špatný rollout ohodnotit kladně; ne bug |
| bad-turn s pozitivní advantage | 4–7 % | nízké |
| podíl episode rewardu v G_0 | ~0.19 | process/episode se nedouble-countuje; episodický signál je v G_t započten jednou |

### Verdikt

Předchozí verdikt „CREDIT_ASSIGNMENT_SUSPECT" **není podložen**: všechny
naměřené jevy jsou definiční důsledky group-relativní normalizace při
vysokém podílu all-failure groups (50–55 %). Skutečný test (matched-prefix /
first-divergence: dostává správný prefix systematicky zápor kvůli pozdější
chybě?) je `UNTESTABLE_WITH_CURRENT_LOGS` — vyžaduje persistenci per-turn
parsed calls v rollout workeru (instrumentace specifikována v
NEXT_DECISION.md).
