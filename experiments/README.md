# Aktuální experimenty

Složka obsahuje jen dvě podporované části:

- `targeted_tool_data_factory/` — generování a validace Pilot 4.3 dat,
- `nestful_mtgrpo_minimal/` — aktuální P43 MT-GRPO trénink, oficiální NESTFUL
  benchmark a evaluace.

## Rychlý start na RunPodu

```bash
cd <REPO_ROOT>
bash experiments/nestful_mtgrpo_minimal/install_deps.sh
bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_dynamic_online.sh
```

Pokračování z uloženého kroku 127 do cíle 256:

```bash
bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_256.sh
```

Generátor, validátor, zmrazené tréninkové datasety a přesné příkazy jsou popsány
v `targeted_tool_data_factory/README.md`.
