# DAG / program diversity audit — Pilot3 train-300

- **topology_id**: n + dependency edges only (no tools/constants)
- **primitive_program_id**: topology + mapped primitive sids + arg ref/type pattern (no constants/surfaces)
- **surface_program_id**: tool names + param schemas + labels
- **semantic_program_family**: factory provenance field when present; else primitive_program_id

## topology_id

- unique families: **40** / 300
- top-1 share: **29.7%**
- top-10 share: **74.0%**
- Shannon entropy: **4.015 bits**
- mean tasks/family: **7.50**

| rank | id | n | share |
|---:|---|---:|---:|
| 1 | `topo_fdeb50f24873` | 89 | 29.7% |
| 2 | `topo_911586a55488` | 43 | 14.3% |
| 3 | `topo_bd3dd51ec1a9` | 21 | 7.0% |
| 4 | `topo_02b29308ce2b` | 16 | 5.3% |
| 5 | `topo_1c8dbe1830b2` | 15 | 5.0% |
| 6 | `topo_5296967693d8` | 10 | 3.3% |
| 7 | `topo_926c72bac680` | 7 | 2.3% |
| 8 | `topo_ff8ead63c33b` | 7 | 2.3% |
| 9 | `topo_0c51481d8aeb` | 7 | 2.3% |
| 10 | `topo_d228dd6db98a` | 7 | 2.3% |

## primitive_program_id

- unique families: **295** / 300
- top-1 share: **0.7%**
- top-10 share: **5.0%**
- Shannon entropy: **8.195 bits**
- mean tasks/family: **1.02**

| rank | id | n | share |
|---:|---|---:|---:|
| 1 | `prim_d45c19b58375` | 2 | 0.7% |
| 2 | `prim_72f085b6b3b5` | 2 | 0.7% |
| 3 | `prim_66dc5d6a8319` | 2 | 0.7% |
| 4 | `prim_8c06ea1494b7` | 2 | 0.7% |
| 5 | `prim_26be69ef3d7b` | 2 | 0.7% |
| 6 | `prim_80fb7a045b14` | 1 | 0.3% |
| 7 | `prim_2f7aee542af7` | 1 | 0.3% |
| 8 | `prim_168cbbc96c9d` | 1 | 0.3% |
| 9 | `prim_606d63f7dbca` | 1 | 0.3% |
| 10 | `prim_dc0e8ea424c8` | 1 | 0.3% |

## surface_program_id

- unique families: **298** / 300
- top-1 share: **0.7%**
- top-10 share: **4.0%**
- Shannon entropy: **8.215 bits**
- mean tasks/family: **1.01**

| rank | id | n | share |
|---:|---|---:|---:|
| 1 | `surf_45571be68722` | 2 | 0.7% |
| 2 | `surf_f9fe67b5274b` | 2 | 0.7% |
| 3 | `surf_2ff77ecbfaed` | 1 | 0.3% |
| 4 | `surf_dc002f0d3c50` | 1 | 0.3% |
| 5 | `surf_982976a06f33` | 1 | 0.3% |
| 6 | `surf_fe8879b984b5` | 1 | 0.3% |
| 7 | `surf_f3dd650dfba9` | 1 | 0.3% |
| 8 | `surf_24718c9f43a6` | 1 | 0.3% |
| 9 | `surf_58f7a56c53c1` | 1 | 0.3% |
| 10 | `surf_d949f340325c` | 1 | 0.3% |

## semantic_program_family

- unique families: **294** / 300
- top-1 share: **0.7%**
- top-10 share: **5.3%**
- Shannon entropy: **8.189 bits**
- mean tasks/family: **1.02**

| rank | id | n | share |
|---:|---|---:|---:|
| 1 | `pf_c218401edf38` | 2 | 0.7% |
| 2 | `pf_f2f1f9deb112` | 2 | 0.7% |
| 3 | `pf_e21c7ecec690` | 2 | 0.7% |
| 4 | `pf_5cf8fa30504f` | 2 | 0.7% |
| 5 | `pf_4f6a2062a24b` | 2 | 0.7% |
| 6 | `pf_b8b72ab43a11` | 2 | 0.7% |
| 7 | `pf_5240c3db5eff` | 1 | 0.3% |
| 8 | `pf_44698d2d849b` | 1 | 0.3% |
| 9 | `pf_b68927c61851` | 1 | 0.3% |
| 10 | `pf_b4e1f57c175e` | 1 | 0.3% |

## Diversity by call count / motif (unique topologies)

- by call: `{'2': 1, '3': 2, '4': 5, '5': 7, '6': 10, '7': 9, '8': 6}`
- by motif: `{'branch_aggregate': 6, 'fan_in': 27, 'linear': 7}`
