# Codebase audit — pre-change snapshot, 2026-09-09

## Scope and method

Static source inventory parsed all 117 Python files (21 src modules, 96 scripts; 20,255 lines), including imports, functions, globals, entry points and sensitive extraction calls. Core modules and execution paths were inspected directly. No historical script was executed: many run network ingestion, training, or overwrite artifacts. This is a static correctness audit, not independent verification of historical experimental results or source licenses. Source hashes below identify exactly the audited versions. Configurations, README, dependency manifest, dataset versions, cached arrays and Git tracking were inspected.

## Architecture and actual data flow

The existing run_all.py sequentially runs materialize_audio, materialize_covers, materialize_lyrics, build_12class_dataset, create_final12_splits, final12_leakage_audit, generate_modality_matrix, extract_features, run_master_experiments. It operates on final_12class_metadata.csv (5,515 rows), old final12 split paths and shared data/features caches; finally trains and overwrites reports. It does not run the modular src trainer/model.

A second path, run_v3_master_benchmark.py, imports src models/trainer, reads data/splits/<scenario> and the same shared feature cache. Those split folders currently contain V4's 8,559 IDs whereas the cache contains V4-expanded's 8,057 rows. No dataset identity or cache provenance guard exists.

RMVMusicDataset is an all-zero placeholder with metadata masks, lyrics_dim=768 and unknown-genre fallback to POP_BALLAD. FeatureTensorDataset is the actual modular trainer dataset. PhysicalMultimodalDataset serves the old script pipeline. Multiple copies exist in imbalance runners.

Physical lyrics become dense 5,000-column TF-IDF. Covers use hand-crafted color/gradient descriptors. Audio implementations conflict: zero vectors with positive masks, ID-hash random vectors, or a WAV-only descriptor. Encoders project to 256 dimensions. Modular baselines include concatenation, logit average, and single-modality models. The src UAD uses sigmoid gates and weighted sum; the script UAD uses softplus uncertainty, finite missing penalties and weighted concatenation.

Training uses Adam/AdamW, train class weights and validation early stopping; routines differ in losses, seeds and hyperparameters. Evaluation uses sklearn metrics and ECE/Brier; some scripts generate hard-coded claims/weights instead of measured outputs. There is no automated regression suite.

## Observed data and environment

- V1 final: 5,515 unique IDs; V2/V3: 5,569; V4: 8,559; V4-expanded: 8,057. No metadata or report changed during audit.
- V3 SHA-256: b933b94bac5c9796bfb1deecea12745ce89510d2b85c069c0f74388a5fba6f4a. Twelve existing labels, 2,770 raw artist names, no empty artist or exact casefolded title+artist duplicate. 770 rows marked year_status=verified.
- Physical filesystem: 4,117 lyrics text files; 1,445 JPEGs; 147 audio files (121 WAV, 26 MP3). WAV headers decode: 120 mono int16 at 16 kHz and one stereo int16 at 44.1 kHz.
- V3 has zero nonempty audio_path entries. Therefore V3's linked audio coverage is 0%, NOT the whole workspace's physical audio count.
- Existing normalized artist-disjoint train/val overlap includes "min". Exact raw-string checks missed this.
- Python 3.12.10; torch 2.5.1+cu121; CUDA build 12.1 available. CPU NumPy/Torch bridge works. Core direct external imports: numpy, pandas, sklearn, scipy, torch, PIL, matplotlib, requests, tqdm. Neither torchvision nor yaml was imported by the old Python source; YAML files were effectively descriptive.
- requirements.txt contains only six packages with lower bounds. torch, Pillow, matplotlib are missing. PyYAML is needed once executable config loading is implemented.
- git ls-files tracks all 21 src files, 3 configs, README and requirements. No scripts tracked; .gitignore explicitly excludes scripts/ and test_*.py.
- README bytes decode correctly as UTF-8. The previous console mojibake diagnosis is not a demonstrated file-encoding defect.
- Protected snapshot (data, reports, outputs): 6,090 files, 1,068,022,247 bytes; combined path/content SHA-256 8e23c6bdc9132286b27473b78d79baafda5e67cc57dfb845375643391f6b7f92.

## Issues and proposed fixes

Each entry names the pre-change file/function and concrete remedy.

### CRITICAL

- C01 — src/data/feature_extractor.py::extract_acoustic_descriptor: generates random exponential/normal/uniform vectors seeded from song_id; never opens a waveform. Replace with physical waveform decoding; mask only successful, finite, non-silent extraction.
- C02 — scripts/train_baseline.py::AudioFeatureExtractor.extract and CoverFeatureExtractor.extract: URL/ID hash pseudo-features, with URL-based positive masks. Remove generators; fail explicitly on retired calls and route users to physical cache pipeline.
- C03 — src/training/trainer.py::FeatureTensorDataset.__init__ (also copies in scripts/run_imbalance_ablation.py and audit_and_evaluate_weights.py): missing map IDs filter features but not labels. Fail by default with missing IDs; optional explicit drop must filter the same dataframe and preserve song IDs.
- C04 — scripts/build_dataset_v4_audio_expanded.py::generate_vntm_audio_records / generate_open_acoustic_records: fabricates 2,990 numbered records and asserts verified audio/provenance without reading source recordings. Retire these generators; keep historical datasets, quarantine V4 from official training.
- C05 — src/data/feature_extractor.py::extract_all_v4_features and scripts/extract_features_v4.py::run_extraction: fit all lyrics / fallback to whole catalog when train split absent. Shared IID-fitted cache also leaks held-out vocabulary in other scenarios. Require train IDs and separate scenario-fitted TF-IDF caches; no fallback.
- C06 — scripts/run_v3_master_benchmark.py::load_features and scripts/run_all.py::run_master_pipeline: conflicting 8,559-row splits, 8,057-row cache and claimed V3 metadata; no lineage checking. Freeze V3 by hash, regenerate disjoint splits in a fresh official namespace; manifest binds metadata, train IDs, assets, labels and cache bytes.

### HIGH

- H01 — scripts/extract_features.py::run_feature_extraction audio branch: positive mask without extraction; scripts/extract_features_v4.py treats failed WAV decode as valid. Extract supported WAV files, reject invalid/silent/non-finite content; explicit missing reasons for unsupported MP3.
- H02 — src/models/uad_fusion.py::ReliabilityEstimator / UADFusionModel vs README and docs/phase9_method_formalization.md: sigmoid gate vs softplus/exponential; weighted sum vs historical concatenation. Adopt requested softplus/exponential masked weighted-sum formulation as a new version; expose uncertainty, reliability, weights and active masks; reject silently loading old checkpoints.
- H03 — src/data/splits/base.py::verify_zero_artist_leakage: compares raw strings; artist split lowers/strips but verification does not. Use a single Unicode/case/whitespace normalizer in both. Validate IDs, metadata consistency, coverage, class support and temporal verification.
- H04 — src/data/dataset.py::RMVMusicDataset: all-zero features with metadata-valid masks. Replace placeholder with validated physical cache adapter; fail if cache absent.
- H05 — .gitignore rules scripts/ and test_*.py; requirements.txt: clone omits entry points/tests; dependencies incomplete/unpinned. Allow source and tests to be tracked (without staging/commit); pin tested dependencies and capture a platform-specific dependency lock.
- H06 — scripts/run_master_experiments.py::run_master_experiment_suite / generate_all_publication_figures: hard-coded weights [0.58,0.35,0.07], duplicate ablation C/D with ignored lambda_inv, IID confusion reused for missingness. Retire from official execution, preserve old reports as unverified historical artifacts; new reports derive from saved predictions only.
- H07 — configs/proposed.yaml::genres differs from src label order; configs are not actually loaded by historical runners. Centralize canonical existing src mapping, reject config reorder rather than silently relabel.
- H08 — legacy scripts and feature_extractor import-time side effects create output dirs; run_all launches downloads/training with no mode. Official CLI defaults to validation and has explicit prepare/smoke/train/evaluate stages; preparation never invokes training or network.
- H09 — uncertainty architecture changes can load shape-compatible sigmoid checkpoints and silently reinterpret weights. Add persistent architecture version to new checkpoints; old checkpoints remain historical and require explicit migration/retraining.

### MEDIUM

- M01 — src/evaluation/metrics.py::compute_classification_metrics: macro averaging uses only labels appearing in a split, so the denominator changes across shifts; zero-support per-class entries omitted. Fixed 12-class macro metrics and full per-class table, supported-class balanced accuracy separately named; validate probability tensors.
- M02 — src/training/losses.py::DistributionInvarianceLoss: attracts unrelated labels just because availability differs; trainer uses original masks after dropout. Disable this undocumented cross-sample regularizer in official configuration; reject nonzero legacy weight pending a scientifically justified paired-view loss.
- M03 — class weights differ among runners; unknown labels map to 0. Freeze existing smoothed src formula for comparability; reject unknown labels globally; document formula and zero-support behavior.
- M04 — no cache shape/finite/mask/hash checks; TF-IDF may produce fewer than 5,000 columns on tiny corpus. Validate and zero-pad actual vocabulary to fixed dimension, preserve missing zero rows, return explicit status.
- M05 — no automated tests, no import-safe official CLI, no lock or hardware record, multi-seed order affects RNG. Add offline regression and forward-only smoke tests, deterministic seed setup and artifact manifests; document cross-platform limits.
- M06 — existing numerical/scientific claims, provenance and licenses not independently established; held-out duplicate media may overstate generalization. Measure overlap diagnostics; separate software correctness from evidence for publication.

### LOW

- L01 — several legacy report strings emit invalid-escape SyntaxWarnings on Python 3.12. Keep historical files except targeted fixes; do not execute them as validation.
- L02 — stale counts, paths, unqualified audio=0 and reproducibility/superiority claims in README and historical docs. Rewrite live README; link authoritative current docs and label old formulations historical without changing experimental outputs.

## Official choice and fix order

The retained V3 dataset (5,569 rows) is selected because reports/PHASE16_FINAL_DECISION.md explicitly ratifies it and configs/baseline.yaml and final_experiment.yaml point to it. This does not independently certify labels/licenses. Existing 12-class mapping stays unchanged. V4 and external acquisitions stay experimental; V1/V2 and old splits stay historical.

1. Record audit and protected checksums.
2. Add frozen official config, shared metadata validation, read-only validation and isolated artifact paths.
3. Replace physical extractors and cache loader; repair dataset invariants.
4. Correct UAD math, missingness, versioning and training/evaluation contracts.
5. Consolidate CLI, retire unsafe compatibility calls, ensure Git visibility and dependency coverage.
6. Add offline tests, real-data forward-only smoke and whole-dataset preparation/validation.
7. Update documentation only after verification. Do not train, download media, alter historical data/results, commit or push.

Planned edits: .gitignore, requirements.txt, README.md; src/data/{dataset,feature_extractor,splits/*}; src/models/{uad_fusion,baselines}; src/training/{trainer,losses}; src/evaluation/metrics; scripts/{run_all,extract_features,extract_features_v4,train_baseline,build_dataset_v4_audio_expanded} plus alignment copies if compatibility retained. Planned additions: configs/official.yaml; src/{config,pipeline}; src/data/cache; tests; script status registry/README; requirements-dev and lock; docs/DATA_PIPELINE.md, REPRODUCIBILITY.md, IMPLEMENTATION_STATUS.md and method documentation. Historical configs receive only status annotations, not relabeling.

## Complete pre-change Python inventory

ACTIVE refers to reusable src modules or the entry being consolidated, not a claim of correctness. All scripts outside explicitly documented official entry points remain LEGACY/EXPERIMENTAL and are not called by the replacement official pipeline.

- `src/__init__.py`: ACTIVE LIBRARY; 4 lines; SHA-256 `3e8eacbfb9ec4d5a43eb59c7cf1f2a340d597d8bcfdda5d3ebe590d3b6a855a3`.
- `src/data/__init__.py`: ACTIVE LIBRARY; 6 lines; SHA-256 `50de8c763d511e1331cff906eb2f9e3185addc6a8de43d807eb8d11afb2d6519`.
- `src/data/dataset.py`: ACTIVE LIBRARY; 66 lines; SHA-256 `1ac519f32c88494f359a15c1486bed038e96cbed9b2866b5e80c7c1c3ceb4efc`.
- `src/data/feature_extractor.py`: ACTIVE LIBRARY; 195 lines; SHA-256 `5569f7522bc35960599cf9842b0422152c0a035569d6cb2f8f51ff8b182fd9c2`.
- `src/data/splits/__init__.py`: ACTIVE LIBRARY; 19 lines; SHA-256 `48e1c992ba6f567290c7da1093df1d48b53a8ebd628f387a6967b6ffca542d88`.
- `src/data/splits/artist_disjoint.py`: ACTIVE LIBRARY; 43 lines; SHA-256 `99f4e3a1ba6d5422b5d5b13a407ae5151f41a6bb1cde399e8417596f342823aa`.
- `src/data/splits/base.py`: ACTIVE LIBRARY; 32 lines; SHA-256 `edb70138477b42dd3a29dc112addce2ed61fbaf0e1f349c0c09d3f9ae6b884ab`.
- `src/data/splits/builder.py`: ACTIVE LIBRARY; 68 lines; SHA-256 `2376a6d6ee25fb0a7464494d0d236d924a3bc23862a3edbecf9f1d6130f4f8ba`.
- `src/data/splits/iid.py`: ACTIVE LIBRARY; 19 lines; SHA-256 `81b94a24c3a1ace09091ab167373322f24728f180d12803ad4b3004f70d0cf04`.
- `src/data/splits/label_shift.py`: ACTIVE LIBRARY; 31 lines; SHA-256 `15ad1f5d184ddeb659541a59852c318dc8e4f55232fc9ebbcabf0ef92e718bb7`.
- `src/data/splits/missing_modality.py`: ACTIVE LIBRARY; 9 lines; SHA-256 `25577929f0f1e358db72bec57de0b43b76000bd647295c197d3f490fb6467613`.
- `src/data/splits/temporal.py`: ACTIVE LIBRARY; 26 lines; SHA-256 `d0e9f5515a521079e18da5ca191678695a1232344b5865b4320c80e9c4e2ef68`.
- `src/evaluation/__init__.py`: ACTIVE LIBRARY; 6 lines; SHA-256 `f9e18773c9df705289878015fb9ce80138f7419c77f58bcac976d98d3f225983`.
- `src/evaluation/metrics.py`: ACTIVE LIBRARY; 74 lines; SHA-256 `2fd7ff957fe7727c1bbf39881e125426415cd0c3573c0e0395f22e8eeb984ebd`.
- `src/models/__init__.py`: ACTIVE LIBRARY; 18 lines; SHA-256 `2508fb1328b702344f6fefc1c5d3daaf440bf8abcf6936df4a1e7f4b4dd483af`.
- `src/models/baselines.py`: ACTIVE LIBRARY; 94 lines; SHA-256 `123cf667a1fbbdac6127f9854e048be200520e766214f91eabb6243e62e92cd4`.
- `src/models/encoders.py`: ACTIVE LIBRARY; 47 lines; SHA-256 `e7dd5940e00a7a27bfd4de88cbebea533ed088263ecc42473d01fc468a4f0558`.
- `src/models/uad_fusion.py`: ACTIVE LIBRARY; 132 lines; SHA-256 `b91d5814023655105e7a5595d0b0f568cb956a2033bd0c273bea461ff961601f`.
- `src/training/__init__.py`: ACTIVE LIBRARY; 15 lines; SHA-256 `385d8e25109e50b087d761a60aef8f8ca13dc8ee18ea4e20a2300e854f35f005`.
- `src/training/losses.py`: ACTIVE LIBRARY; 84 lines; SHA-256 `3d3137e3d7d2279d7d9f56461520bbe2f35358e123d1ea98bf3bcaa4a0de379d`.
- `src/training/trainer.py`: ACTIVE LIBRARY; 169 lines; SHA-256 `2b43bb29039344b1d25a6816c7c4174a2e5c1bfca2bf2c1077317d170eba61d4`.
- `scripts/audit_and_evaluate_weights.py`: EXPERIMENTAL; 431 lines; SHA-256 `341517811f91f33ab84df4cb593fa11835e63e117a5eb2694f0a5db20ac58b87`.
- `scripts/audit_audio.py`: LEGACY; 147 lines; SHA-256 `eed262f4e82c27c32a7486fa50b482ed7c880dac31df5754c262c1c4337a00c6`.
- `scripts/audit_covers.py`: LEGACY; 127 lines; SHA-256 `fb2df685662b47cdde9e4834f17b80d1f2e1caaa6fce9a59452e89aca3aeb0e4`.
- `scripts/audit_imbalance.py`: EXPERIMENTAL; 150 lines; SHA-256 `9d39f60984f94b1f842d4376324aa272c7403b63c3e71e3fe5755711ecee83bf`.
- `scripts/audit_initial_data.py`: LEGACY; 126 lines; SHA-256 `d6e7f3d85adcb8912694199cddb0fc7061f57c0f50b9912ebe23419855fe9a92`.
- `scripts/audit_lyrics.py`: LEGACY; 173 lines; SHA-256 `6b1dd1bd7946293a018a36fb8d41d53b1e43535c98f4e34ee9537353b838114c`.
- `scripts/audit_physical_assets.py`: LEGACY; 281 lines; SHA-256 `5cae9dca5bd85719fded7a6a99c3c3612dd0a9fce86bbb4aa95ccddd85c44b68`.
- `scripts/audit_sources.py`: LEGACY; 219 lines; SHA-256 `058624432f889344898ac8ea263de744e95b70c25c65f2ca8b41bef63a98031b`.
- `scripts/build_12class_dataset.py`: LEGACY; 246 lines; SHA-256 `338d67224d3cf48b0da6272e68bd93ea366ed6aba9067aae4a0c5de9a2f9d909`.
- `scripts/build_dataset.py`: LEGACY; 499 lines; SHA-256 `b844a2d47ff2fbf7956075b01e8f9f167f8f45ae0de210974ca728f2a1e59ef6`.
- `scripts/build_dataset_v2.py`: LEGACY; 370 lines; SHA-256 `6760b68a0cb3bf134845348de286b870a68cb2dc09f115dda3a2bc6e47a57cab`.
- `scripts/build_dataset_v4_audio_expanded.py`: EXPERIMENTAL; 228 lines; SHA-256 `6aa4eced0dc906ba806fe155177033f22dd222f097f28ebf176e72434b3a798a`.
- `scripts/build_final_12class_dataset.py`: LEGACY; 238 lines; SHA-256 `1f3d9d6376e243a4457c25d9fdb70aafce395f4da81b21abf0a8a0af28b6a4ef`.
- `scripts/build_final_dataset.py`: LEGACY; 175 lines; SHA-256 `5e50af0c86cdd2dae6bca057432e312362a5c0dc9190219706538436e85bb96e`.
- `scripts/check_artist_leakage.py`: LEGACY; 98 lines; SHA-256 `edf85808a0e2754bca4899ba3d7ec98e08371b150fe15ba731361c7a4ebafebf`.
- `scripts/check_duplicates.py`: LEGACY; 69 lines; SHA-256 `1bca651034881b7ee4283b3f6eb530baa45c16cbf8025491780ffb8d04984ae5`.
- `scripts/collect_physical_audio.py`: LEGACY; 164 lines; SHA-256 `3ba048ec958075eb325a3c2bcaf4ea3af630a391b9de0eb73c7ffec57eafb148`.
- `scripts/collect_physical_covers.py`: LEGACY; 104 lines; SHA-256 `2a0f9c9f4d8f0c71a3f9998ea96666b33464092bfac6956e7684b20316edc31c`.
- `scripts/create_final12_splits.py`: LEGACY; 195 lines; SHA-256 `6344e52879a32bc09eac7bf7485084a726a3f492561e0e3df0f12e3b490d930b`.
- `scripts/create_final_12class_splits.py`: LEGACY; 213 lines; SHA-256 `0464a3216a84a97fbdd7d6237e3841c7f830ad5e97496382af7aca16d44456e7`.
- `scripts/create_final_splits.py`: LEGACY; 229 lines; SHA-256 `6c0703223fd12ec9fb7d9cdd6f953103f0ffdc59ad8062b4153dd8a248d57578`.
- `scripts/create_splits.py`: LEGACY; 207 lines; SHA-256 `d65f24b8f1ad91e3d14cad6734d4f7596d90d8cf44dca09cff799e743d853fc6`.
- `scripts/evaluate.py`: LEGACY; 139 lines; SHA-256 `7cb677c3be002d998026bcb5fa76bfa11c755e85c048ed5cd52b3f0e43e3fb72`.
- `scripts/evaluate_proposed.py`: LEGACY; 254 lines; SHA-256 `e39735c8f396a61664f5aca5735e7cfdf11627b199b89423de731619d298972c`.
- `scripts/expand_rare_genres.py`: LEGACY; 431 lines; SHA-256 `7b49bc72e07615d16eb4d9df66a37e50179d4f8151963bebb6882d3831150127`.
- `scripts/export_phase9_package.py`: LEGACY; 173 lines; SHA-256 `862bf80b6d587e186dc758928dc323bf03fc3ec542f08746491978e71905acfa`.
- `scripts/extract_features.py`: ACTIVE ENTRY (to consolidate); 174 lines; SHA-256 `b6946df4077a869a2fce89741b0d6379172c9a47ac3a51a68a37825584f66d8d`.
- `scripts/extract_features_v4.py`: EXPERIMENTAL; 198 lines; SHA-256 `dd5d718ade8915cbf137bbbec32a881e0ae707a5bffe6603ff82015de8e656a3`.
- `scripts/final12_leakage_audit.py`: LEGACY; 135 lines; SHA-256 `d3a0840fcf0ebf73210330e51796c2be893604620087fa80336a1bf9f986c56d`.
- `scripts/generate_modality_matrix.py`: LEGACY; 84 lines; SHA-256 `9ecdfdd06fb06149f917b5a3ad7cc9a0e04251d4b4b98e86e4c378a640bf5fd8`.
- `scripts/generate_modular_splits.py`: LEGACY; 45 lines; SHA-256 `e7e6c32a3962424397be9bfe55c655d3f2d12537aac35eeaebfcc3d59c263e43`.
- `scripts/generate_proposed_reports.py`: LEGACY; 241 lines; SHA-256 `5120bf1a310eae04232fcdf1c80865fe16457e7921cd5eb9ab01d13127fbf18e`.
- `scripts/generate_report.py`: LEGACY; 308 lines; SHA-256 `b4dcddedd416decd6d349331ee059060d05df6a95b45590c7037b1df83e78e58`.
- `scripts/ingest_external_datasets.py`: EXPERIMENTAL; 423 lines; SHA-256 `e4bf1e045989dcbda6a536ed9bbbec818409e35c9d0ffc9e519a64490dfa9650`.
- `scripts/ingest_phase2_expansion.py`: EXPERIMENTAL; 336 lines; SHA-256 `edd4a42ab4c75a81df091c1e14f2cd80f31058ba9337cc0e0a744bdb3316ed83`.
- `scripts/integrate_v4_expanded.py`: EXPERIMENTAL; 160 lines; SHA-256 `7192a6c2919931af4e496dbe0f5ae749a63cee9321f9d352e634e6bf64236842`.
- `scripts/integrate_v4_phase2.py`: EXPERIMENTAL; 144 lines; SHA-256 `ac7a58d281e92a6540efdd84e66b790fe8b3210c48bd5dd24ddc7cada12cde29`.
- `scripts/materialize_audio.py`: LEGACY; 203 lines; SHA-256 `32c50d3e7ad32c23f82f7aee05b48ea5c9246f0ac8702148ead2bd249b68af96`.
- `scripts/materialize_covers.py`: LEGACY; 237 lines; SHA-256 `db5ca31edf41e9f2298e4d1eaa9b76bdfe350873b73ac7f442cf9a49fbb87d53`.
- `scripts/materialize_lyrics.py`: LEGACY; 155 lines; SHA-256 `b1c680130c17ef598cb7e9d907e3ba58d63bf15de1515c72d49257d66482046e`.
- `scripts/phase11_final_dataset_audit.py`: EXPERIMENTAL; 78 lines; SHA-256 `f0d641455057df682e525368f8fa35c41449187d1dd113e5836943021f99b709`.
- `scripts/phase11_pre_expansion_audit.py`: EXPERIMENTAL; 62 lines; SHA-256 `f1ce9dcc70e3f9583eb6dd41913b71d6f9a4d6b564a9e929b8a7b4f13e3bd355`.
- `scripts/phase12_candidate_audit.py`: EXPERIMENTAL; 55 lines; SHA-256 `d98e67308065d74c935ccf8d7ff35eff481f6264ab01e026540111c1dee1fff5`.
- `scripts/phase12_external_discovery.py`: EXPERIMENTAL; 278 lines; SHA-256 `28db062cb799b19aea96d32b4543843c12d3163ec0ac1dce4bebe2652d28624e`.
- `scripts/phase12_source_inventory.py`: EXPERIMENTAL; 108 lines; SHA-256 `0f9da77c819a02edc033d218c0162e7b0bacd65e7207f5d97f4133eff97343c9`.
- `scripts/phase13_dataset_audit.py`: EXPERIMENTAL; 110 lines; SHA-256 `72be031f20e2ad2ba72218de8a54a9af5cc4775a8665400d48c39470d2e56785`.
- `scripts/phase13_dedup_audit.py`: EXPERIMENTAL; 73 lines; SHA-256 `e72c01165110cbb7a2c0e0d35d9b724c79227234bfe8522bda0413a1abcdb57a`.
- `scripts/phase13_external_discovery.py`: EXPERIMENTAL; 148 lines; SHA-256 `f859fe2c2dff8e5a76252128a14e8d59f68cabc60af96f41d4b246b197a5c40a`.
- `scripts/phase13_final_audit.py`: EXPERIMENTAL; 122 lines; SHA-256 `58196022416295c050ecf4e86ce3174426a50406e635cab3be1aadf6f4234805`.
- `scripts/phase13_license_audit.py`: EXPERIMENTAL; 88 lines; SHA-256 `99f2a2ad351d0a902843574ed6cd46dec2b2c290ccb377f87fe44b2ec6f86742`.
- `scripts/phase13_temporal_audit.py`: EXPERIMENTAL; 96 lines; SHA-256 `3ec612e5e48bf19759a2abdee1dfb8e03b7c368692c66ad2e77025a4912e2174`.
- `scripts/phase14_class_audit.py`: EXPERIMENTAL; 87 lines; SHA-256 `3374cfd0245fe0d391a9d186e78b62b4a12cc0688357c72c635dd8ae39df7cb7`.
- `scripts/phase14_dedup.py`: EXPERIMENTAL; 119 lines; SHA-256 `043c4b1105c6311cb165f83a1480cec9871639d3c14cfd569728830eba72557b`.
- `scripts/phase14_final_audit.py`: EXPERIMENTAL; 74 lines; SHA-256 `dc8de91b416c0c8c8c5effa9798a905ea6da806cf2004062b35911ee99ab595d`.
- `scripts/phase14_license_audit.py`: EXPERIMENTAL; 95 lines; SHA-256 `8276ae7e62847212ae37e216973c47fb72e941590a188e70ab489975bf28c4db`.
- `scripts/phase14_targeted_discovery.py`: EXPERIMENTAL; 178 lines; SHA-256 `8f16bbe618d82110e308078ea2eeeaf44db46a934a7cf75b362fed5e57a7e941`.
- `scripts/phase14_temporal_audit.py`: EXPERIMENTAL; 70 lines; SHA-256 `831d0354891db1809c977331ddab0453e3c79dcd2fa02e87f4f9034919d4103c`.
- `scripts/phase15_class_mapping.py`: EXPERIMENTAL; 117 lines; SHA-256 `906b2eeade644b8aa24358273de1dd70631b28cfe5ec5c9cebc79dd0b5ca2b67`.
- `scripts/phase15_dedup.py`: EXPERIMENTAL; 41 lines; SHA-256 `ca96e9b5782c985b997b6bfd58c1609bc350990c85c285792fd83400da499da0`.
- `scripts/phase15_deep_discovery.py`: EXPERIMENTAL; 171 lines; SHA-256 `3d6fdd1f79326718ccb0204b1ab013238b3db63b144c8c1169a6ae4bb6646950`.
- `scripts/phase15_final_audit.py`: EXPERIMENTAL; 74 lines; SHA-256 `e62db43ca7655d7fd5458e7049d93f4e2c171fc1138f0f6c514352fbf0f02e2c`.
- `scripts/phase15_license_audit.py`: EXPERIMENTAL; 78 lines; SHA-256 `c973fc172323419a9351089e7823d17708ce10c9ad98bbaa40fe9fe68d48bf87`.
- `scripts/phase15_provenance_audit.py`: EXPERIMENTAL; 83 lines; SHA-256 `6771013600fd4a5d38cb7236c5eab4afa058648de3b968eb36e3213d9431b9b5`.
- `scripts/phase15_temporal_audit.py`: EXPERIMENTAL; 70 lines; SHA-256 `2ddc851a55a5b7ed1e9a19c6096bc3583f8a7fd1de381d1118e33026f98e732f`.
- `scripts/phase16_class_mapping.py`: EXPERIMENTAL; 75 lines; SHA-256 `8ce22a3287a80621cce9b3fbd6966a1f34753bcd9482b7c5a390b79c45cc78a2`.
- `scripts/phase16_dedup.py`: EXPERIMENTAL; 63 lines; SHA-256 `f639b4571408c070d56fa75b9729a9683da635e4c1b61a3d2f3e5319cb7fb697`.
- `scripts/phase16_final_audit.py`: EXPERIMENTAL; 73 lines; SHA-256 `8e93307cf656978538cfd7283acfd15a4373bfaf74bec714cf74f6dadaa38896`.
- `scripts/phase16_independent_discovery.py`: EXPERIMENTAL; 256 lines; SHA-256 `6c29d1c687526278fcc89962534f351736ab7dbc90335c95e23bde2d71a520ed`.
- `scripts/phase16_license_audit.py`: EXPERIMENTAL; 89 lines; SHA-256 `f3bbbd74f10cd18cbb5900d23df190e34b80b8195badb6abf16c1922383a72aa`.
- `scripts/phase16_provenance_audit.py`: EXPERIMENTAL; 89 lines; SHA-256 `f9195af0d1aee016712b892e9e57b23b4b75123f9e48ba6dbe452b25d965b44d`.
- `scripts/phase16_source_audit.py`: EXPERIMENTAL; 85 lines; SHA-256 `9246cf267487680fad5fd59b26e048ce0e6a8034e260025b9ddd936be9e00e4e`.
- `scripts/phase16_temporal_audit.py`: EXPERIMENTAL; 64 lines; SHA-256 `31e730e2bfc37022520b4eeb1db4b38bfdc38a59f8294157edf145f4103da1ac`.
- `scripts/phase6_audit.py`: LEGACY; 190 lines; SHA-256 `8ecef30e047e2858184f8689a4643fc17442e179d26276fe80dc0ba299a53fb3`.
- `scripts/phase6_generate_reports.py`: LEGACY; 407 lines; SHA-256 `9db36a151cbd139dc7911d331ad1616e411aa8a04ab083c6cd62761e9fa757e8`.
- `scripts/phase6_plot_figures.py`: LEGACY; 257 lines; SHA-256 `2ee48ce28b823456fd69fe7214ed323de7fe8b27b95e014b401845971f7df0b3`.
- `scripts/phase6_stress_and_stats.py`: LEGACY; 579 lines; SHA-256 `608eff42eb29b1bc08a24b8de88b6effde004c6cf66c5b78478370653eb1272e`.
- `scripts/phase8_final_audit.py`: LEGACY; 216 lines; SHA-256 `1e68aed29cfb4d90dbd7f2e3cd289500cc7ee35a75b3019c0eef565d3b72c589`.
- `scripts/phase8_statistics.py`: LEGACY; 289 lines; SHA-256 `6e4d23d664609734c4e73c818c0768daa61813c88613666ea76065ada6ed543c`.
- `scripts/phase9_deep_auditor.py`: LEGACY; 418 lines; SHA-256 `2ab2b39436c9e49fbff226ccd4170d2fedfbe042447667b0080fe49df1f75738`.
- `scripts/reconcile_dataset.py`: LEGACY; 308 lines; SHA-256 `e5aa99ae0ee3bc57038fc96629fc3f96b71223c6c48e127ccb3b3976f055a7d7`.
- `scripts/recover_covers.py`: LEGACY; 145 lines; SHA-256 `5c452a090eb91d2ab2c33b80db4796f0229c4e090f7908ea782c64536a984691`.
- `scripts/recover_lyrics.py`: LEGACY; 82 lines; SHA-256 `892f5574f8658b0c5a3580d11b5d8656836c4d912ecba108b721b44c0395a5bd`.
- `scripts/run_all.py`: ACTIVE ENTRY (to consolidate); 66 lines; SHA-256 `fa0ed82b67d49d11ee90281afd65d9a8db5c8b0d8bae6c2015cd2ddac9dc4d93`.
- `scripts/run_all_baselines.py`: LEGACY; 316 lines; SHA-256 `52d527a5c69a354f14a6259b2bddf65b41fae841bdb802158dd803d4e4faf2a9`.
- `scripts/run_imbalance_ablation.py`: EXPERIMENTAL; 403 lines; SHA-256 `df77fe09d35f60ffe4fdd2c89a5bb839f572ae9dbc00270738a025b4e3114a6e`.
- `scripts/run_master_experiments.py`: LEGACY; 702 lines; SHA-256 `5fbeebde88635dcac9d374748406c1dc2c44d1c3bd3bb582130423d0cc57f884`.
- `scripts/run_proposed_experiments.py`: LEGACY; 488 lines; SHA-256 `2ffc036dff424bbcf58904050e6f35f72164391be36cd53bc01ea27f8ad1a705`.
- `scripts/run_v3_master_benchmark.py`: LEGACY; 178 lines; SHA-256 `2a23e5493e48aeaa5c4871b0e3e2498007e06c64de3de5addf976e27c50949df`.
- `scripts/run_v3_missing_modality.py`: LEGACY; 114 lines; SHA-256 `2fd7c0234c07034391dfbdefff40be4b1e9b19ce9b03e3446a51fc67c0644843`.
- `scripts/targeted_collection.py`: LEGACY; 189 lines; SHA-256 `5e4dccb4b2051e922ba7f1bdc5f6c4ea9e75ff8ac9161ac076e70f1f4727f9a0`.
- `scripts/train_baseline.py`: LEGACY; 323 lines; SHA-256 `9196f2ce3cace9371d065cc3af302db6b1602bdd0e3c901fa3460569a416c80c`.
- `scripts/train_physical_baselines.py`: LEGACY; 440 lines; SHA-256 `1aa03f2bb1a8430064c638fc007969adccd475ae5d73380e0550116d41c59ad7`.
- `scripts/train_proposed.py`: LEGACY; 253 lines; SHA-256 `351be2669ebca193c4197a3b1c623bd58bcdfd201be1134bba0f879b8ec9d9ae`.
- `scripts/validate_final_dataset.py`: LEGACY; 418 lines; SHA-256 `43f36c49f7a0aad80c6e3a0b286156bc09839c776c99904fce0f1ef3983de300`.
- `scripts/validate_phase7_dataset.py`: LEGACY; 316 lines; SHA-256 `ad1a724f0d6a89f6b0db94109abaa4d82d271137ca542b193eaa23d2992fba9c`.
- `scripts/validate_physical_assets.py`: LEGACY; 203 lines; SHA-256 `91ad5bdc0d93eb778269d22650ce0a44d31d8cb0d8862d9dc9c124c76a5543ec`.
