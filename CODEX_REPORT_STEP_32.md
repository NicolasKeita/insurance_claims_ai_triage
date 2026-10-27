# STEP 32 - Document AI robustness and OCR evaluation

## Summary

Benchmark `document_ocr_v1` implemente et execute le 27 octobre 2026 sur CPU.
Le run final conserve dans `artifacts/evaluation/document_ocr_v1/verified/`
120 resultats : 15 NATIVE, 90 OCR et 15 HYBRID. Aucun checkpoint vision,
YOLO, CUDA, entrainement ou serveur Ollama n'est requis ou charge par ce chemin.

Resultats : 120/120 documents parses, 999/1000 champs exacts, 360/360 champs
critiques corrects. Une rotation de 2 degres fait perdre le chiffre final du
nom `Demo Garage 5`. Les 34 groupes degrades evaluables preservent leurs faits
de consistency ; un groupe supplementaire est exclu car un champ est incorrect.

Le point d'entree public est `evaluation.document_robustness` et le script est
`scripts/evaluate_document_robustness.py`. L'implementation est rangee sous
`src/evaluation/deterministic/document_robustness/` afin de conserver la limite
de huit sous-repertoires directs, sans restructuration du benchmark E2E.
La baseline mesuree est `configs/evaluation/document_ocr_v1_baseline.json`.

Inspection initiale : `extract_pdf_text` utilise pypdf ; `extract_pdf_text_auto`
conserve chaque page native et rend uniquement les pages sans texte avec
PyMuPDF avant `TesseractOcrEngine`/pytesseract. Les methodes existantes
`TextExtractionMethod.NATIVE/OCR/HYBRID` sont reutilisees et verifiees par cas.
Les vrais extracteurs `extract_claim_form`, `extract_accident_report` et
`extract_garage_quote`, leurs modeles Pydantic et `compare_claim_documents`
sont executes. Aucun second pipeline OCR ni parser reserve au benchmark.

La branche facultative `claims.llm_document_extraction`/StructuredLlm existe
toujours, distincte du chemin deterministe. Son extracteur de devis peut
retenter une fois lorsqu'un champ obligatoire manque. Elle n'est pas appelee
ici : ces chiffres ne mesurent pas OCR + LLM. Les enregistrements du benchmark
conservent les tentatives, les erreurs et le texte brut sans retry masque.
Les imports du backend deterministe sont devenus paresseux ; des controles
en processus neuf ont confirme l'absence de torch, ultralytics, mlflow et vision
lors de l'import de ce benchmark et de son preflight.

## Environment

Valeurs constatees dans la `.venv` locale :

| Composant | Valeur |
| --- | --- |
| Python | 3.12.14 |
| Tesseract | 5.5.3.20260724 |
| Executable | `C:\Program Files\Tesseract-OCR\tesseract.EXE` |
| Langue demandee | `eng` |
| Packs disponibles | `eng`, `fra`, `osd` |
| PyMuPDF | 1.28.2 |
| pytesseract | 0.3.13 |
| Pillow | 12.3.0 |
| numpy | 2.5.3 |
| pypdf | 6.19.0 |

Les documents generes ont des libelles anglais, donc le choix explicite est
`eng`. Aucun pack systeme n'a ete installe. Le preflight importe effectivement
les dependances, resout l'executable, interroge sa version et ses langues.
Les tests ont aussi bloque les imports de PyMuPDF et pypdf dans un processus
neuf : le CLI retourne 2 et `OCR benchmark: NOT EXECUTED`, avec la cause exacte,
sans traceback ni chiffres OCR fabriques. Les cas non executes sont exclus des
scores agreges ; leurs mesures textuelles et d'etapes restent nulles. Leurs
attentes de champs conservent zero correspondance et parse_success=false,
avec le statut not_executed explicite, sans compter comme echecs mesures.
Le chemin natif reste utilisable
sans pytesseract lorsque ses propres dependances sont disponibles.

## Benchmark corpus

Cinq dossiers `demo_01` a `demo_05`, trois documents chacun : claim form,
accident report et garage quote. Huit variantes par document, soit 120 cas.
Il s'agit exclusivement de donnees demo/synthetiques ; aucun dossier client
reel ou document personnel n'a ete ajoute.

La structure source reprend les donnees demo et le moteur de texte existant
`evaluation.deterministic.documents.document_texts`. Les attentes sont
construites directement comme modeles metier avant la generation des PDF,
jamais obtenues en reparsant les PDF. Elles varient les IDs, dates, lieux,
vehicules/annees, blessures, dommages, garages, devises EUR/USD/GBP, montants
et nombres de pieces. Les modeles actuels ne comportent pas de champ client
dans les extractions ; aucun score artificiel n'a ete ajoute pour ce champ.

`demo_03` ajoute HOOD au devis ; `demo_04` contient une date de rapport decalee
de huit jours. Les autres dossiers sont coherents selon les regles existantes.

| Variante | Transformation appliquee |
| --- | --- |
| native_clean | Texte natif, police Helvetica 12, page 595 x 842 points |
| scan_clean | Rasterisation 180 DPI, PDF image-only |
| scan_low_resolution | Rasterisation 100 DPI, PDF image-only |
| scan_jpeg_compressed | Rasterisation 180 DPI, JPEG quality 35 |
| scan_rotated_small | Rasterisation 180 DPI, rotation 2 degres, fond blanc |
| scan_blurred | Rasterisation 180 DPI, Gaussian blur radius 0.65 |
| scan_noisy | Rasterisation 180 DPI, bruit gaussien moyenne 0, ecart type 8 |
| mixed_native_and_scan | Source partagee entre une page native et une page image-only |

Le bruit utilise un seed explicite derive de SHA-256(document_id). Les memes
parametres produisent les memes octets PDF pour les huit variantes testees.
Le rendu OCR utilise 300 DPI et le timeout produit de 30 secondes par page.
Chaque resultat conserve sa configuration et son chemin PDF. Une inspection
visuelle du devis tourne a confirme un document lisible, sans texte tronque ;
le chiffre perdu par OCR est visible sur la source.

## OCR metrics

Normalisation limitee aux espaces, retours de ligne et espaces repetes.
Aucune correction des chiffres, dates, montants, casse ou ponctuation.
CER = distance de Levenshtein caractere / longueur de reference ; WER utilise
les mots. Les agregats somment erreurs et denominateurs. Vide/vide donne 0 ;
une insertion face a une reference vide donne un taux indefini avec erreurs
conservees. Le corpus effectif ne contient pas de reference vide.

Sur les 120 documents, CER = **0.00008245**, WER = **0.00029274**.
Sur les seuls 90 documents OCR image-only : CER = **0.00010994**,
WER = **0.00039032**. Les 15 HYBRID ont CER/WER = 0 sur leur texte complet ;
ils sont exclus des agregats OCR pur. Les 15 NATIVE ont CER/WER = 0.
Deux caracteres et un mot sont perdus dans un seul document.

Les CER/WER du tableau portent sur le texte complet de chaque variante ; les
lignes native et hybride ne sont donc pas des mesures OCR pur.

| Variant | CER | WER | Field exact match | Critical fields | Parse success |
| --- | ---: | ---: | ---: | ---: | ---: |
| native_clean | 0 | 0 | 100% | 100% | 100% |
| scan_clean | 0 | 0 | 100% | 100% | 100% |
| scan_low_resolution | 0 | 0 | 100% | 100% | 100% |
| scan_jpeg_compressed | 0 | 0 | 100% | 100% | 100% |
| scan_rotated_small | 0.00065963 | 0.00234192 | 99.2% | 100% | 100% |
| scan_blurred | 0 | 0 | 100% | 100% | 100% |
| scan_noisy | 0 | 0 | 100% | 100% | 100% |
| mixed_native_and_scan | 0 | 0 | 100% | 100% | 100% |

Chaque ligne represente 15 documents, 125 champs et 45 champs critiques.
Ces scores textuels sont des diagnostics, pas une mesure de decision assurance.

## Structured extraction

- Parse success : **120/120 (100%)** ; aucun echec de parsing.
- Field exact match : **999/1000 (99.9%)**.
- Required-field recall : **999/1000 (99.9%)**, valeurs correctes retrouvees.
- Required-field presence : **1000/1000 (100%)** ; une valeur presente est fausse.
- Unexpected value count : **1**, la valeur incorrecte du garage.
- Documents dont tous les champs sont exacts : **119/120**.

Tous les champs metier obligatoires des modeles actuels sont compares ; la
provenance `source` est exclue. Les montants Decimal sont compares numeriquement.
Les dommages sont des ensembles et les pieces sont comparees sans fuzzy
matching par description exacte, montant numerique et devise, avec multiplicite.
La collection `parts` compte pour un champ ; ses lignes ont aussi leur mesure
separee : **88 attendues, 88 extraites, 88 correspondantes, 0 manquante et
0 inattendue** sur les 40 devis. La main-d'oeuvre `labor` est mesuree
separement des pieces et participe a la validation financiere du produit.

## Critical fields

| Champ critique | Corrects / attendus | Accuracy |
| --- | ---: | ---: |
| Claim ID | 120 / 120 | 100% |
| Incident date | 80 / 80 | 100% |
| Repair total | 40 / 40 | 100% |
| Currency du total | 40 / 40 | 100% |
| Damage sets des formulaires/rapports | 80 / 80 | 100% |
| Total | 360 / 360 | 100% |

La correspondance des dommages du devis est egalement couverte par les lignes
de pieces et les faits HOOD de consistency ; le modele de devis n'a pas de
champ damage-types autonome.

Les tests de securite ont verifie la preservation exacte de 0 et 1 et le rejet
des montants ambigus contenant O/I, `1 200.00` ou `3,160`, plutot qu'une
transformation arbitraire. `1200.00` et `3160` restent numeriques exacts.
Une ligne de montant connue mais malformee ne peut plus disparaitre alors
qu'un total resterait coherent. Les valeurs contradictoires repetees pour un
champ et pour Labor/Total sont rejetees. Les doublons identiques restent permis.

Les dates invalides/OCR ambigu sont rejetees sans correction inventive. Une
substitution donnant une autre date valide ne peut pas etre reconnue a partir
du texte d'un seul document : la comparaison au ground truth et la consistency
interdocuments detectent ce cas dans les tests controles. Aucun cas de date
ou montant errone n'a ete observe dans les 120 scans du run mesure.

## By document type

| Type | Documents | CER complet | WER complet | Exact fields | Parse success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claim form | 40 | 0 | 0 | 360/360 (100%) | 40/40 |
| Accident report | 40 | 0 | 0 | 360/360 (100%) | 40/40 |
| Garage quote | 40 | 0.00031017 | 0.00099206 | 279/280 (99.6429%) | 40/40 |

L'erreur du devis concerne ici le nom du garage. Les chiffres des devis n'ont
pas montre de fragilite particuliere dans ce corpus ; aucune degradation de
leur precision numerique n'est deduite sans observation.

## Downstream consistency

`consistency_preservation_rate` = groupes degrades evaluables dont tous les
faits attendus sont preserves / groupes degrades evaluables. Une variante
est evaluable seulement si ses trois documents sont correctement extraits,
avec tous leurs champs exacts. Le chemin execute le vrai
`compare_claim_documents` retournant `ClaimConsistencyReport`.

Resultat : **34/34 = 100%**, parmi **35 groupes degrades** possibles.
Le groupe `demo_05 / scan_rotated_small` est exclu pour le champ garage errone.
Les cinq groupes natifs sont aussi evaluables et corrects, mais exclus du
denominateur de preservation des degradations. Les faits de dossier coherent,
de HOOD additionnel (`demo_03`) et de mismatch de date (`demo_04`) sont preserves
dans toutes leurs variantes evaluables. Ce taux conditionnel ne mesure pas
le comportement downstream apres des champs incorrects ou un parsing echoue.

## Failure analysis

Seul echec de champ observe : `demo_05_garage_quote / scan_rotated_small`.
Attendu : `garage = "Demo Garage 5"` ; extrait : `"Demo Garage"`.
Le texte OCR contient `Garage: Demo Garage` puis `Claim ID: DEMO-OCR-005`.
Le parseur aboutit, et les montants, devise, ID et pieces restent exacts.
Pour ce document seul : CER = 2/182 = 0.01098901 ; WER = 1/28 = 0.03571429.

Le PDF source et les parametres 180 DPI / 2 degres sont conserves dans
`verified/sources/demo_05_garage_quote__scan_rotated_small.pdf`, le texte dans
`verified/ocr_text/`, et les valeurs attendue/reelle ainsi que leur contexte
dans `verified/failures/demo_05_garage_quote__scan_rotated_small.json`.
L'artefact d'echec ne disparait pas lorsque le parsing produit reussit.

## Latency

Secondes moyennes par cas du run final ; elles ne constituent pas un
microbenchmark isole. Le temps OCR inclut ouverture, rendu et Tesseract.
Le total inclut aussi generation/degradation synthetique et calcul CER/WER.
Les etapes non mesurees apres une erreur restent nulles dans les artefacts.

| Variant | Native ingestion | OCR | Structured extraction | Total case |
| --- | ---: | ---: | ---: | ---: |
| native_clean | 0.0071 | 0 | 0.000246 | 0.0153 |
| scan_clean | 0.0254 | 1.1967 | 0.000271 | 1.3648 |
| scan_low_resolution | 0.0200 | 0.9859 | 0.000217 | 1.0606 |
| scan_jpeg_compressed | 0.0186 | 1.2530 | 0.000234 | 1.3119 |
| scan_rotated_small | 0.0229 | 1.1867 | 0.000261 | 1.6884 |
| scan_blurred | 0.0225 | 1.1807 | 0.000253 | 1.5287 |
| scan_noisy | 0.0238 | 2.9079 | 0.000351 | 3.8232 |
| mixed_native_and_scan | 0.0377 | 1.0533 | 0.000237 | 1.2332 |

Moyenne sur les 120 cas : total **1.5033 s**. Les traitements sont executes
successivement ; un run complet n'est pas obligatoire pour une mesure ciblee.

## Tests

Commande documentaire finale :

```powershell
.\.venv\Scripts\python.exe -u -m pytest tests/domain/claims tests/evaluation/document_robustness -m unit --ignore=tests/e2e --basetemp=.pytest-tmp-step32-documents-final -q
```

Resultat : **134 passed, 6 deselected en 29.72 s**. Couverture : ground truth,
PDF raster/rotation/bruit, image-only et HYBRID, CER/WER, Decimal/ensembles/
pieces, champs critiques, montants/dates, agregats, consistency, preflight
indisponible, reprise, protection contre l'ecrasement et comparaisons baseline.
Les huit variantes ont aussi ete regenerees et comparees octet par octet.

Regression unitaire generale : **1639 passed, 53 deselected, 5 warnings en
156.77 s**, avec la commande suivante executee hors sandbox :

```powershell
.\.venv\Scripts\python.exe -u -m pytest tests -m unit --ignore=tests/e2e --basetemp=.pytest-tmp-step32-unit-unsandboxed -q
```

Les premiers essais dans le sandbox ont ete interrompus sans resultat de suite :
le premier test API attendait `socket.accept` pendant la creation de la paire
de sockets locale de la boucle asyncio Windows, avant toute requete applicative.
La trace est conservee dans `artifacts/step32_pytest_hang.log`. L'execution hors
sandbox a resolu ce blocage sans changement du code. Les cinq warnings du run
termine concernent l'alias AnyIO de Starlette et quatre serialisations enum de
fixtures Pydantic ; aucun echec de test.

Tests Tesseract reels sur les sources finales : **6 passed en 9.04 s**, trois types x scan_clean/HYBRID,
avec `RUN_OCR_TESTS=1`. Sans opt-in : **6 skips explicites**. Le marker `ocr`
est enregistre ; les tests standards n'exigent pas le binaire systeme.
Le controle `nk-python` des 38 fichiers Python concernes a reussi sans warning
ni difference de formatage ; `git diff --check` a reussi.

La reprise du run complet avec comparaison a la baseline a termine avec code 0
et `regressions: []`, sans nouvelle extraction de cas. La baseline a ete creee
uniquement apres les 120 executions ; un repertoire existant ou une baseline
existante ne sont pas ecrases silencieusement. Les tests de regression simules
detectent baisse d'accuracy critique, nouveaux parse failures et baisse de
consistency, et refusent les populations/configurations incompatibles.

Les tests end-to-end n'ont pas ete executes faute de demande explicite.
Les scenarios E2E, leurs checkpoints et leurs seuils restent separes de ce run.

## Limitations

- Documents et degradations synthetiques, mises en page simples et police unique.
- Cinq dossiers et 15 documents sources ne representent pas tous les layouts assureur.
- Benchmark specifique a Tesseract et a la configuration locale `eng` constatee.
- Aucune evaluation d'ecriture manuscrite.
- Aucune affirmation de performance OCR generale en production.
- Aucun benchmark de la branche LLM facultative ni de documents libres complexes.
- Les resultats de consistency sont conditionnels aux trois extractions exactes.
- Une date valide mais fausse reste indetectable avec un seul document sans reference externe.
- Une latence mesuree pendant le developpement ne remplace pas un benchmark machine isole.

## Recommendations

Conserver le cas tourne `Demo Garage 5` comme exemple de regression et examiner
ce token lorsque l'on evalue un futur traitement de rotation. Son omission
montre que parse success et accuracy critique ne suffisent pas a certifier
tous les champs : continuer a presenter aussi l'exact match et la presence.
Ne pas ajouter de correction automatique du nom, des dates ou des montants sur
la base de ce cas. Les mesures ne montrent pas de probleme numerique appelant
une nouvelle heuristique de montant. Toute comparaison future doit conserver
les memes sources/configurations et utiliser une nouvelle version de benchmark
si scenarios, degradations, ground truth ou definitions de metriques changent.

## Reproduction

Depuis `C:\Users\nicol\Documents\insurance_claims_ai_triage`, les chemins de
sortie suivants doivent etre nouveaux ; sinon ajouter `--resume` pour les cas
enregistres avec la meme implementation/configuration. Les arguments de
selection peuvent etre repetes. Aucun GPU ni service applicatif n'est requis.

```powershell
# Preflight reel
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --preflight-only

# Commande du run final mesure et de sa baseline (creation exclusive)
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --output-dir artifacts/evaluation/document_ocr_v1/verified --write-baseline configs/evaluation/document_ocr_v1_baseline.json

# Reprise du run existant et verification contre la baseline versionnee
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --resume --output-dir artifacts/evaluation/document_ocr_v1/verified --baseline configs/evaluation/document_ocr_v1_baseline.json

# Nouveau run complet comparable, sans remplacer la baseline
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --output-dir artifacts/evaluation/document_ocr_v1/reproduction --baseline configs/evaluation/document_ocr_v1_baseline.json

# Un type de document
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --document-type garage_quote --output-dir artifacts/evaluation/document_ocr_v1/quotes_only

# Une degradation
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --variant scan_rotated_small --output-dir artifacts/evaluation/document_ocr_v1/rotated_only

# Un dossier
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --scenario demo_03 --output-dir artifacts/evaluation/document_ocr_v1/scenario_only

# Un seul document/une seule variante
.\.venv\Scripts\python.exe scripts/evaluate_document_robustness.py --scenario demo_01 --document-type claim_form --variant scan_clean --output-dir artifacts/evaluation/document_ocr_v1/one_case_only

# Tests OCR reels, puis retour au comportement standard
$env:RUN_OCR_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest tests/evaluation/document_robustness/test_real_ocr.py -m ocr --ignore=tests/e2e --basetemp=.pytest-tmp-step32-real-ocr-reproduction -q
Remove-Item Env:RUN_OCR_TESTS
```

La version applicative statique est dans `pyproject.toml` (Python/setuptools).
Les corrections documentaires validees ont incremente 0.1.37 vers 0.1.38 dans
le commit `da43034`. Le benchmark valide et son rapport font l'objet du PATCH
suivant, 0.1.38 vers 0.1.39. Aucun tag, release, push ou deploiement automatique.
