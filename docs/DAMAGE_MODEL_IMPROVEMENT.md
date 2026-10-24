# Améliorer Damage avec le dataset actuel

Les expériences réutilisent exactement les images, annotations et splits préparés
de la baseline CarDD HF. Aucun ajout d'image ni nouvelle préparation n'est requis.
La baseline `damage_segmentation_v1` et le checkpoint de l'application restent en
place. Chaque expérience et chaque diagnostic utilisent un nouveau répertoire.

## Préparer la session PowerShell

Depuis la racine du dépôt, reprendre le chemin réellement utilisé pour la baseline :

```powershell
$baseline = Get-Content .\artifacts\vision\damage_segmentation_v1\training_summary.json -Raw | ConvertFrom-Json
$env:DAMAGE_DATASET_YAML = $baseline.config.dataset_yaml
if (-not (Test-Path -LiteralPath $env:DAMAGE_DATASET_YAML -PathType Leaf)) {
    throw "Dataset préparé introuvable. Renseigner DAMAGE_DATASET_YAML avec son emplacement actuel, sans le régénérer."
}
$env:DAMAGE_CONFIDENCE_THRESHOLD = "0.25"
$env:DAMAGE_BASE_MODEL = $null
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`DAMAGE_BASE_MODEL` est vidé pour que le modèle choisi dans chaque TOML soit
respecté. CUDA doit être disponible pour `--device 0`. `--device cpu` reste
possible pour un diagnostic, mais sera beaucoup plus lent. Les commandes ne
modifient ni le dataset ni les paramètres d'inférence de l'application.

## Diagnostiquer avant de réentraîner

Vérifier d'abord les chemins, le checkpoint et la provenance, sans charger le modèle :

```powershell
.\.venv\Scripts\python.exe .\scripts\diagnose_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_v1.toml --output-dir .\artifacts\vision\damage_diagnostics\baseline_val640_v1 --device 0 --check-only
```

Puis mesurer sur **VAL**, à 640 et à 960 pixels, avec le même checkpoint existant :

```powershell
.\.venv\Scripts\python.exe .\scripts\diagnose_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_v1.toml --output-dir .\artifacts\vision\damage_diagnostics\baseline_val640_v1 --device 0 --operating-confidence 0.25
.\.venv\Scripts\python.exe .\scripts\diagnose_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_v1.toml --output-dir .\artifacts\vision\damage_diagnostics\baseline_val960_v1 --device 0 --imgsz 960 --batch 4 --operating-confidence 0.25
```

Le second diagnostic teste une résolution d'inférence plus élevée **sans
réentraînement**. Comparer le rappel et la précision des fissures, rayures et bosses,
les AP masque par classe et le temps de calcul. Une amélioration à l'inférence peut
suffire au besoin ; elle ne prouve pas qu'un entraînement à 960 sera meilleur.

Chaque diagnostic produit :

- `val_metrics.json` : contexte, provenance, métriques AP et seuil opérationnel ;
- `per_class_metrics.csv` : précision/rappel agrégés et AP boîtes/masques par classe ;
- `operating_point_metrics.csv` : TP, FP, FN, précision et rappel au seuil fixé,
  avec appariement séparé des boîtes et des masques à IoU 0,5 ;
- `ap/` : courbes précision-rappel et matrice de confusion Ultralytics ;
- `operating/` : passage VAL au seuil opérationnel, sans graphiques.

Le JSON enregistre aussi le batch effectif et `ap_speed_ms_per_image` (temps de
prétraitement, inférence et post-traitement du passage AP). `--batch 4` réduit la
pression mémoire du diagnostic à 960 pixels sans changer la configuration du run
d'entraînement existant.

Le calcul AP conserve les prédictions à partir de `conf=0.001`. Les P/R résumés par
Ultralytics utilisent son point de meilleur F1 ; les valeurs au seuil exact de 0,25
sont dans le rapport opérationnel séparé. La matrice Ultralytics apparie les
**boîtes**, même pour un modèle de segmentation ; elle ne mesure pas directement
les confusions de masques. Les sorties existantes ne sont pas écrasées : choisir
un nouveau `--output-dir` pour un nouveau diagnostic.

## Réentraîner seulement les essais utiles

Chaque ligne est une expérience indépendante, initialisée depuis les poids
préentraînés correspondants, avec 100 epochs maximum, patience 20 et seed 42.
Le batch réduit accompagne les essais plus coûteux pour la RTX 3070 Laptop 8 Gio ;
la mémoire et le temps réels restent à vérifier.

| Essai | Modèle | Image | Batch | Changement étudié | Répertoire de sortie |
|---|---|---:|---:|---|---|
| Résolution | YOLO26n-seg | 960 | 4 | Détails fins | `artifacts/vision/damage_segmentation_960_v2` |
| Capacité | YOLO26s-seg | 640 | 4 | Modèle plus grand | `artifacts/vision/damage_segmentation_small_v2` |
| Masques | YOLO26n-seg | 640 | 8 | `mask_ratio=2`, au lieu de 4 | `artifacts/vision/damage_segmentation_masks_v2` |

Lancer **un essai à la fois**, selon le diagnostic :

```powershell
# Premier candidat pour les petits dommages, après les diagnostics ci-dessus.
.\.venv\Scripts\python.exe .\scripts\train_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_960_v2.toml --device 0

# Alternative indépendante pour tester la capacité du modèle.
.\.venv\Scripts\python.exe .\scripts\train_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26s_seg_640_v2.toml --device 0

# Alternative indépendante pour tester les cibles de masque plus fines.
.\.venv\Scripts\python.exe .\scripts\train_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_masks_v2.toml --device 0
```

Les poids `yolo26s-seg.pt` peuvent être téléchargés par Ultralytics au premier
lancement explicite si absents localement. `mask_ratio` règle les masques cibles
d'entraînement ; il ne modifie pas directement la résolution des masques prédits.
Ne pas employer `--resume` sur la baseline terminée. Cette option sert uniquement
à reprendre le même nouvel essai interrompu, avec sa configuration inchangée.

Après un entraînement terminé, lancer le diagnostic du candidat, par exemple :

```powershell
.\.venv\Scripts\python.exe .\scripts\diagnose_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_960_v2.toml --output-dir .\artifacts\vision\damage_diagnostics\nano960_v2_val --device 0 --operating-confidence 0.25
```

Comparer aux diagnostics de baseline avec le même protocole. Regarder les trois
classes faibles et vérifier l'absence de régression sur les autres. Une résolution
ou un modèle plus grand n'est pas une garantie de gain. Si un essai paraît meilleur,
confirmer sa stabilité avec d'autres seeds dans de nouvelles configurations et de
nouveaux répertoires avant d'investir dans des changements supplémentaires.

Les réglages optionnels `mosaic`, `scale` et `close_mosaic` sont aussi disponibles
pour de futurs essais isolés. Ils ne sont pas modifiés dans les trois TOML afin de
garder les comparaisons lisibles. Sans valeur explicite, le comportement historique
d'Ultralytics est conservé.

## Réserver TEST au choix final

Choisir le modèle, sa résolution et le seuil opérationnel sur VAL. Ne pas lancer
TEST après chaque essai et ne pas adapter les paramètres à ses résultats.

Pour le candidat retenu uniquement, le nouvel évaluateur sépare aussi le plancher
AP du seuil opérationnel enregistré lors de l'entraînement :

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_damage_segmentation.py --config .\configs\vision\cardd_hf_yolo26n_seg_960_v2.toml --device 0 --ap-confidence 0.001
```

Adapter le TOML au candidat choisi. Le seuil opérationnel reste 0,25 dans cette
préparation et doit correspondre aux métadonnées d'entraînement. Le rapport note
le plancher AP séparément. Les anciens rapports TEST de v1 ont été calculés avec
`conf=0.25` : leur AP est tronquée et ne doit pas être comparée directement aux
nouvelles AP à 0,001. Ils sont conservés sans modification. La séparation VAL/TEST
interdit d'utiliser une nouvelle évaluation TEST pour décider du prochain essai.

Aucun changement de dataset ni remplacement automatique du modèle en application
n'est effectué par cette préparation. Les tests E2E nécessitent une demande
explicite et ne font pas partie de cette démarche.
