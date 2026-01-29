# Piste 3: Modification du Preprocessing - Analyse Détaillée

## Lignes Critiques Identifiées

**Fichier:** `src/plaid/storage/common/preprocessor.py`
**Lignes:** 559-560 (dans la fonction `preprocess()`)

```python
for split_name in split_flat_cst.keys():
    for path in var_features:
        if not path.endswith("_times") and path not in split_all_paths[split_name]:
            split_flat_cst[split_name][path + "_times"] = None  # ← LIGNE PROBLÉMATIQUE
        if path in split_flat_cst[split_name]:
            split_flat_cst[split_name].pop(path)  # pragma: no cover
```

## Explication du Code Actuel

Cette boucle traite chaque feature variable (`var_features`) pour chaque split:

1. **Ligne 559:** Si la feature n'est PAS une `_times` ET n'existe pas dans `split_all_paths`
2. **Ligne 560:** Ajouter `path_times=None` à `split_flat_cst`
3. **Ligne 561-562:** Si la feature de base existe dans flat_cst, la retirer (car elle est variable)

**Intention Originale:** S'assurer que toutes les features variables ont une entrée `_times` dans flat_cst, même pour les splits où elles n'apparaissent pas.

**Problème Créé:** Cela ajoute des `_times` orphelines (sans leur feature de base) dans `flat_cst`, ce qui cause le déséquilibre dans `_split_dict`.

## Changements Nécessaires pour Piste 3

### Option 3A: Ne pas ajouter de _times orphelines

**Modification:**
```python
for split_name in split_flat_cst.keys():
    for path in var_features:
        # CHANGEMENT: Ne pas ajouter _times orphelines
        # Commenté la ligne problématique:
        # if not path.endswith("_times") and path not in split_all_paths[split_name]:
        #     split_flat_cst[split_name][path + "_times"] = None

        if path in split_flat_cst[split_name]:
            split_flat_cst[split_name].pop(path)
```

**Impact:**
- ✅ Résout le problème pour WebDataset
- ⚠️ Change le comportement pour TOUS les backends
- ⚠️ Peut affecter zarr et hf_datasets si ils dépendent de ces _times orphelines
- ⚠️ Besoin de valider les 400 tests

### Option 3B: Ajouter aussi la feature de base

**Modification:**
```python
for split_name in split_flat_cst.keys():
    for path in var_features:
        if not path.endswith("_times") and path not in split_all_paths[split_name]:
            # CHANGEMENT: Ajouter AUSSI la feature de base, pas seulement _times
            split_flat_cst[split_name][path] = None
            split_flat_cst[split_name][path + "_times"] = None
        if path in split_flat_cst[split_name]:
            split_flat_cst[split_name].pop(path)
```

**Impact:**
- ✅ Crée des paires cohérentes feat/feat_times
- ⚠️ Ajoute des None artificiels dans flat_cst
- ⚠️ Change le comportement pour tous les backends
- ⚠️ Peut avoir effets de bord sur la reconstruction des samples

### Option 3C: Nettoyage post-preprocessing

**Modification:**
Ajouter une étape de nettoyage après `preprocess_splits()` mais avant le retour:

```python
# À la fin de preprocess(), après construction des schemas:
# Nettoyer les _times orphelines dans split_flat_cst
for split_name in split_flat_cst.keys():
    orphan_times = []
    for key in split_flat_cst[split_name].keys():
        if key.endswith("_times"):
            base_key = key[:-6]
            # Si la feature de base n'est ni dans flat_cst ni dans variable_schema
            if base_key not in split_flat_cst[split_name] and base_key not in variable_schema:
                orphan_times.append(key)

    for key in orphan_times:
        del split_flat_cst[split_name][key]
```

**Impact:**
- ✅ Nettoie après coup sans changer la logique principale
- ✅ Plus sûr pour les autres backends
- ⚠️ Ajoute complexité au preprocessing
- ⚠️ Peut masquer un problème de design plus profond

## Analyse des Risques

### Risques Globaux de la Piste 3
1. **Code Partagé:** `preprocessor.py` est utilisé par TOUS les backends
2. **Comportement Établi:** Ce code existe depuis longtemps, peut-être avec raison
3. **Tests Indirects:** Modification peut casser des tests non-évidents
4. **Maintenance:** Complexifie le code de preprocessing déjà complexe

### Tests à Valider si Piste 3 Implémentée
```bash
# Test complet pour détecter régressions
pytest tests/storage/test_storage.py -v

# Tests des autres backends
pytest tests/storage/test_storage.py::Test_Storage::test_hf_datasets -xvs
pytest tests/storage/test_storage.py::Test_Storage::test_zarr -xvs
pytest tests/storage/test_storage.py::Test_Storage::test_cgns -xvs

# Tests de conversion
pytest tests/bridges/test_huggingface_bridge.py -v
```

## Comparaison Piste 2 vs Piste 3

| Aspect | Piste 2 (Converter) | Piste 3 (Preprocessing) |
|--------|---------------------|-------------------------|
| **Localisation** | Converter.to_dict() | preprocess() |
| **Impact** | Webdataset uniquement | Tous les backends |
| **Risque** | Faible | Moyen-Élevé |
| **Complexité** | Simple (5-10 lignes) | Moyenne (15-25 lignes) |
| **Tests Required** | 2 tests | 400 tests |
| **Maintenance** | Isolé | Code critique partagé |

## Recommandation Finale

**NE PAS implémenter Piste 3** sans validation approfondie. Les risques sont trop élevés pour un problème qui peut être résolu avec Piste 2.

**Piste 2 est préférable car:**
- Modification localisée dans Converter
- Peut être conditionnelle au backend (if self.backend == "webdataset")
- Pas d'impact sur les autres backends
- Plus facile à reverter si problème
- Plus facile à maintenir

## Implémentation Alternative: Piste 2 Améliorée

Si nécessaire, voici une implémentation plus robuste de la Piste 2:

```python
# Dans Converter.to_dict(), après var_sample_dict = ...
# Et avant to_sample_dict(var_sample_dict, self.flat_cst, ...)

# Clean flat_cst for backends that don't store None features
if self.backend in ["webdataset", "zarr"]:
    clean_flat_cst = {}
    for key, val in self.flat_cst.items():
        if key.endswith("_times"):
            base_key = key[:-6]
            # Keep _times only if base feature is in variable_schema
            # OR if it's in flat_cst with non-None value
            if base_key in self.variable_schema or base_key in self.flat_cst:
                clean_flat_cst[key] = val
        else:
            clean_flat_cst[key] = val
    use_flat_cst = clean_flat_cst
else:
    use_flat_cst = self.flat_cst

return to_sample_dict(var_sample_dict, use_flat_cst, self.cgns_types, features)
```

## Conclusion

La Piste 3 nécessiterait des changements au cœur du preprocessing, avec des risques significatifs pour tous les backends. **La Piste 2 est fortement recommandée** comme solution pragmatique et sûre.
