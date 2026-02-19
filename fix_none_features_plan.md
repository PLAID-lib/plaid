# Plan de Travail: Résolution du Problème des Features None dans WebDataset

## Problème Identifié

**Symptôme:** `AssertionError: did you forget to specify the features arg?` dans `flat_dict_to_sample_dict`

**Cause Racine:**
Lorsque `flat_cst` contient `feature_times=None` sans la `feature` correspondante, le merge avec `var_sample_dict` crée un dictionnaire déséquilibré:
- `row_tim` contient la clé extraite de `feature_times`
- `row_val` ne contient pas cette clé
- Le `zip(row_tim.items(), row_val.items())` pair des clés différentes

**Exemple:**
```python
flat_dict = {
    'Global/global_1': [1,2,3],
    'Global/global_1_times': [0,0,-1],
    'Global/global_2_times': None,  # ← Orpheline
}
# Après _split_dict:
# row_val = {'Global/global_1'}
# row_tim = {'Global/global_1', 'Global/global_2'}  ← Déséquilibre!
```

## Pistes de Solution

### Piste 1: Filtrage dans bridge.py (to_var_sample_dict)
**Approche:** Ajouter automatiquement `feature=None` quand `feature_times` existe

**Avantages:**
- Fix localisé dans le bridge WebDataset
- Pas de changement aux autres backends

**Inconvénients:**
- Ajoute artificiellement des clés None
- Peut créer confusion sur ce qui existe réellement

**Implémentation:**
```python
if features is None:
    result = dict(wds_sample)
    # Pour chaque _times, s'assurer que la feature de base existe
    for key in list(result.keys()):
        if key.endswith("_times"):
            base_feat = key[:-6]
            if base_feat not in result:
                result[base_feat] = None
    return result
```

### Piste 2: Nettoyage dans Converter (reader.py)
**Approche:** Filtrer `flat_cst` avant le merge pour retirer les `_times` orphelines

**Avantages:**
- Nettoie les données avant utilisation
- Fix applicable à tous les backends

**Inconvénients:**
- Modifie le Converter (code partagé)
- Peut affecter autres backends si mal fait

**Implémentation:**
```python
# Dans Converter.to_dict(), avant l'appel à to_sample_dict:
clean_flat_cst = {}
for key, val in self.flat_cst.items():
    if key.endswith("_times"):
        base_key = key[:-6]
        # Only keep _times if base feature will be in var_sample_dict
        if base_key in self.variable_schema or val is not None:
            clean_flat_cst[key] = val
    else:
        clean_flat_cst[key] = val
```

### Piste 3: Modification du preprocessing
**Approche:** Empêcher la création de `_times` orphelines dans flat_cst dès le preprocessing

**Avantages:**
- Fix à la source du problème
- Données cohérentes dès la génération

**Inconvénients:**
- Modification du preprocessing (code complexe)
- Risque d'impact sur autres backends

**Implémentation:**
Modifier `preprocess_splits` dans `src/plaid/storage/common/preprocessor.py` pour exclure les `_times` des features constantes None du constant_schema.

### Piste 4: Relaxer l'assertion dans flat_dict_to_sample_dict
**Approche:** Rendre `flat_dict_to_sample_dict` plus tolérant aux clés manquantes

**Avantages:**
- Fix générique pour tous les backends
- Robustesse accrue du code

**Inconvénients:**
- Modifie comportement existant
- Peut masquer d'autres bugs

**Implémentation:**
```python
# Au lieu de zip strict, itérer sur row_tim et chercher dans row_val
for path_t, times_struc in row_tim.items():
    val = row_val.get(path_t, None)
    # Traiter val même si None...
```

## Recommandation

**Piste 1** est la plus simple et localisée. Elle résout le problème directement dans le bridge WebDataset sans affecter le reste du système.

## Implémentation de la Piste 1

Modifier `src/plaid/storage/webdataset/bridge.py` pour s'assurer que toutes les `_times` ont leur feature de base correspondante, même si None.

## Tests de Validation

Après fix, vérifier:
1. `pytest tests/storage/test_storage.py::Test_Storage::test_webdataset` passe
2. `pytest tests/storage/test_storage.py::Test_Storage::test_registry` passe toujours
3. Aucune régression sur autres backends
4. Pre-commit hooks passent
