# Implementation details

Everything you need to know if you need a deeper understanding than in [Tutorial](tutorial.md).

---

- [1. `Dataset` class](#1-dataset-class)
- [2. `Sample` class](#2-sample-class)
- [3. `DataCollection` class](#3-datacollection-class)
- [4. `BaseStorage` class](#4-basestorage-class)
- [5. Data/Feature classes](#5-datafeature-classes)
  - [5.1. `SpatialSupport` class](#51-spatialsupport-class)
  - [5.2. `TemporalSupport` class](#52-temporalsupport-class)
  - [5.3. `Quantity` class](#53-quantity-class)
  - [5.4. `Scalar` class ?](#54-scalar-class-)
  - [5.5. `Vector` class ?](#55-vector-class-)
  - [5.6. `Categorical` class ?](#56-categorical-class-)
  - [5.7. `ScalarField` class ?](#57-scalarfield-class-)
  - [5.8. `VectorField` class ?](#58-vectorfield-class-)
  - [5.9. `CategoricalField` class ?](#59-categoricalfield-class-)
  - [5.10. `ScalarTimeSeries` class ?](#510-scalartimeseries-class-)
  - [5.11. `VectorTimeSeries` class ?](#511-vectortimeseries-class-)
  - [5.12. `CategoricalTimeSeries` class ?](#512-categoricaltimeseries-class-)
  - [5.13. `ScalarFieldTimeSeries` class ?](#513-scalarfieldtimeseries-class-)
  - [5.14. `VectorFieldTimeSeries` class ?](#514-vectorfieldtimeseries-class-)
  - [5.15. `CategoricalFieldTimeSeries` class ?](#515-categoricalfieldtimeseries-class-)

---

## 1. `Dataset` class

**TODO**: trouver comment référencer la classe dans l’API générée automatiquement par autoapi

See [Dataset](../autoapi/plaid/dataset/index) ([download](../../src/plaid/dataset.py))

contains only 4 attributes:

- a list of [`Sample`s](#2-sample-class)
- and some metadata giving a view over data:
  - list of input names
  - list of output names
  - dict of split names to split ids

## 2. `Sample` class

**TODO**: trouver comment référencer la classe dans l’API générée automatiquement par autoapi

See [Sample](../autoapi/plaid/sample/index)  ([download](../../src/plaid/sample.py))

a Sample is only a view on a [`DataCollection`](#3-datacollection-class)

Besides the `DataCollection`, it has only one attribute that allows locate corresponding data in the `DataCollection`

## 3. `DataCollection` class

**TODO**: trouver comment référencer la classe dans l’API générée automatiquement par autoapi

See [DataCollection](../autoapi/plaid/data_collection/index)  ([download](../../src/plaid/data_collection.py))

it contains the data/features stored by type (called `feature_type` throughout the code) and names (called `feature_name`)

## 4. `BaseStorage` class

See [BaseStorage](../autoapi/plaid/base_storage/index)  ([download](../../src/plaid/base_storage.py))

## 5. Data/Feature classes

All those classes inherit from [`BaseStorage`](#4-basestorage-class)

### 5.1. `SpatialSupport` class

See [SpatialSupport](../autoapi/plaid/spatial_support/index)  ([download](../../src/plaid/spatial_support.py))

**TODO**: trouver comment référencer la classe dans l’API générée automatiquement par autoapi

### 5.2. `TemporalSupport` class

See [TemporalSupport](../autoapi/plaid/temporal_support/index)  ([download](../../src/plaid/temporal_support.py))

**TODO**: trouver comment référencer la classe dans l’API générée automatiquement par autoapi

### 5.3. `Quantity` class

See [Quantity](../autoapi/plaid/quantity/index)  ([download](../../src/plaid/quantity.py))

**TODO**: trouver comment référencer la classe dans l’API générée automatiquement par autoapi

### 5.4. `Scalar` class ?

See [Scalar](../autoapi/plaid/scalar/index)  ([download](../../src/plaid/scalar.py))

**TODO**: implique de faire toutes les classes qui suivent et de les maintenir, les tester... je pense vraiment qu’on peut tout mettre dans une seule classe `Quantity`

### 5.5. `Vector` class ?

Not implemented yet

### 5.6. `Categorical` class ?

Not implemented yet

### 5.7. `ScalarField` class ?

Not implemented yet

### 5.8. `VectorField` class ?

Not implemented yet

### 5.9. `CategoricalField` class ?

Not implemented yet

### 5.10. `ScalarTimeSeries` class ?

Not implemented yet

### 5.11. `VectorTimeSeries` class ?

Not implemented yet

### 5.12. `CategoricalTimeSeries` class ?

Not implemented yet

### 5.13. `ScalarFieldTimeSeries` class ?

Not implemented yet

### 5.14. `VectorFieldTimeSeries` class ?

Not implemented yet

### 5.15. `CategoricalFieldTimeSeries` class ?

Not implemented yet
