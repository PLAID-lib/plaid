---
title: Downloadable samples
---

# Downloadable samples

## First retrieval

Retrieving sample examples is as easy as:

```python
from plaid.downloadable_examples import AVAILABLE_EXAMPLES, samples

print(AVAILABLE_EXAMPLES)
print("samples.vki_ls59:", samples.vki_ls59)
```

The first call to `samples.vki_ls59` triggers a download and takes a few seconds.

## Cached retrieval

Subsequent calls are instantaneous because they reuse the cached sample.