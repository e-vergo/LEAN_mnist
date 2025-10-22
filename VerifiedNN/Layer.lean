/-
# Layer Module

Re-exports all layer components for convenient importing.

Instead of:
```lean
import VerifiedNN.Layer.Dense
import VerifiedNN.Layer.Composition
import VerifiedNN.Layer.Properties
```

You can use:
```lean
import VerifiedNN.Layer
```

This provides access to all layer definitions, composition utilities, and proven properties.
-/

import VerifiedNN.Layer.Dense
import VerifiedNN.Layer.Composition
import VerifiedNN.Layer.Properties
