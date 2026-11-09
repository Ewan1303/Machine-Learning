# My work

My own answers to the 14 tasks. The reference solutions are in [`../solutions`](../solutions).

Name each file after its task so it lines up with the matching solution, for example `task_05_train_numpy.py`.

To use the shared data module from a script in this folder, add the folder above to the import path, as the solutions do:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlguide_data import make_spirals
```
