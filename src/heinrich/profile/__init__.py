"""Model profiling: .frt (tokenizer), .shrt (weights), .sht (output).

The full digestive tract of a language model:
  frt  → the atoms the tokenizer produces
  shrt → the model's response to each atom
  sht  → the output distribution the user receives

Three stages. Three files. Three measurements.
The tokenizer farts out atoms. The model sharts on them.
What comes out is the shit.
"""
from __future__ import annotations
