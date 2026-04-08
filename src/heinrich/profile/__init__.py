"""Model profiling: .frt (tokenizer), .shrt (weights), .sht (output), .trd (threads).

The full digestive tract of a language model:
  frt  → the atoms the tokenizer produces
  shrt → the model's response to each atom
  sht  → the output distribution the user receives
  trd  → the per-head threads that carry each atom through the layers

Four stages. Four files. Four measurements.
The tokenizer farts out atoms. The model sharts on them.
What comes out is the shit. The thread is how it got there.
"""
from __future__ import annotations
