"""adpipe — skin-tone-aware AD severity classification pipeline.

Reimplementation of the methods in:
"Reducing Skin-Tone Bias in AI-Based Atopic Dermatitis Diagnosis:
Class-Balanced Transfer Learning with Generative Data Augmentation".
"""

SEVERITY = ["mild", "moderate", "severe"]
FITZ_GROUPS = ["I-III", "IV-VI"]

__version__ = "1.0.0"
