"""Step 05 -- environment gates (once per session).

The three things that silently ruin a run: transformers outside the v5 window
(step 01), fp16 instead of bf16, and a QLoRA path that Unsloth advises against
for this model. bf16 is MANDATORY on this arch -- GDN layers give NaN grad norms
in fp16, and Unsloth lists qwen3_5 in FORCE_FLOAT32.

HAS_CUT_CROSS_ENTROPY should be True: with a 248K vocab it computes the loss
WITHOUT materialising [batch, seq, vocab] logits. HAS_FLASH_ATTENTION is NOT
required here: Qwen3.5 is a hybrid, only 8 of its 32 layers are full attention,
and Unsloth's own kernels are the fast path it documents for this model. A
"flash_attention_2 is not supported" line in the model load is expected -- move
on.

First steps are slow by design: the Mamba/GDN Triton kernels compile at load.
The probe discards those steps before timing.
"""
dapt_lib.verify_env(run)
