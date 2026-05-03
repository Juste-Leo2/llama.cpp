#pragma once

struct ggml_tensor;

struct llama_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer_shortconv {
    struct ggml_tensor * in_proj  = nullptr;
    struct ggml_tensor * conv     = nullptr;
    struct ggml_tensor * out_proj = nullptr;
};

struct llama_layer_nextn {
    struct ggml_tensor * eh_proj          = nullptr;
    struct ggml_tensor * embed_tokens     = nullptr;
    struct ggml_tensor * enorm            = nullptr;
    struct ggml_tensor * hnorm            = nullptr;
    struct ggml_tensor * shared_head_head = nullptr;
    struct ggml_tensor * shared_head_norm = nullptr;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_norm_b     = nullptr;
    struct ggml_tensor * attn_norm_2     = nullptr;
    struct ggml_tensor * attn_norm_2_b   = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_q_norm_b   = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_k_norm_b   = nullptr;
    struct ggml_tensor * attn_out_norm   = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm   = nullptr;
    struct ggml_tensor * attn_kv_a_norm  = nullptr;
    struct ggml_tensor * attn_sub_norm   = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_sub_norm    = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc   = nullptr;
    struct ggml_tensor * ssm_norm        = nullptr;
    struct ggml_tensor * ssm_dt_norm     = nullptr;
    struct ggml_tensor * ssm_b_norm      = nullptr;
    struct ggml_tensor * ssm_c_norm      = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wq_a      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkv_b     = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wqkv_b    = nullptr;
    struct ggml_tensor * wo_b      = nullptr;
    struct ggml_tensor * wq_cross  = nullptr;
    struct ggml_tensor * wk_cross  = nullptr;
    struct ggml_tensor * wv_cross  = nullptr;
    struct ggml_tensor * wo_cross  = nullptr;
    struct ggml_tensor * wq_enc    = nullptr;
    struct ggml_tensor * wk_enc    = nullptr;
    struct ggml_tensor * wv_enc    = nullptr;
    struct ggml_tensor * wo_enc    = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;

    // relative position bias
    struct ggml_tensor * attn_rel_b       = nullptr;
    struct ggml_tensor * attn_rel_b_enc   = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_norm_b       = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;
    struct ggml_tensor * ffn_post_norm_1  = nullptr; // gemma4
    struct ggml_tensor * ffn_post_norm_2  = nullptr; // gemma4
    struct ggml_tensor * ffn_pre_norm_2   = nullptr; // gemma4
    struct ggml_tensor * layer_out_norm   = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps    = nullptr;
    struct ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp      = nullptr;
    struct ggml_tensor * ffn_gate_inp_s    = nullptr; // gemma4
    struct ggml_tensor * ffn_gate_exps     = nullptr;
    struct ggml_tensor * ffn_down_exps     = nullptr;
    struct ggml_tensor * ffn_up_exps       = nullptr;
    struct ggml_tensor * ffn_gate_up_exps  = nullptr;
    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
    struct ggml_tensor * ffn_gate_exps_b   = nullptr;
    struct ggml_tensor * ffn_down_exps_b   = nullptr;
    struct ggml_tensor * ffn_up_exps_b     = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_b = nullptr;

    // ff MoE per-expert scales (NVFP4 per-tensor scale2)
    struct ggml_tensor * ffn_gate_exps_s   = nullptr;
    struct ggml_tensor * ffn_down_exps_s   = nullptr;
    struct ggml_tensor * ffn_up_exps_s     = nullptr;

    // ff MoE latent proj
    struct ggml_tensor * ffn_latent_down = nullptr;
    struct ggml_tensor * ffn_latent_up   = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    // ff adjugate experts (chexps)
    struct ggml_tensor * ffn_gate_chexps     = nullptr;
    struct ggml_tensor * ffn_down_chexps     = nullptr;
    struct ggml_tensor * ffn_up_chexps       = nullptr;

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3
    struct ggml_tensor * ffn_act    = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;

    // mamba proj
    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    // qwen3next
    struct ggml_tensor * ssm_beta_alpha = nullptr;

    // qwen3.5
    struct ggml_tensor * ssm_alpha = nullptr;

    // rwkv
    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    // rwkv7
    struct ggml_tensor * time_mix_w0         = nullptr;
    struct ggml_tensor * time_mix_a0         = nullptr;
    struct ggml_tensor * time_mix_a1         = nullptr;
    struct ggml_tensor * time_mix_a2         = nullptr;
    struct ggml_tensor * time_mix_v0         = nullptr;
    struct ggml_tensor * time_mix_v1         = nullptr;
    struct ggml_tensor * time_mix_v2         = nullptr;
    struct ggml_tensor * time_mix_g1         = nullptr;
    struct ggml_tensor * time_mix_g2         = nullptr;
    struct ggml_tensor * time_mix_k_k        = nullptr;
    struct ggml_tensor * time_mix_k_a        = nullptr;
    struct ggml_tensor * time_mix_r_k        = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct ggml_tensor * wq_s       = nullptr;
    struct ggml_tensor * wk_s       = nullptr;
    struct ggml_tensor * wv_s       = nullptr;
    struct ggml_tensor * wo_s       = nullptr;
    struct ggml_tensor * wqkv_s     = nullptr;
    struct ggml_tensor * wqkv_gate_s = nullptr;
    struct ggml_tensor * ffn_gate_s = nullptr;
    struct ggml_tensor * ffn_up_s   = nullptr;
    struct ggml_tensor * ffn_down_s = nullptr;
    struct ggml_tensor * ffn_gate_shexp_s = nullptr;
    struct ggml_tensor * ffn_up_shexp_s   = nullptr;
    struct ggml_tensor * ffn_down_shexp_s = nullptr;
    struct ggml_tensor * ssm_in_s    = nullptr;
    struct ggml_tensor * ssm_out_s   = nullptr;
    struct ggml_tensor * ssm_alpha_s = nullptr;
    struct ggml_tensor * ssm_beta_s  = nullptr;

    // input scales
    struct ggml_tensor * wq_in_s            = nullptr;
    struct ggml_tensor * wk_in_s            = nullptr;
    struct ggml_tensor * wv_in_s            = nullptr;
    struct ggml_tensor * wo_in_s            = nullptr;
    struct ggml_tensor * wqkv_in_s          = nullptr;
    struct ggml_tensor * wqkv_gate_in_s     = nullptr;
    struct ggml_tensor * ffn_gate_in_s      = nullptr;
    struct ggml_tensor * ffn_up_in_s        = nullptr;
    struct ggml_tensor * ffn_down_in_s      = nullptr;
    struct ggml_tensor * ffn_gate_exps_in_s = nullptr;
    struct ggml_tensor * ffn_down_exps_in_s = nullptr;
    struct ggml_tensor * ffn_up_exps_in_s   = nullptr;
    struct ggml_tensor * ffn_gate_shexp_in_s= nullptr;
    struct ggml_tensor * ffn_up_shexp_in_s  = nullptr;
    struct ggml_tensor * ffn_down_shexp_in_s= nullptr;
    struct ggml_tensor * ssm_in_in_s        = nullptr;
    struct ggml_tensor * ssm_out_in_s       = nullptr;
    struct ggml_tensor * ssm_alpha_in_s     = nullptr;
    struct ggml_tensor * ssm_beta_in_s      = nullptr;

    // altup & laurel
    struct ggml_tensor * per_layer_inp_gate   = nullptr;
    struct ggml_tensor * per_layer_proj       = nullptr;
    struct ggml_tensor * per_layer_post_norm  = nullptr;
    struct ggml_tensor * altup_correct_coef   = nullptr;
    struct ggml_tensor * altup_correct_scale  = nullptr;
    struct ggml_tensor * altup_predict_coef   = nullptr;
    struct ggml_tensor * altup_router         = nullptr;
    struct ggml_tensor * altup_router_norm    = nullptr;
    struct ggml_tensor * laurel_l             = nullptr;
    struct ggml_tensor * laurel_r             = nullptr;
    struct ggml_tensor * laurel_post_norm     = nullptr;

    // openai-moe
    struct ggml_tensor * attn_sinks = nullptr;

    // cogvlm
    struct ggml_tensor * visexp_attn_wqkv = nullptr;
    struct ggml_tensor * visexp_attn_wo   = nullptr;
    struct ggml_tensor * visexp_ffn_gate  = nullptr;
    struct ggml_tensor * visexp_ffn_down  = nullptr;
    struct ggml_tensor * visexp_ffn_up    = nullptr;

    // xIELU activation parameters for Apertus
    struct ggml_tensor * ffn_act_alpha_n = nullptr;
    struct ggml_tensor * ffn_act_alpha_p = nullptr;
    struct ggml_tensor * ffn_act_beta    = nullptr;
    struct ggml_tensor * ffn_act_eps     = nullptr;

    // Kimi Linear KDA (using ssm_ prefix for consistency)
    // Note: ssm_dt_b already exists above (mamba bias), reused for Kimi dt_bias
    struct ggml_tensor * ssm_q_conv = nullptr;
    struct ggml_tensor * ssm_k_conv = nullptr;
    struct ggml_tensor * ssm_v_conv = nullptr;
    struct ggml_tensor * ssm_f_a    = nullptr;
    struct ggml_tensor * ssm_f_b    = nullptr;
    struct ggml_tensor * ssm_beta   = nullptr;
    struct ggml_tensor * ssm_g_a    = nullptr;
    struct ggml_tensor * ssm_g_b    = nullptr;
    struct ggml_tensor * ssm_o_norm = nullptr;

    // DSA (deepseek sparse attention)
    struct ggml_tensor * indexer_k_norm   = nullptr;
    struct ggml_tensor * indexer_k_norm_b = nullptr;
    struct ggml_tensor * indexer_proj     = nullptr;
    struct ggml_tensor * indexer_attn_k   = nullptr;
    struct ggml_tensor * indexer_attn_q_b = nullptr; // note: for lora a/b, not bias

    // gemma4 layer output scale
    struct ggml_tensor * out_scale = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;

    struct llama_layer_shortconv shortconv;

    struct llama_layer_nextn nextn;
};
