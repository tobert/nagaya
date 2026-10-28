//! The single-file checkpoint of SDXL, with Stability's tensor names (spec 11.9, "The served image model").
//!
//! The map gives the name of each tensor in the folder layout of diffusers, as the weights of the engine name them.
//! It is the map of diffusers 0.42.1 (`loaders/single_file_utils.py`): `convert_ldm_unet_checkpoint`,
//! `convert_ldm_vae_checkpoint`, `convert_ldm_clip_checkpoint` and `convert_open_clip_checkpoint`, for SDXL base.

/// How a tensor of the folder layout comes from a tensor of the single file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// The tensor as it is.
    Whole,
    /// `count` rows of the first dimension, from row `from`: a part of a fused query, key and value.
    Rows { from: usize, count: usize },
    /// The transpose of a matrix: the text projection of the second encoder.
    Transpose,
    /// The tensor without its trailing dimensions of size 1: a 1x1 convolution of the VAE attention as a linear.
    Squeeze,
}

/// The width of the second text encoder: the rows of each of its query, key and value.
const TEXT2_WIDTH: usize = 1280;

/// The layers of each down and up level of the UNet, after the first: 2 resnets, then an up or down sampler.
const LAYERS_PER_BLOCK: usize = 2;

/// The up levels of the VAE decoder. Stability counts them from the latent side in reverse.
const VAE_UP_BLOCKS: usize = 4;

fn unknown(key: &str) -> String {
    format!(
        "the single file has the tensor {key}, which the SDXL map does not know; use an SDXL checkpoint with \
         Stability's tensor names"
    )
}

/// The tensors of the folder layout, with their prefixes `te1.`, `te2.`, `unet.` and `vae.`, that the tensor `key` of
/// the single file gives. An empty list drops the tensor: no op reads it.
pub fn map_name(key: &str) -> Result<Vec<(String, Take)>, String> {
    let one = |name: String| Ok(vec![(name, Take::Whole)]);
    if let Some(rest) = key.strip_prefix("conditioner.embedders.0.transformer.") {
        if rest.ends_with("position_ids") {
            return Ok(Vec::new());
        }
        return one(format!("te1.{rest}"));
    }
    if let Some(rest) = key.strip_prefix("conditioner.embedders.1.model.") {
        return text2(rest).ok_or_else(|| unknown(key));
    }
    if let Some(rest) = key.strip_prefix("first_stage_model.") {
        return vae(rest)
            .map(|(name, take)| vec![(format!("vae.{name}"), take)])
            .ok_or_else(|| unknown(key));
    }
    if let Some(rest) = key.strip_prefix("model.diffusion_model.") {
        return unet(rest)
            .map(|name| vec![(format!("unet.{name}"), Take::Whole)])
            .ok_or_else(|| unknown(key));
    }
    Err(unknown(key))
}

/// The second text encoder (OpenCLIP), from `conditioner.embedders.1.model.`.
fn text2(rest: &str) -> Option<Vec<(String, Take)>> {
    let one = |name: &str, take| Some(vec![(format!("te2.{name}"), take)]);
    match rest {
        "logit_scale" => return Some(Vec::new()),
        "positional_embedding" => return one("text_model.embeddings.position_embedding.weight", Take::Whole),
        "token_embedding.weight" => return one("text_model.embeddings.token_embedding.weight", Take::Whole),
        "ln_final.weight" => return one("text_model.final_layer_norm.weight", Take::Whole),
        "ln_final.bias" => return one("text_model.final_layer_norm.bias", Take::Whole),
        "text_projection" => return one("text_projection.weight", Take::Transpose),
        _ => {}
    }
    let rest = rest.strip_prefix("transformer.resblocks.")?;
    let (layer, part) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let at = format!("te2.text_model.encoder.layers.{layer}.");
    if let Some(kind) = part.strip_prefix("attn.in_proj_") {
        let kind = match kind {
            "weight" => "weight",
            "bias" => "bias",
            _ => return None,
        };
        return Some(
            ["q", "k", "v"]
                .iter()
                .enumerate()
                .map(|(i, which)| {
                    let take = Take::Rows {
                        from: i * TEXT2_WIDTH,
                        count: TEXT2_WIDTH,
                    };
                    (format!("{at}self_attn.{which}_proj.{kind}"), take)
                })
                .collect(),
        );
    }
    let (module, param) = part.rsplit_once('.')?;
    let module = match module {
        "ln_1" => "layer_norm1",
        "ln_2" => "layer_norm2",
        "attn.out_proj" => "self_attn.out_proj",
        "mlp.c_fc" => "mlp.fc1",
        "mlp.c_proj" => "mlp.fc2",
        _ => return None,
    };
    matches!(param, "weight" | "bias").then(|| vec![(format!("{at}{module}.{param}"), Take::Whole)])
}

/// The name of a resnet parameter of the VAE: only the shortcut changes its name.
fn vae_resnet(param: &str) -> String {
    param.replace("nin_shortcut", "conv_shortcut")
}

/// The VAE, from `first_stage_model.`.
fn vae(rest: &str) -> Option<(String, Take)> {
    match rest {
        "quant_conv.weight" | "quant_conv.bias" | "post_quant_conv.weight" | "post_quant_conv.bias" => {
            return Some((rest.to_owned(), Take::Whole));
        }
        _ => {}
    }
    let (side, inner) = rest.split_once('.')?;
    if side != "encoder" && side != "decoder" {
        return None;
    }
    let name = |tail: String| Some((format!("{side}.{tail}"), Take::Whole));
    match inner {
        "conv_in.weight" | "conv_in.bias" | "conv_out.weight" | "conv_out.bias" => return name(inner.to_owned()),
        "norm_out.weight" => return name("conv_norm_out.weight".to_owned()),
        "norm_out.bias" => return name("conv_norm_out.bias".to_owned()),
        _ => {}
    }
    if let Some(mid) = inner.strip_prefix("mid.") {
        if let Some(param) = mid.strip_prefix("attn_1.") {
            let (module, kind) = param.split_once('.')?;
            let module = match module {
                "norm" => "group_norm",
                "q" => "to_q",
                "k" => "to_k",
                "v" => "to_v",
                "proj_out" => "to_out.0",
                _ => return None,
            };
            // The query, key, value and output weights are 1x1 convolutions here, and linears in diffusers.
            let take = if module != "group_norm" && kind == "weight" {
                Take::Squeeze
            } else {
                Take::Whole
            };
            return Some((format!("{side}.mid_block.attentions.0.{module}.{kind}"), take));
        }
        let (block, param) = mid.split_once('.')?;
        let index: usize = block.strip_prefix("block_")?.parse().ok()?;
        if !(1..=2).contains(&index) {
            return None;
        }
        return name(format!("mid_block.resnets.{}.{}", index - 1, vae_resnet(param)));
    }
    let (level, tail) = if side == "encoder" {
        inner.strip_prefix("down.")?.split_once('.')?
    } else {
        inner.strip_prefix("up.")?.split_once('.')?
    };
    let level: usize = level.parse().ok()?;
    let (blocks, level) = if side == "encoder" {
        ("down_blocks", level)
    } else {
        ("up_blocks", VAE_UP_BLOCKS.checked_sub(1 + level)?)
    };
    if let Some(conv) = tail
        .strip_prefix("downsample.conv.")
        .or_else(|| tail.strip_prefix("upsample.conv."))
    {
        let sampler = if side == "encoder" {
            "downsamplers"
        } else {
            "upsamplers"
        };
        return name(format!("{blocks}.{level}.{sampler}.0.conv.{conv}"));
    }
    let (resnet, param) = tail.strip_prefix("block.")?.split_once('.')?;
    let resnet: usize = resnet.parse().ok()?;
    name(format!("{blocks}.{level}.resnets.{resnet}.{}", vae_resnet(param)))
}

/// The name of a resnet parameter of the UNet.
fn unet_resnet(param: &str) -> Option<String> {
    let (module, kind) = param.rsplit_once('.')?;
    let module = match module {
        "in_layers.0" => "norm1",
        "in_layers.2" => "conv1",
        "out_layers.0" => "norm2",
        "out_layers.3" => "conv2",
        "emb_layers.1" => "time_emb_proj",
        "skip_connection" => "conv_shortcut",
        _ => return None,
    };
    Some(format!("{module}.{kind}"))
}

/// The UNet, from `model.diffusion_model.`.
fn unet(rest: &str) -> Option<String> {
    let fixed = [
        ("time_embed.0.", "time_embedding.linear_1."),
        ("time_embed.2.", "time_embedding.linear_2."),
        ("label_emb.0.0.", "add_embedding.linear_1."),
        ("label_emb.0.2.", "add_embedding.linear_2."),
        ("input_blocks.0.0.", "conv_in."),
        ("out.0.", "conv_norm_out."),
        ("out.2.", "conv_out."),
    ];
    for (old, new) in fixed {
        if let Some(kind) = rest.strip_prefix(old) {
            return matches!(kind, "weight" | "bias").then(|| format!("{new}{kind}"));
        }
    }
    let per_block = LAYERS_PER_BLOCK + 1;
    if let Some(tail) = rest.strip_prefix("input_blocks.") {
        let (i, tail) = tail.split_once('.')?;
        let i: usize = i.parse().ok()?;
        let i = i.checked_sub(1)?;
        let (block, layer) = (i / per_block, i % per_block);
        let (part, param) = tail.split_once('.')?;
        return match part {
            "0" => match param.strip_prefix("op.") {
                Some(kind) => Some(format!("down_blocks.{block}.downsamplers.0.conv.{kind}")),
                None => Some(format!("down_blocks.{block}.resnets.{layer}.{}", unet_resnet(param)?)),
            },
            "1" => Some(format!("down_blocks.{block}.attentions.{layer}.{param}")),
            _ => None,
        };
    }
    if let Some(tail) = rest.strip_prefix("middle_block.") {
        let (part, param) = tail.split_once('.')?;
        return match part {
            "0" => Some(format!("mid_block.resnets.0.{}", unet_resnet(param)?)),
            "1" => Some(format!("mid_block.attentions.0.{param}")),
            "2" => Some(format!("mid_block.resnets.1.{}", unet_resnet(param)?)),
            _ => None,
        };
    }
    let tail = rest.strip_prefix("output_blocks.")?;
    let (i, tail) = tail.split_once('.')?;
    let i: usize = i.parse().ok()?;
    let (block, layer) = (i / per_block, i % per_block);
    let (part, param) = tail.split_once('.')?;
    if let Some(kind) = param.strip_prefix("conv.")
        && (part == "1" || part == "2")
    {
        return Some(format!("up_blocks.{block}.upsamplers.0.conv.{kind}"));
    }
    match part {
        "0" => Some(format!("up_blocks.{block}.resnets.{layer}.{}", unet_resnet(param)?)),
        "1" => Some(format!("up_blocks.{block}.attentions.{layer}.{param}")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole(key: &str) -> String {
        let mapped = map_name(key).unwrap();
        assert_eq!(mapped.len(), 1, "{key}: {mapped:?}");
        assert_eq!(mapped[0].1, Take::Whole, "{key}");
        mapped[0].0.clone()
    }

    #[test]
    fn the_unet_names_follow_diffusers() {
        let cases = [
            ("time_embed.2.weight", "time_embedding.linear_2.weight"),
            ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
            ("input_blocks.0.0.weight", "conv_in.weight"),
            (
                "input_blocks.1.0.in_layers.2.weight",
                "down_blocks.0.resnets.0.conv1.weight",
            ),
            (
                "input_blocks.2.0.emb_layers.1.bias",
                "down_blocks.0.resnets.1.time_emb_proj.bias",
            ),
            ("input_blocks.3.0.op.weight", "down_blocks.0.downsamplers.0.conv.weight"),
            (
                "input_blocks.4.0.skip_connection.weight",
                "down_blocks.1.resnets.0.conv_shortcut.weight",
            ),
            (
                "input_blocks.5.1.transformer_blocks.1.attn2.to_k.weight",
                "down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k.weight",
            ),
            (
                "input_blocks.7.1.proj_in.bias",
                "down_blocks.2.attentions.0.proj_in.bias",
            ),
            ("middle_block.0.out_layers.3.weight", "mid_block.resnets.0.conv2.weight"),
            ("middle_block.1.norm.weight", "mid_block.attentions.0.norm.weight"),
            ("middle_block.2.out_layers.0.bias", "mid_block.resnets.1.norm2.bias"),
            (
                "output_blocks.0.0.in_layers.0.weight",
                "up_blocks.0.resnets.0.norm1.weight",
            ),
            (
                "output_blocks.2.1.proj_out.weight",
                "up_blocks.0.attentions.2.proj_out.weight",
            ),
            ("output_blocks.2.2.conv.bias", "up_blocks.0.upsamplers.0.conv.bias"),
            ("output_blocks.5.2.conv.weight", "up_blocks.1.upsamplers.0.conv.weight"),
            (
                "output_blocks.8.0.skip_connection.bias",
                "up_blocks.2.resnets.2.conv_shortcut.bias",
            ),
            ("out.0.weight", "conv_norm_out.weight"),
            ("out.2.bias", "conv_out.bias"),
        ];
        for (key, want) in cases {
            assert_eq!(
                whole(&format!("model.diffusion_model.{key}")),
                format!("unet.{want}"),
                "{key}"
            );
        }
    }

    #[test]
    fn the_vae_names_follow_diffusers_and_its_attention_drops_the_conv_dimensions() {
        let cases = [
            ("decoder.norm_out.weight", "decoder.conv_norm_out.weight"),
            (
                "decoder.up.3.block.0.conv1.weight",
                "decoder.up_blocks.0.resnets.0.conv1.weight",
            ),
            (
                "decoder.up.0.block.2.nin_shortcut.bias",
                "decoder.up_blocks.3.resnets.2.conv_shortcut.bias",
            ),
            (
                "decoder.up.2.upsample.conv.weight",
                "decoder.up_blocks.1.upsamplers.0.conv.weight",
            ),
            (
                "decoder.mid.block_2.norm1.bias",
                "decoder.mid_block.resnets.1.norm1.bias",
            ),
            (
                "decoder.mid.attn_1.norm.weight",
                "decoder.mid_block.attentions.0.group_norm.weight",
            ),
            ("decoder.mid.attn_1.q.bias", "decoder.mid_block.attentions.0.to_q.bias"),
            (
                "encoder.down.1.downsample.conv.bias",
                "encoder.down_blocks.1.downsamplers.0.conv.bias",
            ),
            (
                "encoder.down.0.block.1.conv2.weight",
                "encoder.down_blocks.0.resnets.1.conv2.weight",
            ),
            ("post_quant_conv.weight", "post_quant_conv.weight"),
        ];
        for (key, want) in cases {
            assert_eq!(
                whole(&format!("first_stage_model.{key}")),
                format!("vae.{want}"),
                "{key}"
            );
        }
        assert_eq!(
            map_name("first_stage_model.decoder.mid.attn_1.proj_out.weight").unwrap(),
            [(
                "vae.decoder.mid_block.attentions.0.to_out.0.weight".to_owned(),
                Take::Squeeze
            )]
        );
    }

    #[test]
    fn the_second_encoder_splits_its_fused_projection_and_transposes_its_text_projection() {
        let fused = map_name("conditioner.embedders.1.model.transformer.resblocks.3.attn.in_proj_weight").unwrap();
        let names: Vec<(String, Take)> = ["q", "k", "v"]
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let take = Take::Rows {
                    from: i * 1280,
                    count: 1280,
                };
                (
                    format!("te2.text_model.encoder.layers.3.self_attn.{x}_proj.weight"),
                    take,
                )
            })
            .collect();
        assert_eq!(fused, names);
        assert_eq!(
            map_name("conditioner.embedders.1.model.text_projection").unwrap(),
            [("te2.text_projection.weight".to_owned(), Take::Transpose)]
        );
        let cases = [
            (
                "transformer.resblocks.0.ln_2.bias",
                "text_model.encoder.layers.0.layer_norm2.bias",
            ),
            (
                "transformer.resblocks.31.mlp.c_fc.weight",
                "text_model.encoder.layers.31.mlp.fc1.weight",
            ),
            (
                "transformer.resblocks.7.attn.out_proj.bias",
                "text_model.encoder.layers.7.self_attn.out_proj.bias",
            ),
            (
                "positional_embedding",
                "text_model.embeddings.position_embedding.weight",
            ),
            ("ln_final.weight", "text_model.final_layer_norm.weight"),
        ];
        for (key, want) in cases {
            assert_eq!(
                whole(&format!("conditioner.embedders.1.model.{key}")),
                format!("te2.{want}"),
                "{key}"
            );
        }
        assert!(
            map_name("conditioner.embedders.1.model.logit_scale")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn the_first_encoder_drops_its_prefix_and_an_unknown_tensor_is_an_error() {
        assert_eq!(
            whole("conditioner.embedders.0.transformer.text_model.encoder.layers.4.mlp.fc2.bias"),
            "te1.text_model.encoder.layers.4.mlp.fc2.bias"
        );
        assert!(
            map_name("conditioner.embedders.0.transformer.text_model.embeddings.position_ids")
                .unwrap()
                .is_empty()
        );
        for key in [
            "model.diffusion_model.output_blocks.2.3.conv.weight",
            "first_stage_model.decoder.mid.attn_1.x.weight",
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_scale",
            "cond_stage_model.transformer.text_model.final_layer_norm.weight",
        ] {
            assert!(map_name(key).is_err(), "{key}");
        }
    }
}
