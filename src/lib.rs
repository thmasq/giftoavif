use av_scenechange::{Rational32, SceneChangeDetector, SceneDetectionSpeed};
use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;
use rav1e::prelude::{ChromaSubsampling, Frame};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::io::Cursor;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

pub use wasm_bindgen_rayon::init_thread_pool;

mod encoder;
mod frame_utils;
mod isobmff;
mod resampler;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

#[wasm_bindgen(start)]
pub fn main_js() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

fn get_rav1e_detector_config(
    width: usize,
    height: usize,
    speed_preset: u8,
    fps: u32,
) -> SceneChangeDetector<u8> {
    let detection_mode = if speed_preset >= 10 {
        SceneDetectionSpeed::Fast
    } else {
        SceneDetectionSpeed::Standard
    };

    SceneChangeDetector::new(
        (width, height),
        8,
        Rational32::new(fps as i32, 1),
        ChromaSubsampling::Yuv420,
        5,
        detection_mode,
        12,
        240,
    )
}

#[wasm_bindgen]
pub fn gif_to_avif(
    gif_data: &[u8],
    fps: Option<u32>,
    crf: Option<u8>,
    encoder_speed: Option<u8>,
    playback_speed: Option<f32>,
    interpolate: Option<bool>,
) -> Result<Vec<u8>, String> {
    let target_fps = fps.unwrap_or(30);
    let target_crf = crf.unwrap_or(30);
    let target_speed = encoder_speed.unwrap_or(6);
    let target_playback_speed = playback_speed.unwrap_or(1.0);
    let should_interpolate = interpolate.unwrap_or(false);

    let cursor = Cursor::new(gif_data);
    let decoder = GifDecoder::new(cursor).map_err(|e| e.to_string())?;

    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|e: image::ImageError| e.to_string())?;

    if frames.is_empty() {
        return Err("No frames found in GIF".into());
    }

    let is_opaque = frames
        .iter()
        .all(|frame| frame.buffer().pixels().all(|p| p[3] == 255));

    let width = frames[0].buffer().width();
    let height = frames[0].buffer().height();

    log(&format!("Initializing Encoder: {}x{}", width, height));

    let enc_width = if width & 1 == 1 { width + 1 } else { width };
    let enc_height = if height & 1 == 1 { height + 1 } else { height };

    let alloc_manager = encoder::EncoderManager::new(
        enc_width as usize,
        enc_height as usize,
        target_speed,
        target_crf,
        target_fps,
        false,
    );

    let resampler_iter = resampler::Resampler::new(
        frames,
        target_fps,
        target_playback_speed as f64,
        should_interpolate,
    );

    let resampled_frames: Vec<_> = resampler_iter.collect();

    let color_frames: Vec<Arc<Frame<u8>>> = resampled_frames
        .iter()
        .map(|img| Arc::new(frame_utils::to_rav1e_yuv(img, &alloc_manager.ctx)))
        .collect();

    let alpha_frames: Option<Vec<Arc<Frame<u8>>>> = if !is_opaque {
        let alpha_ctx = encoder::EncoderManager::new(
            enc_width as usize,
            enc_height as usize,
            target_speed,
            target_crf,
            target_fps,
            true,
        )
        .ctx;

        Some(
            resampled_frames
                .iter()
                .map(|img| Arc::new(frame_utils::extract_alpha(img, &alpha_ctx)))
                .collect(),
        )
    } else {
        None
    };

    let mut keyframes = vec![0];
    {
        let mut detector = get_rav1e_detector_config(
            enc_width as usize,
            enc_height as usize,
            target_speed,
            target_fps,
        );
        let mut last_keyframe = 0;

        let frame_refs: Vec<&Arc<Frame<u8>>> = color_frames.iter().collect();

        let lookahead = 5;

        for i in 1..color_frames.len() {
            let end = (i + lookahead).min(color_frames.len());
            let window = &frame_refs[i..end];

            let (is_scenecut, _score) = detector.analyze_next_frame(window, i, last_keyframe);
            if is_scenecut {
                keyframes.push(i);
                last_keyframe = i;
            }
        }
    }

    let color_frames_owned: Vec<Frame<u8>> = color_frames
        .into_iter()
        .map(|a| Arc::try_unwrap(a).unwrap_or_else(|_| panic!("Failed to unwrap Arc")))
        .collect();

    let alpha_frames_owned = alpha_frames.map(|v| {
        v.into_iter()
            .map(|a| Arc::try_unwrap(a).unwrap_or_else(|_| panic!("Failed to unwrap Arc")))
            .collect::<Vec<_>>()
    });

    let split_into_chunks = |mut frames: Vec<Frame<u8>>,
                             indices: &[usize]|
     -> Vec<Vec<Frame<u8>>> {
        let mut chunks = Vec::new();
        let mut split_indices: Vec<usize> = indices.iter().cloned().filter(|&x| x > 0).collect();
        split_indices.sort_unstable();

        for &idx in split_indices.iter().rev() {
            if idx < frames.len() {
                let tail = frames.split_off(idx);
                chunks.push(tail);
            }
        }
        chunks.push(frames);
        chunks.reverse();
        chunks
    };

    let color_chunks = split_into_chunks(color_frames_owned, &keyframes);

    let alpha_chunks = if let Some(a_frames) = alpha_frames_owned {
        Some(split_into_chunks(a_frames, &keyframes))
    } else {
        None
    };

    let encoded_segments: Vec<(Vec<Vec<u8>>, Option<Vec<Vec<u8>>>)> = color_chunks
        .into_par_iter()
        .enumerate()
        .map(|(i, color_chunk)| {
            let alpha_chunk = alpha_chunks.as_ref().map(|chunks| chunks[i].clone());

            let mut chunk_color_manager = encoder::EncoderManager::new(
                enc_width as usize,
                enc_height as usize,
                target_speed,
                target_crf,
                target_fps,
                false,
            );

            let chunk_alpha_manager = if alpha_chunk.is_some() {
                Some(encoder::EncoderManager::new(
                    enc_width as usize,
                    enc_height as usize,
                    target_speed,
                    target_crf,
                    target_fps,
                    true,
                ))
            } else {
                None
            };

            for frame in color_chunk {
                chunk_color_manager
                    .ctx
                    .send_frame(frame)
                    .map_err(|e| format!("{:?}", e))?;
            }
            chunk_color_manager.ctx.flush();
            let c_frames = chunk_color_manager.collect_frames()?;

            let a_frames = if let Some(mut am) = chunk_alpha_manager {
                if let Some(frames) = alpha_chunk {
                    for frame in frames {
                        am.ctx.send_frame(frame).map_err(|e| format!("{:?}", e))?;
                    }
                }
                am.ctx.flush();
                Some(am.collect_frames()?)
            } else {
                None
            };

            Ok((c_frames, a_frames))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let final_color_frames: Vec<Vec<u8>> = encoded_segments
        .iter()
        .flat_map(|(c, _)| c.clone())
        .collect();

    let final_alpha_frames: Option<Vec<Vec<u8>>> = if is_opaque {
        None
    } else {
        Some(
            encoded_segments
                .iter()
                .flat_map(|(_, a)| a.clone().unwrap_or_default())
                .collect(),
        )
    };

    let output = isobmff::serialize_to_vec(
        &final_color_frames,
        final_alpha_frames.as_deref(),
        width,
        height,
        target_fps,
    )?;

    Ok(output)
}
