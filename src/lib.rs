use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

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

    let mut color_manager = encoder::EncoderManager::new(
        width as usize,
        height as usize,
        target_speed,
        target_crf,
        target_fps,
        false,
    );

    let mut alpha_manager = if !is_opaque {
        Some(encoder::EncoderManager::new(
            width as usize,
            height as usize,
            target_speed,
            target_crf,
            target_fps,
            true,
        ))
    } else {
        None
    };

    let mut resampler_iter = resampler::Resampler::new(
        frames,
        target_fps,
        target_playback_speed as f64,
        should_interpolate,
    );

    loop {
        let frame_opt = resampler_iter.next();

        match frame_opt {
            Some(buffer) => {
                let color_yuv = frame_utils::to_rav1e_yuv(&buffer, &color_manager.ctx);

                let alpha_yuv = if let Some(am) = &alpha_manager {
                    Some(frame_utils::extract_alpha(&buffer, &am.ctx))
                } else {
                    None
                };

                color_manager
                    .ctx
                    .send_frame(color_yuv)
                    .map_err(|e| format!("{:?}", e))?;

                if let Some(am) = &mut alpha_manager {
                    if let Some(ay) = alpha_yuv {
                        am.ctx.send_frame(ay).map_err(|e| format!("{:?}", e))?;
                    }
                }
            }
            None => break,
        }
    }

    color_manager.ctx.flush();
    if let Some(am) = &mut alpha_manager {
        am.ctx.flush();
    }

    let color_frames = color_manager.collect_frames()?;

    let alpha_frames_vec = if let Some(mut am) = alpha_manager {
        Some(am.collect_frames()?)
    } else {
        None
    };

    let output = isobmff::serialize_to_vec(
        &color_frames,
        alpha_frames_vec.as_deref(),
        width,
        height,
        target_fps,
    )?;

    Ok(output)
}
