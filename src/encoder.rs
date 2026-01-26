use rav1e::{
    color::{
        ColorDescription, ColorPrimaries, MatrixCoefficients, PixelRange, TransferCharacteristics,
    },
    prelude::{ChromaSampling, Config, Context, EncoderConfig, EncoderStatus, Rational},
};

pub struct EncoderManager {
    pub ctx: Context<u8>,
}

impl EncoderManager {
    pub fn new(width: usize, height: usize, speed: u8, crf: u8, fps: u32, is_alpha: bool) -> Self {
        let mut enc = EncoderConfig::with_speed_preset(speed);
        enc.width = width;
        enc.height = height;
        enc.quantizer = crf as usize;
        enc.time_base = Rational {
            num: 1,
            den: fps as u64,
        };
        enc.enable_timing_info = true;

        if is_alpha {
            enc.chroma_sampling = ChromaSampling::Cs400;
            enc.pixel_range = PixelRange::Full;
        } else {
            enc.chroma_sampling = ChromaSampling::Cs420;
            enc.pixel_range = PixelRange::Limited;
        }

        enc.still_picture = false;

        enc.color_description = Some(ColorDescription {
            transfer_characteristics: TransferCharacteristics::BT709,
            color_primaries: ColorPrimaries::BT709,
            matrix_coefficients: MatrixCoefficients::BT709,
        });

        let cfg = Config::new().with_encoder_config(enc);
        let ctx: Context<u8> = cfg.new_context().expect("Failed to create context");

        Self { ctx }
    }

    pub fn collect_frames(&mut self) -> Result<Vec<Vec<u8>>, String> {
        let mut frames = Vec::new();
        loop {
            match self.ctx.receive_packet() {
                Ok(pkt) => {
                    frames.push(pkt.data.to_vec());
                }
                Err(EncoderStatus::LimitReached) => break,
                Err(EncoderStatus::Encoded) => continue,
                Err(e) => return Err(format!("Encoding error: {:?}", e)),
            }
        }
        Ok(frames)
    }
}
