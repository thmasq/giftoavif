use image::{Frame, RgbaImage};

pub struct Resampler {
    frames: Vec<Frame>,
    timestamps: Vec<f64>,
    total_duration: f64,
    interval: f64,
    playback_speed: f64,
    current_time: f64,
    width: u32,
    height: u32,
    interpolate: bool,
}

impl Resampler {
    pub fn new(
        frames: Vec<Frame>,
        target_fps: u32,
        playback_speed: f64,
        interpolate: bool,
    ) -> Self {
        if frames.is_empty() {
            panic!("Resampler initialized with no frames");
        }

        let width = frames[0].buffer().width();
        let height = frames[0].buffer().height();

        let mut timestamps = Vec::with_capacity(frames.len());
        let mut current_ts = 0.0;

        for frame in &frames {
            timestamps.push(current_ts);
            let delay_ms = frame.delay().numer_denom_ms().0 as f64;
            current_ts += delay_ms / 1000.0;
        }

        Self {
            frames,
            timestamps,
            total_duration: current_ts,
            interval: 1.0 / target_fps as f64,
            playback_speed,
            current_time: 0.0,
            width,
            height,
            interpolate,
        }
    }

    fn get_pixel_component(&self, frame_idx: usize, x: u32, y: u32, c: usize) -> f64 {
        let pixel = self.frames[frame_idx].buffer().get_pixel(x, y);
        pixel[c] as f64
    }

    fn catmull_rom(
        &self,
        p0_idx: usize,
        p1_idx: usize,
        p2_idx: usize,
        p3_idx: usize,
        t: f64,
        x: u32,
        y: u32,
        c: usize,
    ) -> u8 {
        let p0 = self.get_pixel_component(p0_idx, x, y, c);
        let p1 = self.get_pixel_component(p1_idx, x, y, c);
        let p2 = self.get_pixel_component(p2_idx, x, y, c);
        let p3 = self.get_pixel_component(p3_idx, x, y, c);

        let v = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);

        v.clamp(0.0, 255.0) as u8
    }
}

impl Iterator for Resampler {
    type Item = RgbaImage;

    fn next(&mut self) -> Option<Self::Item> {
        let scaled_duration = self.total_duration / self.playback_speed;

        if self.current_time > scaled_duration + 0.0001 {
            return None;
        }

        let src_time = self.current_time * self.playback_speed;

        let mut idx = 0;
        for (i, ts) in self.timestamps.iter().enumerate().rev() {
            if *ts <= src_time {
                idx = i;
                break;
            }
        }

        if !self.interpolate {
            self.current_time += self.interval;
            return Some(self.frames[idx].buffer().clone());
        }

        let p1_idx = idx;
        let p2_idx = (idx + 1).min(self.frames.len() - 1);
        let p0_idx = if idx == 0 { 0 } else { idx - 1 };
        let p3_idx = (idx + 2).min(self.frames.len() - 1);

        let t1 = self.timestamps[p1_idx];
        let t2 = if p1_idx == p2_idx {
            t1 + 1.0
        } else {
            self.timestamps[p2_idx]
        };

        let duration = t2 - t1;
        let t = if duration <= 0.0 {
            0.0
        } else {
            (src_time - t1) / duration
        };

        let mut buffer = RgbaImage::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let r = self.catmull_rom(p0_idx, p1_idx, p2_idx, p3_idx, t, x, y, 0);
                let g = self.catmull_rom(p0_idx, p1_idx, p2_idx, p3_idx, t, x, y, 1);
                let b = self.catmull_rom(p0_idx, p1_idx, p2_idx, p3_idx, t, x, y, 2);
                let a = self.catmull_rom(p0_idx, p1_idx, p2_idx, p3_idx, t, x, y, 3);

                buffer.put_pixel(x, y, image::Rgba([r, g, b, a]));
            }
        }

        self.current_time += self.interval;
        Some(buffer)
    }
}
