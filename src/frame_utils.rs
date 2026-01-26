use image::RgbaImage;
use rav1e::prelude::{Context, Frame};

pub fn to_rav1e_yuv(img: &RgbaImage, ctx: &Context<u8>) -> Frame<u8> {
    let mut frame = ctx.new_frame();
    let width = img.width() as usize;
    let height = img.height() as usize;

    let mut y_data = Vec::with_capacity(width * height);
    for pixel in img.pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        y_data.push(y.clamp(0.0, 255.0) as u8);
    }

    frame.planes[0].copy_from_raw_u8(&y_data, width, 1);

    let u_width = frame.planes[1].cfg.width;
    let u_height = frame.planes[1].cfg.height;

    let ss_x = if u_width < width { 1 } else { 0 };
    let ss_y = if u_height < height { 1 } else { 0 };

    let mut u_data = Vec::with_capacity(u_width * u_height);
    let mut v_data = Vec::with_capacity(u_width * u_height);

    for row in 0..u_height {
        for col in 0..u_width {
            let src_x = col << ss_x;
            let src_y = row << ss_y;

            let mut r_sum = 0.0;
            let mut g_sum = 0.0;
            let mut b_sum = 0.0;
            let mut count = 0.0;

            for dy in 0..(1 << ss_y) {
                for dx in 0..(1 << ss_x) {
                    let px = src_x + dx;
                    let py = src_y + dy;

                    if px < width && py < height {
                        let pixel = img.get_pixel(px as u32, py as u32);
                        r_sum += pixel[0] as f32;
                        g_sum += pixel[1] as f32;
                        b_sum += pixel[2] as f32;
                        count += 1.0;
                    }
                }
            }

            let r = r_sum / count;
            let g = g_sum / count;
            let b = b_sum / count;

            let u = -0.1146 * r - 0.3854 * g + 0.5000 * b + 128.0;
            let v = 0.5000 * r - 0.4542 * g - 0.0458 * b + 128.0;

            u_data.push(u.clamp(0.0, 255.0) as u8);
            v_data.push(v.clamp(0.0, 255.0) as u8);
        }
    }

    frame.planes[1].copy_from_raw_u8(&u_data, u_width, 1);
    frame.planes[2].copy_from_raw_u8(&v_data, u_width, 1);

    frame
}

pub fn extract_alpha(img: &RgbaImage, ctx: &Context<u8>) -> Frame<u8> {
    let mut frame = ctx.new_frame();
    let width = img.width() as usize;

    let plane_y = &mut frame.planes[0];

    let alpha_data: Vec<u8> = img.pixels().map(|p| p[3]).collect();

    plane_y.copy_from_raw_u8(&alpha_data, width, 1);

    frame
}
