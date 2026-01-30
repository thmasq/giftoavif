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

    let y_plane = &mut frame.y_plane;
    for (dst_row, src_row) in y_plane.rows_mut().zip(y_data.chunks(width)) {
        dst_row[..width].copy_from_slice(src_row);
    }

    let u_plane = frame.u_plane.as_mut().unwrap();
    let u_width = u_plane.width().get();
    let u_height = u_plane.height().get();

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

    // FIX: Copy data to U and V planes row by row
    let u_plane = frame.u_plane.as_mut().unwrap();
    for (dst_row, src_row) in u_plane.rows_mut().zip(u_data.chunks(u_width)) {
        dst_row[..u_width].copy_from_slice(src_row);
    }

    let v_plane = frame.v_plane.as_mut().unwrap();
    for (dst_row, src_row) in v_plane.rows_mut().zip(v_data.chunks(u_width)) {
        dst_row[..u_width].copy_from_slice(src_row);
    }

    frame
}

pub fn extract_alpha(img: &RgbaImage, ctx: &Context<u8>) -> Frame<u8> {
    let mut frame = ctx.new_frame();
    let width = img.width() as usize;

    let plane_y = &mut frame.y_plane;

    let alpha_data: Vec<u8> = img.pixels().map(|p| p[3]).collect();

    // FIX: Copy alpha data row by row
    for (dst_row, src_row) in plane_y.rows_mut().zip(alpha_data.chunks(width)) {
        dst_row[..width].copy_from_slice(src_row);
    }

    frame
}
