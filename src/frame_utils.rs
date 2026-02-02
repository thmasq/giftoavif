use image::RgbaImage;
use rav1e::prelude::{Context, Frame, PlanePad};

pub fn to_rav1e_yuv(img: &RgbaImage, ctx: &Context<u8>) -> Frame<u8> {
    let mut frame = ctx.new_frame();
    let width = img.width() as usize;
    let height = img.height() as usize;

    {
        let y_plane = &mut frame.y_plane;
        let plane_width = y_plane.width().get();
        let plane_height = y_plane.height().get();

        for row_idx in 0..plane_height {
            let y_row = y_plane.row_mut(row_idx).expect("Y row out of bounds");

            let clamped_y = row_idx.min(height - 1);

            for col_idx in 0..plane_width {
                let clamped_x = col_idx.min(width - 1);

                let pixel = img.get_pixel(clamped_x as u32, clamped_y as u32);
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                y_row[col_idx] = luma.clamp(0.0, 255.0) as u8;
            }
        }
        y_plane.pad(plane_width, plane_height);
    }

    if let (Some(u_plane), Some(v_plane)) = (frame.u_plane.as_mut(), frame.v_plane.as_mut()) {
        let plane_width = u_plane.width().get();
        let plane_height = u_plane.height().get();

        let chroma_width = (width + 1) / 2;
        let chroma_height = (height + 1) / 2;

        let shift_x = 1;
        let shift_y = 1;

        for row_idx in 0..plane_height {
            let u_row = u_plane.row_mut(row_idx).expect("U row out of bounds");
            let v_row = v_plane.row_mut(row_idx).expect("V row out of bounds");

            let clamped_y = row_idx.min(chroma_height - 1);
            let src_y = clamped_y << shift_y;

            for col_idx in 0..plane_width {
                let clamped_x = col_idx.min(chroma_width - 1);
                let src_x = clamped_x << shift_x;

                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;
                let mut count = 0.0;

                for dy in 0..(1 << shift_y) {
                    for dx in 0..(1 << shift_x) {
                        let px = (src_x + dx).min(width - 1);
                        let py = (src_y + dy).min(height - 1);

                        let pixel = img.get_pixel(px as u32, py as u32);
                        r_sum += pixel[0] as f32;
                        g_sum += pixel[1] as f32;
                        b_sum += pixel[2] as f32;
                        count += 1.0;
                    }
                }

                let r = r_sum / count;
                let g = g_sum / count;
                let b = b_sum / count;

                let u = -0.1146 * r - 0.3854 * g + 0.5000 * b + 128.0;
                let v = 0.5000 * r - 0.4542 * g - 0.0458 * b + 128.0;

                u_row[col_idx] = u.clamp(0.0, 255.0) as u8;
                v_row[col_idx] = v.clamp(0.0, 255.0) as u8;
            }
        }

        u_plane.pad(plane_width, plane_height);
        v_plane.pad(plane_width, plane_height);
    }

    frame
}

pub fn extract_alpha(img: &RgbaImage, ctx: &Context<u8>) -> Frame<u8> {
    let mut frame = ctx.new_frame();
    let width = img.width() as usize;
    let height = img.height() as usize;

    {
        let y_plane = &mut frame.y_plane;
        let plane_width = y_plane.width().get();
        let plane_height = y_plane.height().get();

        for row_idx in 0..plane_height {
            let alpha_row = y_plane.row_mut(row_idx).expect("Alpha row out of bounds");
            let clamped_y = row_idx.min(height - 1);

            for col_idx in 0..plane_width {
                let clamped_x = col_idx.min(width - 1);

                let pixel = img.get_pixel(clamped_x as u32, clamped_y as u32);
                alpha_row[col_idx] = pixel[3];
            }
        }
        y_plane.pad(plane_width, plane_height);
    }

    frame
}
