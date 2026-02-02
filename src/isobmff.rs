use bitstream_io::{BigEndian, BitRead, BitReader};
use byteorder::{BigEndian as BE, WriteBytesExt};
use std::io::{Cursor, Seek, SeekFrom, Write};

const BRAND_AVIS: &[u8; 4] = b"avis";
const BRAND_AVIF: &[u8; 4] = b"avif";
const BRAND_MSF1: &[u8; 4] = b"msf1";
const BRAND_ISO8: &[u8; 4] = b"iso8";
const BRAND_MIF1: &[u8; 4] = b"mif1";
const BRAND_MIAF: &[u8; 4] = b"miaf";
const BRAND_MA1B: &[u8; 4] = b"MA1B";
const BRAND_MA1A: &[u8; 4] = b"MA1A";

const MOVIE_TIMESCALE: u32 = 1000;
const MEDIA_TIMESCALE: u32 = 15360;

#[derive(Debug, Clone, Default)]
pub struct Av1SequenceInfo {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range: bool,
    pub raw_obu: Vec<u8>,
}

pub fn serialize_to_vec(
    color_frames: &[Vec<u8>],
    alpha_frames: Option<&[Vec<u8>]>,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<Vec<u8>, String> {
    if color_frames.is_empty() {
        return Err("No color frames provided".to_string());
    }

    let delta = MEDIA_TIMESCALE / fps;

    let color_sync_samples: Vec<bool> = color_frames
        .iter()
        .map(|f| parse_sequence_header(f).is_some())
        .collect();

    let color_config = parse_sequence_header(&color_frames[0])
        .ok_or("Failed to parse AV1 Sequence Header from first color frame")?;

    let (alpha_config, alpha_sync_samples) = if let Some(frames) = alpha_frames {
        if !frames.is_empty() {
            let samples: Vec<bool> = frames
                .iter()
                .map(|f| parse_sequence_header(f).is_some())
                .collect();
            (
                Some(
                    parse_sequence_header(&frames[0])
                        .ok_or("Failed to parse AV1 Sequence Header from first alpha frame")?,
                ),
                Some(samples),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let mut mdat_payload = Vec::new();
    let mut color_chunk_sizes = Vec::with_capacity(color_frames.len());
    let mut alpha_chunk_sizes = Vec::new();

    let mut first_color_size = 0u32;
    let mut first_alpha_size = 0u32;

    for i in 0..color_frames.len() {
        if let Some(alpha_frames_ref) = alpha_frames {
            if i < alpha_frames_ref.len() {
                let frame = &alpha_frames_ref[i];
                mdat_payload.extend_from_slice(frame);
                alpha_chunk_sizes.push(frame.len() as u32);
                if i == 0 {
                    first_alpha_size = frame.len() as u32;
                }
            }
        }

        let frame = &color_frames[i];
        mdat_payload.extend_from_slice(frame);
        color_chunk_sizes.push(frame.len() as u32);
        if i == 0 {
            first_color_size = frame.len() as u32;
        }
    }

    let mdat_size = 8 + mdat_payload.len() as u32;
    let mut writer = Cursor::new(Vec::new());

    write_ftyp(&mut writer, &color_config)?;

    let (iloc_pos_alpha, iloc_pos_color) = write_meta(
        &mut writer,
        width,
        height,
        &color_config,
        alpha_config.as_ref(),
        first_color_size,
        first_alpha_size,
    )?;

    let moov_start_pos = writer.position();
    let mut temp_moov = Cursor::new(Vec::new());

    write_moov(
        &mut temp_moov,
        width,
        height,
        &color_config,
        alpha_config.as_ref(),
        &color_chunk_sizes,
        &color_sync_samples,
        if alpha_config.is_some() {
            Some(&alpha_chunk_sizes)
        } else {
            None
        },
        alpha_sync_samples.as_deref(),
        0,
        delta,
    )?;

    let moov_size = temp_moov.get_ref().len() as u64;
    let mdat_start_offset = moov_start_pos + moov_size + 8;

    write_moov(
        &mut writer,
        width,
        height,
        &color_config,
        alpha_config.as_ref(),
        &color_chunk_sizes,
        &color_sync_samples,
        if alpha_config.is_some() {
            Some(&alpha_chunk_sizes)
        } else {
            None
        },
        alpha_sync_samples.as_deref(),
        mdat_start_offset as u32,
        delta,
    )?;

    write_box_header(&mut writer, b"mdat", mdat_size)?;
    writer.write_all(&mdat_payload).map_err(|e| e.to_string())?;

    let alpha_offset_abs = if alpha_config.is_some() {
        mdat_start_offset
    } else {
        0
    };
    let color_offset_abs = if alpha_config.is_some() {
        mdat_start_offset + first_alpha_size as u64
    } else {
        mdat_start_offset
    };

    if let Some(pos) = iloc_pos_color {
        let current = writer.position();
        writer
            .seek(SeekFrom::Start(pos))
            .map_err(|e| e.to_string())?;
        writer
            .write_u32::<BE>(color_offset_abs as u32)
            .map_err(|e| e.to_string())?;
        writer
            .seek(SeekFrom::Start(current))
            .map_err(|e| e.to_string())?;
    }

    if let Some(pos) = iloc_pos_alpha {
        let current = writer.position();
        writer
            .seek(SeekFrom::Start(pos))
            .map_err(|e| e.to_string())?;
        writer
            .write_u32::<BE>(alpha_offset_abs as u32)
            .map_err(|e| e.to_string())?;
        writer
            .seek(SeekFrom::Start(current))
            .map_err(|e| e.to_string())?;
    }

    Ok(writer.into_inner())
}

fn write_meta<W: Write + Seek>(
    w: &mut W,
    width: u32,
    height: u32,
    color_conf: &Av1SequenceInfo,
    alpha_conf: Option<&Av1SequenceInfo>,
    color_len: u32,
    alpha_len: u32,
) -> Result<(Option<u64>, Option<u64>), String> {
    let meta_start = w.stream_position().map_err(|e| e.to_string())?;
    write_full_box_header(w, b"meta", 0, 0, 0)?;

    let hdlr_name = b"PictureHandler\0";
    write_full_box_header(w, b"hdlr", 32 + hdlr_name.len() as u32, 0, 0)?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_all(b"pict").map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 12]).map_err(|e| e.to_string())?;
    w.write_all(hdlr_name).map_err(|e| e.to_string())?;

    write_full_box_header(w, b"pitm", 14, 0, 0)?;
    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;

    let iloc_start = w.stream_position().map_err(|e| e.to_string())?;
    write_full_box_header(w, b"iloc", 0, 0, 0)?;
    let offset_size = 4u8;
    let length_size = 4u8;
    w.write_u8((offset_size << 4) | length_size)
        .map_err(|e| e.to_string())?;
    w.write_u8(0).map_err(|e| e.to_string())?;

    let item_count = if alpha_conf.is_some() { 2 } else { 1 };
    w.write_u16::<BE>(item_count).map_err(|e| e.to_string())?;

    let mut pos_color = None;
    let mut pos_alpha = None;

    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
    pos_color = Some(w.stream_position().map_err(|e| e.to_string())?);
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(color_len).map_err(|e| e.to_string())?;

    if alpha_conf.is_some() {
        w.write_u16::<BE>(2).map_err(|e| e.to_string())?;
        w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
        w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
        pos_alpha = Some(w.stream_position().map_err(|e| e.to_string())?);
        w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
        w.write_u32::<BE>(alpha_len).map_err(|e| e.to_string())?;
    }

    patch_box_size(w, iloc_start)?;

    let iinf_start = w.stream_position().map_err(|e| e.to_string())?;
    write_full_box_header(w, b"iinf", 0, 0, 0)?;
    w.write_u16::<BE>(item_count).map_err(|e| e.to_string())?;
    write_infe(w, 1, b"Color\0")?;
    if alpha_conf.is_some() {
        write_infe(w, 2, b"Alpha\0")?;
    }
    patch_box_size(w, iinf_start)?;

    if alpha_conf.is_some() {
        let iref_start = w.stream_position().map_err(|e| e.to_string())?;
        write_full_box_header(w, b"iref", 0, 0, 0)?;
        let auxl_start = w.stream_position().map_err(|e| e.to_string())?;
        write_box_header(w, b"auxl", 0)?;
        w.write_u16::<BE>(2).map_err(|e| e.to_string())?;
        w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
        w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
        patch_box_size(w, auxl_start)?;
        patch_box_size(w, iref_start)?;
    }

    let iprp_start = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"iprp", 0)?;
    let ipco_start = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"ipco", 0)?;

    write_ispe(w, width, height)?;
    write_pixi(w, 3, 8)?;
    write_av1c(w, color_conf)?;
    write_colr(w, color_conf)?;

    if let Some(ac) = alpha_conf {
        write_ispe(w, width, height)?;
        write_pixi(w, 1, 8)?;
        write_av1c(w, ac)?;
        let urn = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";
        write_full_box_header(w, b"auxC", 12 + urn.len() as u32, 0, 0)?;
        w.write_all(urn).map_err(|e| e.to_string())?;
    }
    patch_box_size(w, ipco_start)?;

    let ipma_start = w.stream_position().map_err(|e| e.to_string())?;
    write_full_box_header(w, b"ipma", 0, 0, 0)?;
    w.write_u32::<BE>(item_count as u32)
        .map_err(|e| e.to_string())?;
    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u8(4).map_err(|e| e.to_string())?;
    for idx in 1..=4 {
        w.write_u8(idx).map_err(|e| e.to_string())?;
    }

    if alpha_conf.is_some() {
        w.write_u16::<BE>(2).map_err(|e| e.to_string())?;
        w.write_u8(4).map_err(|e| e.to_string())?;
        for idx in 5..=8 {
            w.write_u8(idx).map_err(|e| e.to_string())?;
        }
    }
    patch_box_size(w, ipma_start)?;
    patch_box_size(w, iprp_start)?;
    patch_box_size(w, meta_start)?;

    Ok((pos_alpha, pos_color))
}

fn write_infe<W: Write + Seek>(w: &mut W, id: u16, name: &[u8]) -> Result<(), String> {
    let start = w.stream_position().map_err(|e| e.to_string())?;
    write_full_box_header(w, b"infe", 0, 2, 0)?;
    w.write_u16::<BE>(id).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
    w.write_all(b"av01").map_err(|e| e.to_string())?;
    w.write_all(name).map_err(|e| e.to_string())?;
    w.write_u8(0).map_err(|e| e.to_string())?;
    w.write_u8(0).map_err(|e| e.to_string())?;
    patch_box_size(w, start)?;
    Ok(())
}

fn write_ispe<W: Write>(w: &mut W, width: u32, height: u32) -> Result<(), String> {
    write_full_box_header(w, b"ispe", 20, 0, 0)?;
    w.write_u32::<BE>(width).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(height).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_pixi<W: Write>(w: &mut W, channels: u8, depth: u8) -> Result<(), String> {
    let size = 12 + 1 + channels as u32;
    write_full_box_header(w, b"pixi", size, 0, 0)?;
    w.write_u8(channels).map_err(|e| e.to_string())?;
    for _ in 0..channels {
        w.write_u8(depth).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn write_ftyp<W: Write>(w: &mut W, config: &Av1SequenceInfo) -> Result<(), String> {
    let mut brands = vec![
        BRAND_AVIS, BRAND_AVIF, BRAND_MSF1, BRAND_ISO8, BRAND_MIF1, BRAND_MIAF,
    ];
    let depth = if config.twelve_bit {
        12
    } else if config.high_bitdepth {
        10
    } else {
        8
    };
    if depth <= 10 {
        if config.chroma_subsampling_x && config.chroma_subsampling_y {
            brands.push(BRAND_MA1B);
        } else if !config.chroma_subsampling_x && !config.chroma_subsampling_y {
            brands.push(BRAND_MA1A);
        }
    }
    let size = 8 + 4 + 4 + (brands.len() as u32 * 4);
    write_box_header(w, b"ftyp", size)?;
    w.write_all(BRAND_AVIS).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    for b in brands {
        w.write_all(b).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn write_moov<W: Write + Seek>(
    w: &mut W,
    width: u32,
    height: u32,
    color_config: &Av1SequenceInfo,
    alpha_config: Option<&Av1SequenceInfo>,
    color_sizes: &[u32],
    color_sync_samples: &[bool],
    alpha_sizes: Option<&[u32]>,
    alpha_sync_samples: Option<&[bool]>,
    mdat_start_offset: u32,
    delta: u32,
) -> Result<(), String> {
    let start_pos = w.stream_position().map_err(|e| e.to_string())?;
    let total_duration_media = (color_sizes.len() as u32) * delta;
    let total_duration_movie =
        (total_duration_media as f64 * (MOVIE_TIMESCALE as f64 / MEDIA_TIMESCALE as f64)) as u32;

    write_box_header(w, b"moov", 0)?;
    write_full_box_header(w, b"mvhd", 108, 0, 0)?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(MOVIE_TIMESCALE)
        .map_err(|e| e.to_string())?;
    w.write_u32::<BE>(total_duration_movie)
        .map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0x00010000).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0x0100).map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 10]).map_err(|e| e.to_string())?;
    for val in [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
        w.write_u32::<BE>(val).map_err(|e| e.to_string())?;
    }
    w.write_all(&[0u8; 24]).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(3).map_err(|e| e.to_string())?;

    let mut current_offset = mdat_start_offset;
    let mut color_offsets = Vec::with_capacity(color_sizes.len());
    let mut alpha_offsets = Vec::new();

    for i in 0..color_sizes.len() {
        if let Some(alpha) = alpha_sizes {
            if i < alpha.len() {
                alpha_offsets.push(current_offset);
                current_offset += alpha[i];
            }
        }
        color_offsets.push(current_offset);
        current_offset += color_sizes[i];
    }

    write_track(
        w,
        1,
        width,
        height,
        color_config,
        color_sizes,
        &color_offsets,
        color_sync_samples,
        false,
        None,
        delta,
        total_duration_movie,
    )?;
    if let Some(a_sizes) = alpha_sizes {
        let a_conf = alpha_config.expect("Alpha config missing");
        let a_sync = alpha_sync_samples.expect("Alpha sync samples missing");
        write_track(
            w,
            2,
            width,
            height,
            a_conf,
            a_sizes,
            &alpha_offsets,
            a_sync,
            true,
            Some(1),
            delta,
            total_duration_movie,
        )?;
    }

    patch_box_size(w, start_pos)?;
    Ok(())
}

fn write_track<W: Write + Seek>(
    w: &mut W,
    track_id: u32,
    width: u32,
    height: u32,
    config: &Av1SequenceInfo,
    chunk_sizes: &[u32],
    chunk_offsets: &[u32],
    sync_samples: &[bool],
    is_alpha: bool,
    ref_track: Option<u32>,
    delta: u32,
    total_duration_movie: u32,
) -> Result<(), String> {
    let start_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"trak", 0)?;

    let flags = if is_alpha { 2 } else { 3 };
    write_full_box_header(w, b"tkhd", 104, 1, flags)?;
    w.write_u64::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u64::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(track_id).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u64::<BE>(0xFFFFFFFFFFFFFFFF)
        .map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 16]).map_err(|e| e.to_string())?;
    for val in [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
        w.write_u32::<BE>(val).map_err(|e| e.to_string())?;
    }
    w.write_u32::<BE>(width << 16).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(height << 16).map_err(|e| e.to_string())?;

    if let Some(rid) = ref_track {
        let tref_pos = w.stream_position().map_err(|e| e.to_string())?;
        write_box_header(w, b"tref", 0)?;
        write_box_header(w, b"auxl", 12)?;
        w.write_u32::<BE>(rid).map_err(|e| e.to_string())?;
        patch_box_size(w, tref_pos)?;
    }

    let edts_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"edts", 0)?;
    write_full_box_header(w, b"elst", 28, 0, 0)?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(total_duration_movie)
        .map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
    patch_box_size(w, edts_pos)?;

    let mdia_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"mdia", 0)?;
    write_full_box_header(w, b"mdhd", 32, 0, 0)?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(MEDIA_TIMESCALE)
        .map_err(|e| e.to_string())?;
    w.write_u32::<BE>(chunk_sizes.len() as u32 * delta)
        .map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0x55C4).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;

    let hdlr_type = if is_alpha { b"auxv" } else { b"pict" };
    let hdlr_name: &[u8] = if is_alpha {
        b"AlphaHandler\0"
    } else {
        b"PictureHandler\0"
    };
    write_full_box_header(w, b"hdlr", 32 + hdlr_name.len() as u32, 0, 0)?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_all(hdlr_type).map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 12]).map_err(|e| e.to_string())?;
    w.write_all(hdlr_name).map_err(|e| e.to_string())?;

    let minf_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"minf", 0)?;
    write_full_box_header(w, b"vmhd", 20, 0, 1)?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 6]).map_err(|e| e.to_string())?;
    let dinf_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"dinf", 0)?;
    write_full_box_header(w, b"dref", 28, 0, 0)?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    write_full_box_header(w, b"url ", 12, 0, 1)?;
    patch_box_size(w, dinf_pos)?;
    patch_box_size(w, minf_pos)?;

    let stbl_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"stbl", 0)?;
    let stsd_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_full_box_header(w, b"stsd", 0, 0, 0)?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    let av01_pos = w.stream_position().map_err(|e| e.to_string())?;
    write_box_header(w, b"av01", 0)?;
    w.write_all(&[0u8; 6]).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0).map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 12]).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(width as u16).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(height as u16)
        .map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0x00480000).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0x00480000).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(1).map_err(|e| e.to_string())?;
    w.write_all(b"\x12AOM Coding        ")
        .map_err(|e| e.to_string())?;
    w.write_all(&[0u8; 13]).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0x0018).map_err(|e| e.to_string())?;
    w.write_u16::<BE>(0xFFFF).map_err(|e| e.to_string())?;

    write_av1c(w, config)?;
    write_full_box_header(w, b"ccst", 16, 0, 0)?;
    w.write_u32::<BE>(0x7C000000).map_err(|e| e.to_string())?;
    if !is_alpha {
        write_colr(w, config)?;
    }
    if is_alpha {
        let urn = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";
        write_full_box_header(w, b"auxi", 12 + urn.len() as u32, 0, 0)?;
        w.write_all(urn).map_err(|e| e.to_string())?;
    }
    patch_box_size(w, av01_pos)?;
    patch_box_size(w, stsd_pos)?;

    write_full_box_header(w, b"stts", 24, 0, 0)?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(chunk_sizes.len() as u32)
        .map_err(|e| e.to_string())?;
    w.write_u32::<BE>(delta).map_err(|e| e.to_string())?;

    let sync_count = sync_samples.iter().filter(|&&x| x).count();
    if sync_count < chunk_sizes.len() {
        write_stss(w, sync_samples)?;
    }

    write_sdtp(w, sync_samples)?;

    write_full_box_header(w, b"stsc", 28, 0, 0)?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(1).map_err(|e| e.to_string())?;

    write_full_box_header(w, b"stsz", 20 + chunk_sizes.len() as u32 * 4, 0, 0)?;
    w.write_u32::<BE>(0).map_err(|e| e.to_string())?;
    w.write_u32::<BE>(chunk_sizes.len() as u32)
        .map_err(|e| e.to_string())?;
    for size in chunk_sizes {
        w.write_u32::<BE>(*size).map_err(|e| e.to_string())?;
    }

    write_full_box_header(w, b"stco", 16 + chunk_offsets.len() as u32 * 4, 0, 0)?;
    w.write_u32::<BE>(chunk_offsets.len() as u32)
        .map_err(|e| e.to_string())?;
    for offset in chunk_offsets {
        w.write_u32::<BE>(*offset).map_err(|e| e.to_string())?;
    }

    patch_box_size(w, stbl_pos)?;
    patch_box_size(w, minf_pos)?;
    patch_box_size(w, mdia_pos)?;
    patch_box_size(w, start_pos)?;
    Ok(())
}

fn write_stss<W: Write>(w: &mut W, sync_samples: &[bool]) -> Result<(), String> {
    let entry_count = sync_samples.iter().filter(|&&x| x).count() as u32;
    let size = 16 + entry_count * 4;
    write_full_box_header(w, b"stss", size, 0, 0)?;
    w.write_u32::<BE>(entry_count).map_err(|e| e.to_string())?;
    for (i, &is_sync) in sync_samples.iter().enumerate() {
        if is_sync {
            w.write_u32::<BE>((i + 1) as u32)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn write_sdtp<W: Write>(w: &mut W, sync_samples: &[bool]) -> Result<(), String> {
    let size = 12 + sync_samples.len() as u32;
    write_full_box_header(w, b"sdtp", size, 0, 0)?;
    for &is_sync in sync_samples {
        let depends_on = if is_sync { 2 } else { 1 };
        let byte = depends_on << 4;
        w.write_u8(byte).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn write_av1c<W: Write>(w: &mut W, c: &Av1SequenceInfo) -> Result<(), String> {
    let payload_len = 4 + c.raw_obu.len();
    write_box_header(w, b"av1C", (8 + payload_len) as u32)?;
    w.write_u8(0x81).map_err(|e| e.to_string())?;
    w.write_u8((c.seq_profile << 5) | (c.seq_level_idx_0 & 0x1F))
        .map_err(|e| e.to_string())?;
    let b3 = (c.seq_tier_0 << 7)
        | (if c.high_bitdepth { 0x40 } else { 0 })
        | (if c.twelve_bit { 0x20 } else { 0 })
        | (if c.monochrome { 0x10 } else { 0 })
        | (if c.chroma_subsampling_x { 0x08 } else { 0 })
        | (if c.chroma_subsampling_y { 0x04 } else { 0 })
        | (c.chroma_sample_position & 0x03);
    w.write_u8(b3).map_err(|e| e.to_string())?;
    w.write_u8(0).map_err(|e| e.to_string())?;
    w.write_all(&c.raw_obu).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_colr<W: Write>(w: &mut W, c: &Av1SequenceInfo) -> Result<(), String> {
    write_box_header(w, b"colr", 19)?;
    w.write_all(b"nclx").map_err(|e| e.to_string())?;
    w.write_u16::<BE>(c.color_primaries)
        .map_err(|e| e.to_string())?;
    w.write_u16::<BE>(c.transfer_characteristics)
        .map_err(|e| e.to_string())?;
    w.write_u16::<BE>(c.matrix_coefficients)
        .map_err(|e| e.to_string())?;
    w.write_u8(if c.full_range { 0x80 } else { 0 })
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn write_box_header<W: Write>(w: &mut W, typ: &[u8; 4], size: u32) -> Result<(), String> {
    w.write_u32::<BE>(size).map_err(|e| e.to_string())?;
    w.write_all(typ).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_full_box_header<W: Write>(
    w: &mut W,
    typ: &[u8; 4],
    size: u32,
    v: u8,
    f: u32,
) -> Result<(), String> {
    write_box_header(w, typ, size)?;
    w.write_u32::<BE>((v as u32) << 24 | (f & 0xFFFFFF))
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn patch_box_size<W: Write + Seek>(w: &mut W, start: u64) -> Result<(), String> {
    let end = w.stream_position().map_err(|e| e.to_string())?;
    w.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    w.write_u32::<BE>((end - start) as u32)
        .map_err(|e| e.to_string())?;
    w.seek(SeekFrom::Start(end)).map_err(|e| e.to_string())?;
    Ok(())
}

fn parse_sequence_header(data: &[u8]) -> Option<Av1SequenceInfo> {
    let mut cursor = Cursor::new(data);
    let mut reader = BitReader::endian(&mut cursor, BigEndian);

    let read_bits = |r: &mut BitReader<&mut Cursor<&[u8]>, BigEndian>, n: u32| -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | (r.read::<1, u8>().ok()? as u32);
        }
        Some(val)
    };

    let read_uvlc = |r: &mut BitReader<&mut Cursor<&[u8]>, BigEndian>| -> Option<u32> {
        let mut zeros = 0;
        while r.read::<1, u8>().ok()? == 0 {
            zeros += 1;
        }
        if zeros > 31 {
            return None;
        }
        let val = read_bits(r, zeros)?;
        Some((1 << zeros) + val - 1)
    };

    let read_leb128 = |r: &mut BitReader<&mut Cursor<&[u8]>, BigEndian>| -> Option<u64> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = r.read::<8, u8>().ok()?;
            value |= ((byte & 0x7F) as u64) << shift;
            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
            if shift > 57 {
                return None;
            }
        }
        Some(value)
    };

    while let Ok(bit) = reader.read::<1, u8>() {
        let pos_bits = reader.position_in_bits().ok()?;
        let start_byte_pos = (pos_bits - 1) / 8;
        if bit != 0 {
            return None;
        }
        let obu_type = reader.read::<4, u8>().ok()?;
        let extension_flag = reader.read::<1, u8>().ok()? == 1;
        let has_size_field = reader.read::<1, u8>().ok()? == 1;
        let _reserved = reader.read::<1, u8>().ok()?;
        if extension_flag {
            let _ext = reader.read::<8, u8>().ok()?;
        }
        let payload_size = if has_size_field {
            read_leb128(&mut reader)?
        } else {
            ((data.len() as u64 * 8).saturating_sub(reader.position_in_bits().ok()?)) / 8
        };
        if obu_type != 1 {
            reader.skip((payload_size * 8) as u32).ok()?;
            continue;
        }

        let payload_start_byte = reader.position_in_bits().ok()? / 8;
        let raw_obu =
            data[start_byte_pos as usize..(payload_start_byte + payload_size) as usize].to_vec();
        let seq_profile = reader.read::<3, u8>().ok()?;
        let _still = reader.read::<1, u8>().ok()?;
        let reduced = reader.read::<1, u8>().ok()? == 1;

        let (seq_level_idx_0, seq_tier_0) = if reduced {
            (reader.read::<5, u8>().ok()?, 0)
        } else {
            let timing = reader.read::<1, u8>().ok()? == 1;
            let mut decoder_model = false;
            if timing {
                let _ = reader.read::<32, u32>().ok()?;
                let _ = reader.read::<32, u32>().ok()?;
                if reader.read::<1, u8>().ok()? == 1 {
                    let _ = read_uvlc(&mut reader)?;
                }
                decoder_model = reader.read::<1, u8>().ok()? == 1;
                if decoder_model {
                    let _ = reader.read::<5, u8>().ok()?;
                    let _ = reader.read::<32, u32>().ok()?;
                    let _ = reader.read::<10, u16>().ok()?;
                }
            }
            let initial_delay = reader.read::<1, u8>().ok()? == 1;
            let op_cnt = reader.read::<5, u8>().ok()?;
            let mut l0 = 0;
            let mut t0 = 0;
            for i in 0..=op_cnt {
                let _ = reader.read::<12, u16>().ok()?;
                let level = reader.read::<5, u8>().ok()?;
                let tier = if level > 7 {
                    reader.read::<1, u8>().ok()?
                } else {
                    0
                };
                if decoder_model && reader.read::<1, u8>().ok()? == 1 {
                    let _ = read_bits(&mut reader, 16)?;
                    let _ = read_bits(&mut reader, 16)?;
                    let _ = reader.read::<1, u8>().ok()?;
                }
                if initial_delay && reader.read::<1, u8>().ok()? == 1 {
                    let _ = reader.read::<4, u8>().ok()?;
                }
                if i == 0 {
                    l0 = level;
                    t0 = tier;
                }
            }
            (l0, t0)
        };

        let w_bits = reader.read::<4, u8>().ok()?;
        let h_bits = reader.read::<4, u8>().ok()?;
        let _ = read_bits(&mut reader, (w_bits + 1) as u32)?;
        let _ = read_bits(&mut reader, (h_bits + 1) as u32)?;
        if !reduced && reader.read::<1, u8>().ok()? == 1 {
            let _ = reader.read::<7, u8>().ok()?;
        }
        let _ = reader.read::<3, u8>().ok()?;
        let high_bitdepth = reader.read::<1, u8>().ok()? == 1;
        let twelve_bit = if seq_profile == 2 && high_bitdepth {
            reader.read::<1, u8>().ok()? == 1
        } else {
            false
        };
        let monochrome = if seq_profile == 1 {
            false
        } else {
            reader.read::<1, u8>().ok()? == 1
        };
        let color_desc = reader.read::<1, u8>().ok()? == 1;
        let mut cp = 1;
        let mut tc = 1;
        let mut mc = 1;
        if color_desc {
            cp = reader.read::<8, u16>().ok()?;
            tc = reader.read::<8, u16>().ok()?;
            mc = reader.read::<8, u16>().ok()?;
        }
        let fr = reader.read::<1, u8>().ok()? == 1;
        let (sub_x, sub_y) = if monochrome {
            (true, true)
        } else if cp == 1 && tc == 13 && mc == 0 {
            (false, false)
        } else {
            match seq_profile {
                0 => (true, true),
                1 => (false, false),
                2 => {
                    if high_bitdepth {
                        let sx = reader.read::<1, u8>().ok()? == 1;
                        (
                            sx,
                            if sx {
                                reader.read::<1, u8>().ok()? == 1
                            } else {
                                false
                            },
                        )
                    } else {
                        (true, false)
                    }
                }
                _ => (true, true),
            }
        };
        let pos = if sub_x && sub_y {
            reader.read::<2, u8>().ok()?
        } else {
            0
        };

        return Some(Av1SequenceInfo {
            seq_profile,
            seq_level_idx_0,
            seq_tier_0,
            high_bitdepth,
            twelve_bit,
            monochrome,
            chroma_subsampling_x: sub_x,
            chroma_subsampling_y: sub_y,
            chroma_sample_position: pos,
            color_primaries: cp,
            transfer_characteristics: tc,
            matrix_coefficients: mc,
            full_range: fr,
            raw_obu,
        });
    }
    None
}
