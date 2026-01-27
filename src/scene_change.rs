#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use std::cmp;
use std::collections::BTreeMap;
use std::num::{NonZeroU8, NonZeroUsize};
use std::sync::{Arc, RwLock};
use v_frame::chroma::ChromaSubsampling;
use v_frame::frame::Frame;
use v_frame::frame::FrameBuilder;
use v_frame::plane::Plane;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

const IMP_BLOCK_SIZE: usize = 8;
const REF_FRAMES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SceneDetectionSpeed {
    Standard,
    Fast,
}

#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    pub use_chroma: bool,
    pub bit_depth: usize,
    pub ignore_flashes: bool,
    pub min_scenecut_distance: Option<usize>,
    pub max_scenecut_distance: Option<usize>,
    pub lookahead: usize,
    pub scene_detection_speed: SceneDetectionSpeed,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            use_chroma: true,
            bit_depth: 8,
            ignore_flashes: false,
            min_scenecut_distance: None,
            max_scenecut_distance: None,
            lookahead: 5,
            scene_detection_speed: SceneDetectionSpeed::Standard,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScenecutResult {
    pub inter_cost: f64,
    pub imp_block_cost: f64,
    pub backward_adjusted_cost: f64,
    pub forward_adjusted_cost: f64,
    pub threshold: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub row: i16,
    pub col: i16,
}

impl MotionVector {
    pub const fn quantize_to_fullpel(self) -> Self {
        Self {
            row: (self.row / 8) * 8,
            col: (self.col / 8) * 8,
        }
    }
}

impl std::ops::Add for MotionVector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            row: self.row + rhs.row,
            col: self.col + rhs.col,
        }
    }
}

impl std::ops::Sub for MotionVector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            row: self.row - rhs.row,
            col: self.col - rhs.col,
        }
    }
}

impl std::ops::Mul<i16> for MotionVector {
    type Output = Self;
    fn mul(self, rhs: i16) -> Self {
        Self {
            row: self.row * rhs,
            col: self.col * rhs,
        }
    }
}

impl std::ops::Shl<u8> for MotionVector {
    type Output = Self;
    fn shl(self, rhs: u8) -> Self {
        Self {
            row: self.row << rhs,
            col: self.col << rhs,
        }
    }
}

impl std::ops::Shr<u8> for MotionVector {
    type Output = Self;
    fn shr(self, rhs: u8) -> Self {
        Self {
            row: self.row >> rhs,
            col: self.col >> rhs,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MEStats {
    mv: MotionVector,
    sad: u32,
}

struct FrameMEStats {
    stats: Vec<MEStats>,
    cols: usize,
    rows: usize,
}

impl FrameMEStats {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            stats: vec![MEStats::default(); cols * rows],
            cols,
            rows,
        }
    }

    fn get(&self, x: usize, y: usize) -> MEStats {
        self.stats[y * self.cols + x]
    }

    fn get_mut(&mut self, x: usize, y: usize) -> &mut MEStats {
        &mut self.stats[y * self.cols + x]
    }
}

pub struct SceneChangeDetector {
    threshold: f64,
    bit_depth: usize,
    min_key_frame_interval: usize,
    max_key_frame_interval: usize,
    lookahead_offset: usize,
    deque_offset: usize,
    score_deque: Vec<ScenecutResult>,
    intra_costs: BTreeMap<usize, Box<[u32]>>,
    frame_me_stats_buffer: Arc<RwLock<[FrameMEStats; REF_FRAMES]>>,
    temp_plane: Plane<u8>,
    detection_speed: SceneDetectionSpeed,
    downscaling_factor: usize,
    scaled_resolution: (usize, usize),
    downscaled_frame_buffer: [Plane<u8>; 2],
    last_analyzed_frame: Option<Arc<Frame<u8>>>,
}

impl SceneChangeDetector {
    pub fn new(opts: DetectionOptions, width: usize, height: usize) -> Self {
        let threshold = 18.0 * (opts.bit_depth as f64) / 8.0;

        let small_edge = cmp::min(width, height);
        let downscaling_factor = match opts.scene_detection_speed {
            SceneDetectionSpeed::Fast => match small_edge {
                0..=240 => 1,
                241..=480 => 2,
                481..=720 => 4,
                721..=1080 => 8,
                1081..=1600 => 16,
                _ => 32,
            },
            _ => 1,
        };

        let scaled_width = width / downscaling_factor;
        let scaled_height = height / downscaling_factor;

        let build_plane = |w, h| {
            FrameBuilder::new(
                NonZeroUsize::new(w).unwrap(),
                NonZeroUsize::new(h).unwrap(),
                ChromaSubsampling::Monochrome,
                NonZeroU8::new(8).unwrap(),
            )
            .build::<u8>()
            .unwrap()
            .y_plane
        };

        let downscaled_frame_buffer = [
            build_plane(scaled_width, scaled_height),
            build_plane(scaled_width, scaled_height),
        ];

        let me_cols = (width + 7) / 8;
        let me_rows = (height + 7) / 8;
        let frame_me_stats_buffer = Arc::new(RwLock::new(std::array::from_fn(|_| {
            FrameMEStats::new(me_cols, me_rows)
        })));

        let temp_plane = build_plane(width, height);

        Self {
            threshold,
            bit_depth: opts.bit_depth,
            min_key_frame_interval: opts.min_scenecut_distance.unwrap_or(0),
            max_key_frame_interval: opts.max_scenecut_distance.unwrap_or(usize::MAX),
            lookahead_offset: 5,
            deque_offset: 5,
            score_deque: Vec::with_capacity(5 + opts.lookahead),
            intra_costs: BTreeMap::new(),
            frame_me_stats_buffer,
            temp_plane,
            detection_speed: opts.scene_detection_speed,
            downscaling_factor,
            scaled_resolution: (scaled_width, scaled_height),
            downscaled_frame_buffer,
            last_analyzed_frame: None,
        }
    }

    pub fn analyze_next_frame(
        &mut self,
        frame_set: &[&Arc<Frame<u8>>],
        input_frameno: usize,
        previous_keyframe: usize,
    ) -> bool {
        let distance = input_frameno - previous_keyframe;

        if frame_set.len() <= self.lookahead_offset {
            return false;
        }

        if self.score_deque.is_empty() {
            let init_len = frame_set.len().saturating_sub(1);
            for x in 0..init_len {
                self.run_comparison(frame_set[x], frame_set[x + 1], input_frameno + x);
            }
            if frame_set.len() <= self.deque_offset + 1 {
                self.deque_offset = frame_set.len().saturating_sub(2);
            }
        } else if frame_set.len() > self.deque_offset + 1 {
            self.run_comparison(
                frame_set[self.deque_offset],
                frame_set[self.deque_offset + 1],
                input_frameno + self.deque_offset,
            );
        }

        let (scenecut, _) = self.adaptive_scenecut();

        if distance < self.min_key_frame_interval {
            return false;
        }
        if distance >= self.max_key_frame_interval {
            return true;
        }

        if self.score_deque.len() > 5 + self.lookahead_offset {
            self.score_deque.pop();
        }

        scenecut
    }

    fn run_comparison(
        &mut self,
        frame1: &Arc<Frame<u8>>,
        frame2: &Arc<Frame<u8>>,
        input_frameno: usize,
    ) {
        let mut result = match self.detection_speed {
            SceneDetectionSpeed::Fast => self.fast_scenecut(frame1, frame2),
            SceneDetectionSpeed::Standard => self.cost_scenecut(frame1, frame2, input_frameno),
        };

        self.last_analyzed_frame = Some(Arc::clone(frame2));

        if self.deque_offset > 0 {
            if input_frameno == 1 {
                result.backward_adjusted_cost = 0.0;
            } else {
                let mut adjusted_cost = f64::MAX;
                for other in self.score_deque.iter().take(self.deque_offset) {
                    let cost = result.inter_cost - other.inter_cost;
                    if cost < adjusted_cost {
                        adjusted_cost = cost;
                    }
                    if adjusted_cost < 0.0 {
                        adjusted_cost = 0.0;
                        break;
                    }
                }
                result.backward_adjusted_cost = adjusted_cost;
            }

            if !self.score_deque.is_empty() {
                for i in 0..cmp::min(self.deque_offset, self.score_deque.len()) {
                    let adjusted = self.score_deque[i].inter_cost - result.inter_cost;
                    if i == 0 || adjusted < self.score_deque[i].forward_adjusted_cost {
                        self.score_deque[i].forward_adjusted_cost = adjusted;
                    }
                    if self.score_deque[i].forward_adjusted_cost < 0.0 {
                        self.score_deque[i].forward_adjusted_cost = 0.0;
                    }
                }
            }
        }
        self.score_deque.insert(0, result);
    }

    fn fast_scenecut(
        &mut self,
        frame1: &Arc<Frame<u8>>,
        frame2: &Arc<Frame<u8>>,
    ) -> ScenecutResult {
        if self.downscaling_factor == 1 {
            let sad = calculate_sad_plane(&frame1.y_plane, &frame2.y_plane);
            let pixels = frame1.y_plane.width().get() * frame1.y_plane.height().get();
            let delta = sad as f64 / pixels as f64;
            return self.build_result(delta);
        }

        let can_reuse_buffer = self
            .last_analyzed_frame
            .as_ref()
            .map_or(false, |last| Arc::ptr_eq(frame1, last));

        if can_reuse_buffer {
            self.downscaled_frame_buffer.swap(0, 1);
            downscale_in_place(
                &frame2.y_plane,
                &mut self.downscaled_frame_buffer[1],
                self.downscaling_factor,
            );
        } else {
            downscale_in_place(
                &frame1.y_plane,
                &mut self.downscaled_frame_buffer[0],
                self.downscaling_factor,
            );
            downscale_in_place(
                &frame2.y_plane,
                &mut self.downscaled_frame_buffer[1],
                self.downscaling_factor,
            );
        }

        let sad = calculate_sad_plane(
            &self.downscaled_frame_buffer[0],
            &self.downscaled_frame_buffer[1],
        );
        let delta = sad as f64 / (self.scaled_resolution.0 * self.scaled_resolution.1) as f64;

        self.build_result(delta)
    }

    fn build_result(&self, cost: f64) -> ScenecutResult {
        ScenecutResult {
            inter_cost: cost,
            imp_block_cost: cost,
            backward_adjusted_cost: 0.0,
            forward_adjusted_cost: 0.0,
            threshold: self.threshold,
        }
    }

    fn cost_scenecut(
        &mut self,
        frame1: &Arc<Frame<u8>>,
        frame2: &Arc<Frame<u8>>,
        input_frameno: usize,
    ) -> ScenecutResult {
        let mut intra_cost = 0.0;
        let mut inter_cost = 0.0;
        let mut imp_cost = 0.0;
        let buffer = self.frame_me_stats_buffer.clone();

        rayon::scope(|s| {
            s.spawn(|_| {
                let costs = estimate_intra_costs(&mut self.temp_plane, frame2, self.bit_depth);
                intra_cost =
                    costs.iter().map(|&c| c as u64).sum::<u64>() as f64 / costs.len() as f64;
                self.intra_costs.insert(input_frameno, costs);
            });
            s.spawn(|_| {
                inter_cost =
                    estimate_inter_costs(frame2, frame1, self.bit_depth, buffer, input_frameno);
            });
            s.spawn(|_| {
                imp_cost = estimate_importance_block_diff(frame2, frame1);
            });
        });

        const BIAS: f64 = 0.7;
        let threshold = intra_cost * (1.0 - BIAS);

        ScenecutResult {
            inter_cost,
            imp_block_cost: imp_cost,
            threshold,
            backward_adjusted_cost: 0.0,
            forward_adjusted_cost: 0.0,
        }
    }

    fn adaptive_scenecut(&self) -> (bool, ScenecutResult) {
        if self.deque_offset >= self.score_deque.len() {
            return (
                false,
                ScenecutResult {
                    inter_cost: 0.0,
                    imp_block_cost: 0.0,
                    backward_adjusted_cost: 0.0,
                    forward_adjusted_cost: 0.0,
                    threshold: 0.0,
                },
            );
        }
        let score = self.score_deque[self.deque_offset];

        let imp_threshold = 7.0 * (self.bit_depth as f64) / 8.0;
        let has_high_imp = self.score_deque[self.deque_offset..]
            .iter()
            .any(|r| r.imp_block_cost >= imp_threshold);

        if !has_high_imp {
            return (false, score);
        }

        let cost = score.forward_adjusted_cost;
        if cost >= score.threshold {
            let back_deque = &self.score_deque[self.deque_offset + 1..];
            let forward_deque = &self.score_deque[..self.deque_offset];

            let back_over = back_deque
                .iter()
                .filter(|r| r.backward_adjusted_cost >= r.threshold)
                .count();
            let fwd_over = forward_deque
                .iter()
                .filter(|r| r.forward_adjusted_cost >= r.threshold)
                .count();

            if fwd_over == 0 && back_over >= 1 {
                return (true, score);
            }
            if back_over == 0
                && fwd_over == 1
                && forward_deque[0].forward_adjusted_cost >= forward_deque[0].threshold
            {
                return (true, score);
            }
            if back_over != 0 || fwd_over != 0 {
                return (false, score);
            }
        }
        (cost >= score.threshold, score)
    }
}

fn estimate_inter_costs(
    frame: &Arc<Frame<u8>>,
    ref_frame: &Arc<Frame<u8>>,
    _bit_depth: usize,
    buffer: Arc<RwLock<[FrameMEStats; REF_FRAMES]>>,
    input_frameno: usize,
) -> f64 {
    let plane = &frame.y_plane;
    let w_in_b = plane.width().get() / IMP_BLOCK_SIZE;
    let h_in_b = plane.height().get() / IMP_BLOCK_SIZE;

    let buf_idx = input_frameno % REF_FRAMES;
    let prev_buf_idx = (input_frameno.wrapping_sub(1)) % REF_FRAMES;

    let (prev_stats_snapshot, prev_cols) = {
        let lock = buffer.read().unwrap();
        (lock[prev_buf_idx].stats.clone(), lock[prev_buf_idx].cols)
    };

    let mut lock = buffer.write().unwrap();
    let curr_stats = &mut lock[buf_idx].stats;

    const ROWS_PER_TILE: usize = 8;
    let chunk_size = w_in_b * ROWS_PER_TILE;

    let total_cost: u64 = curr_stats
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(tile_idx, tile_mvs)| {
            let tile_start_y = tile_idx * ROWS_PER_TILE;
            let mut tile_cost = 0;

            let mut top_row_mvs = vec![MotionVector::default(); w_in_b];

            for r in 0..ROWS_PER_TILE {
                let y = tile_start_y + r;
                if y >= h_in_b {
                    break;
                }

                let mut left_mv = MotionVector::default();

                for x in 0..w_in_b {
                    let top_mv = if r == 0 {
                        MotionVector::default()
                    } else {
                        top_row_mvs[x]
                    };

                    let me_result = estimate_motion(
                        x,
                        y,
                        &frame.y_plane,
                        &ref_frame.y_plane,
                        &prev_stats_snapshot,
                        prev_cols,
                        left_mv,
                        top_mv,
                    );

                    let flat_idx = r * w_in_b + x;
                    tile_mvs[flat_idx] = me_result;

                    tile_cost += me_result.sad as u64;

                    left_mv = me_result.mv;
                    top_row_mvs[x] = me_result.mv;
                }
            }
            tile_cost
        })
        .sum();

    total_cost as f64 / (w_in_b * h_in_b) as f64
}

fn estimate_motion(
    bx: usize,
    by: usize,
    current: &Plane<u8>,
    reference: &Plane<u8>,
    prev_stats: &[MEStats],
    prev_cols: usize,
    left_mv: MotionVector,
    top_mv: MotionVector,
) -> MEStats {
    let x = bx * IMP_BLOCK_SIZE;
    let y = by * IMP_BLOCK_SIZE;

    let predictors = get_subset_predictors(bx, by, prev_stats, prev_cols, left_mv, top_mv);

    let mut best_mv = MotionVector::default();
    let mut best_sad = u32::MAX;

    for &mv in &predictors {
        let sad = get_sad_8x8(current, reference, x, y, mv.col, mv.row);
        if sad < best_sad {
            best_sad = sad;
            best_mv = mv;
        }
    }

    best_mv = hexagon_search(current, reference, x, y, best_mv, &mut best_sad);
    best_mv = square_refine(current, reference, x, y, best_mv, &mut best_sad);

    let (final_mv, final_satd) = subpel_refine(current, reference, x, y, best_mv);

    MEStats {
        mv: final_mv,
        sad: final_satd,
    }
}

fn get_subset_predictors(
    bx: usize,
    by: usize,
    prev_stats: &[MEStats],
    prev_cols: usize,
    left_mv: MotionVector,
    top_mv: MotionVector,
) -> Vec<MotionVector> {
    let mut preds = Vec::with_capacity(5);

    preds.push(MotionVector::default());

    if bx > 0 {
        preds.push(left_mv);
    }
    if by > 0 {
        preds.push(top_mv);
    }

    if by < prev_stats.len() / prev_cols && bx < prev_cols {
        preds.push(prev_stats[by * prev_cols + bx].mv);
    }

    preds
}

fn downscale_in_place(src: &Plane<u8>, dst: &mut Plane<u8>, scale: usize) {
    let src_width = src.width().get();
    let dst_width = dst.width().get();
    let dst_height = dst.height().get();

    for y in 0..dst_height {
        let src_y = y * scale;
        for x in 0..dst_width {
            let src_x = x * scale;
            let mut sum = 0u32;
            let mut count = 0u32;

            for iy in 0..scale {
                for ix in 0..scale {
                    if let Some(row) = src.row(src_y + iy) {
                        if src_x + ix < src_width {
                            sum += row[src_x + ix] as u32;
                            count += 1;
                        }
                    }
                }
            }

            let val = if count > 0 {
                (sum + count / 2) / count
            } else {
                0
            };
            if let Some(row) = dst.row_mut(y) {
                row[x] = val as u8;
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn calculate_sad_plane(p1: &Plane<u8>, p2: &Plane<u8>) -> u64 {
    let width = cmp::min(p1.width().get(), p2.width().get());
    let height = cmp::min(p1.height().get(), p2.height().get());

    let mut sum = 0u64;
    for y in 0..height {
        let r1 = p1.row(y).unwrap();
        let r2 = p2.row(y).unwrap();
        for x in 0..width {
            sum += (r1[x] as i32 - r2[x] as i32).abs() as u64;
        }
    }
    sum
}

#[cfg(target_arch = "wasm32")]
fn calculate_sad_plane(p1: &Plane<u8>, p2: &Plane<u8>) -> u64 {
    let w = p1.width().get();
    let h = p1.height().get();

    let n_rows = h.min(p2.height().get());
    let n_cols = w.min(p2.width().get());

    let stride1 = p1.geometry().stride.get();
    let stride2 = p2.geometry().stride.get();

    let ptr1 = p1.data().as_ptr();
    let ptr2 = p2.data().as_ptr();

    let mut sum_u64 = 0u64;

    for y in 0..n_rows {
        let row1 = unsafe { ptr1.add(y * stride1) };
        let row2 = unsafe { ptr2.add(y * stride2) };
        let mut x = 0;

        let mut row_sum = u32x4_splat(0);
        while x + 16 <= n_cols {
            let a = unsafe { v128_load(row1.add(x) as *const v128) };
            let b = unsafe { v128_load(row2.add(x) as *const v128) };

            let diff = v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a));

            let diff_lo = u16x8_extend_low_u8x16(diff);
            let diff_hi = u16x8_extend_high_u8x16(diff);
            let sum_lo = i32x4_extadd_pairwise_i16x8(diff_lo as v128);
            let sum_hi = i32x4_extadd_pairwise_i16x8(diff_hi as v128);

            row_sum = i32x4_add(row_sum, i32x4_add(sum_lo, sum_hi));
            x += 16;
        }

        sum_u64 += (i32x4_extract_lane::<0>(row_sum)
            + i32x4_extract_lane::<1>(row_sum)
            + i32x4_extract_lane::<2>(row_sum)
            + i32x4_extract_lane::<3>(row_sum)) as u64;

        while x < n_cols {
            let a = unsafe { *row1.add(x) } as i32;
            let b = unsafe { *row2.add(x) } as i32;
            sum_u64 += (a - b).abs() as u64;
            x += 1;
        }
    }
    sum_u64
}

const HEXAGON_PATTERN: [MotionVector; 6] = [
    MotionVector { col: 0, row: -2 },
    MotionVector { col: 2, row: -1 },
    MotionVector { col: 2, row: 1 },
    MotionVector { col: 0, row: 2 },
    MotionVector { col: -2, row: 1 },
    MotionVector { col: -2, row: -1 },
];

const SQUARE_PATTERN: [MotionVector; 8] = [
    MotionVector { col: -1, row: -1 },
    MotionVector { col: 0, row: -1 },
    MotionVector { col: 1, row: -1 },
    MotionVector { col: -1, row: 0 },
    MotionVector { col: 1, row: 0 },
    MotionVector { col: -1, row: 1 },
    MotionVector { col: 0, row: 1 },
    MotionVector { col: 1, row: 1 },
];

fn hexagon_search(
    cur: &Plane<u8>,
    ref_p: &Plane<u8>,
    x: usize,
    y: usize,
    start_mv: MotionVector,
    best_sad: &mut u32,
) -> MotionVector {
    let mut best_mv = start_mv;

    loop {
        let mut found_better = false;
        let mut best_cand_mv = best_mv;

        for &offset in &HEXAGON_PATTERN {
            let cand_mv = best_mv + offset;
            let sad = get_sad_8x8(cur, ref_p, x, y, cand_mv.col, cand_mv.row);
            if sad < *best_sad {
                *best_sad = sad;
                best_cand_mv = cand_mv;
                found_better = true;
            }
        }

        if found_better {
            best_mv = best_cand_mv;
        } else {
            break;
        }
    }
    best_mv
}

fn square_refine(
    cur: &Plane<u8>,
    ref_p: &Plane<u8>,
    x: usize,
    y: usize,
    start_mv: MotionVector,
    best_sad: &mut u32,
) -> MotionVector {
    let mut best_mv = start_mv;
    for &offset in &SQUARE_PATTERN {
        let cand_mv = start_mv + offset;
        let sad = get_sad_8x8(cur, ref_p, x, y, cand_mv.col, cand_mv.row);
        if sad < *best_sad {
            *best_sad = sad;
            best_mv = cand_mv;
        }
    }
    best_mv
}

fn subpel_refine(
    cur: &Plane<u8>,
    ref_p: &Plane<u8>,
    x: usize,
    y: usize,
    full_mv: MotionVector,
) -> (MotionVector, u32) {
    let mut best_mv = full_mv << 3;
    let mut best_satd = get_satd_inter(cur, ref_p, x, y, best_mv);

    let step = 4;
    let dirs = [
        MotionVector { col: 0, row: -step },
        MotionVector { col: -step, row: 0 },
        MotionVector { col: step, row: 0 },
        MotionVector { col: 0, row: step },
    ];

    let mut changed = true;
    while changed {
        changed = false;
        for &dir in &dirs {
            let cand_mv = best_mv + dir;
            let satd = get_satd_inter(cur, ref_p, x, y, cand_mv);
            if satd < best_satd {
                best_satd = satd;
                best_mv = cand_mv;
                changed = true;
            }
        }
    }

    (best_mv, best_satd)
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn get_satd_8x8(p1: &Plane<u8>, p2: &Plane<u8>, x: usize, y: usize) -> u32 {
    let mut diff = [0i16; 64];
    for i in 0..8 {
        if let (Some(r1), Some(r2)) = (p1.row(y + i), p2.row(y + i)) {
            for j in 0..8 {
                diff[i * 8 + j] = (r1[x + j] as i16) - (r2[x + j] as i16);
            }
        }
    }
    hadamard_8x8(&mut diff)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn get_satd_8x8(p1: &Plane<u8>, p2: &Plane<u8>, x: usize, y: usize) -> u32 {
    let stride1 = p1.geometry().stride.get();
    let stride2 = p2.geometry().stride.get();

    let ptr1 = unsafe { p1.data().as_ptr().add(y * stride1 + x) };
    let ptr2 = unsafe { p2.data().as_ptr().add(y * stride2 + x) };

    let mut v = [i32x4_splat(0); 8];
    for i in 0..8 {
        let r1 = unsafe { v128_load64_zero(ptr1.add(i * stride1) as *const u64) };
        let r2 = unsafe { v128_load64_zero(ptr2.add(i * stride2) as *const u64) };

        let r1_16 = u16x8_extend_low_u8x16(r1);
        let r2_16 = u16x8_extend_low_u8x16(r2);
        v[i] = i16x8_sub(r1_16 as v128, r2_16 as v128);
    }

    unsafe { hadamard_butterfly(&mut v) };
    unsafe { transpose_8x8_i16(&mut v) };
    unsafe { hadamard_butterfly(&mut v) };

    let mut sum = u32x4_splat(0);
    for i in 0..8 {
        let abs = i16x8_abs(v[i]);
        sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(abs));
    }

    (i32x4_extract_lane::<0>(sum)
        + i32x4_extract_lane::<1>(sum)
        + i32x4_extract_lane::<2>(sum)
        + i32x4_extract_lane::<3>(sum)) as u32
        / 2
}

#[cfg(not(target_arch = "wasm32"))]
fn get_satd_inter(p1: &Plane<u8>, p2: &Plane<u8>, x: usize, y: usize, mv: MotionVector) -> u32 {
    let mx = x as isize * 8 + mv.col as isize;
    let my = y as isize * 8 + mv.row as isize;

    let x_int = (mx >> 3) as isize;
    let y_int = (my >> 3) as isize;

    let x_frac = (mx & 7) as u16;
    let y_frac = (my & 7) as u16;

    if x_int < 0
        || y_int < 0
        || (x_int as usize + 9) > p2.width().get()
        || (y_int as usize + 9) > p2.height().get()
    {
        return u32::MAX;
    }

    let mut diff = [0i16; 64];

    for i in 0..8 {
        let p2_row0 = p2.row((y_int + i) as usize).unwrap();
        let p2_row1 = p2.row((y_int + i + 1) as usize).unwrap();
        let p1_row = p1.row(y + i as usize).unwrap();

        for j in 0..8 {
            let src_idx = (x_int + j) as usize;

            let a = p2_row0[src_idx] as u16;
            let b = p2_row0[src_idx + 1] as u16;
            let c = p2_row1[src_idx] as u16;
            let d = p2_row1[src_idx + 1] as u16;

            let val_top = (a * (8 - x_frac) + b * x_frac) >> 3;
            let val_bot = (c * (8 - x_frac) + d * x_frac) >> 3;
            let val = (val_top * (8 - y_frac) + val_bot * y_frac) >> 3;

            diff[i as usize * 8 + j as usize] = (p1_row[x + j as usize] as i16) - (val as i16);
        }
    }
    hadamard_8x8(&mut diff)
}

#[cfg(target_arch = "wasm32")]
fn get_satd_inter(p1: &Plane<u8>, p2: &Plane<u8>, x: usize, y: usize, mv: MotionVector) -> u32 {
    let mx = x as isize * 8 + mv.col as isize;
    let my = y as isize * 8 + mv.row as isize;

    let x_int = (mx >> 3) as usize;
    let y_int = (my >> 3) as usize;

    let x_frac = (mx & 7) as u16;
    let y_frac = (my & 7) as u16;

    let stride1 = p1.geometry().stride.get();
    let stride2 = p2.geometry().stride.get();

    let ptr1 = unsafe { p1.data().as_ptr().add(y * stride1 + x) };
    let ptr2 = unsafe { p2.data().as_ptr().add(y_int * stride2 + x_int) };

    let w_xf = i16x8_splat(x_frac as i16);
    let w_yf = i16x8_splat(y_frac as i16);

    let w_8_xf = i16x8_splat((8 - x_frac) as i16);
    let w_8_yf = i16x8_splat((8 - y_frac) as i16);

    let mut v = [i32x4_splat(0); 8];

    for i in 0..8 {
        let r2_a_ptr = unsafe { ptr2.add(i * stride2) };

        let row_a_full = unsafe { v128_load(r2_a_ptr as *const v128) };
        let row_a_0 = u16x8_extend_low_u8x16(row_a_full);
        let row_a_1 = u16x8_extend_low_u8x16(u8x16_shuffle::<
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            0,
        >(row_a_full, row_a_full));

        let r2_b_ptr = unsafe { ptr2.add((i + 1) * stride2) };
        let row_b_full = unsafe { v128_load(r2_b_ptr as *const v128) };
        let row_b_0 = u16x8_extend_low_u8x16(row_b_full);
        let row_b_1 = u16x8_extend_low_u8x16(u8x16_shuffle::<
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            0,
        >(row_b_full, row_b_full));

        let val_top = i16x8_shr(
            i16x8_add(
                i16x8_mul(row_a_0 as v128, w_8_xf),
                i16x8_mul(row_a_1 as v128, w_xf),
            ),
            3,
        );
        let val_bot = i16x8_shr(
            i16x8_add(
                i16x8_mul(row_b_0 as v128, w_8_xf),
                i16x8_mul(row_b_1 as v128, w_xf),
            ),
            3,
        );

        let val = i16x8_shr(
            i16x8_add(i16x8_mul(val_top, w_8_yf), i16x8_mul(val_bot, w_yf)),
            3,
        );

        let r1 = unsafe { v128_load64_zero(ptr1.add(i * stride1) as *const u64) };
        let r1_16 = u16x8_extend_low_u8x16(r1);

        v[i] = i16x8_sub(r1_16 as v128, val);
    }

    unsafe { hadamard_butterfly(&mut v) };
    unsafe { transpose_8x8_i16(&mut v) };
    unsafe { hadamard_butterfly(&mut v) };

    let mut sum = u32x4_splat(0);
    for i in 0..8 {
        sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(i16x8_abs(v[i])));
    }

    (i32x4_extract_lane::<0>(sum)
        + i32x4_extract_lane::<1>(sum)
        + i32x4_extract_lane::<2>(sum)
        + i32x4_extract_lane::<3>(sum)) as u32
        / 2
}

#[cfg(not(target_arch = "wasm32"))]
fn get_sad_8x8(
    p1: &Plane<u8>,
    p2: &Plane<u8>,
    x1: usize,
    y1: usize,
    mv_col: i16,
    mv_row: i16,
) -> u32 {
    let x2 = x1 as isize + mv_col as isize;
    let y2 = y1 as isize + mv_row as isize;

    if x2 < 0
        || y2 < 0
        || (x2 as usize + 8) > p2.width().get()
        || (y2 as usize + 8) > p2.height().get()
    {
        return u32::MAX;
    }

    let mut sum = 0u32;
    for i in 0..8 {
        if let (Some(r1), Some(r2)) = (p1.row(y1 + i), p2.row(y2 as usize + i)) {
            for j in 0..8 {
                let v1 = r1[x1 + j] as i32;
                let v2 = r2[x2 as usize + j] as i32;
                sum += (v1 - v2).abs() as u32;
            }
        }
    }
    sum
}

#[cfg(target_arch = "wasm32")]
fn get_sad_8x8(
    p1: &Plane<u8>,
    p2: &Plane<u8>,
    x1: usize,
    y1: usize,
    mv_col: i16,
    mv_row: i16,
) -> u32 {
    let x2 = (x1 as isize + mv_col as isize) as usize;
    let y2 = (y1 as isize + mv_row as isize) as usize;

    let stride1 = p1.geometry().stride.get();
    let stride2 = p2.geometry().stride.get();
    let ptr1 = unsafe { p1.data().as_ptr().add(y1 * stride1 + x1) };
    let ptr2 = unsafe { p2.data().as_ptr().add(y2 * stride2 + x2) };

    let mut sum = u32x4_splat(0);

    for i in 0..8 {
        let r1 = unsafe { v128_load64_zero(ptr1.add(i * stride1) as *const u64) };
        let r2 = unsafe { v128_load64_zero(ptr2.add(i * stride2) as *const u64) };

        let diff = v128_or(u8x16_sub_sat(r1, r2), u8x16_sub_sat(r2, r1));

        let diff_16 = u16x8_extend_low_u8x16(diff);
        sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(diff_16 as v128));
    }

    (i32x4_extract_lane::<0>(sum)
        + i32x4_extract_lane::<1>(sum)
        + i32x4_extract_lane::<2>(sum)
        + i32x4_extract_lane::<3>(sum)) as u32
}

#[cfg(target_arch = "wasm32")]
unsafe fn hadamard_butterfly(v: &mut [v128; 8]) {
    for i in (0..8).step_by(2) {
        let a = v[i];
        let b = v[i + 1];
        v[i] = i16x8_add(a, b);
        v[i + 1] = i16x8_sub(a, b);
    }
    for i in (0..8).step_by(4) {
        for j in 0..2 {
            let a = v[i + j];
            let b = v[i + j + 2];
            v[i + j] = i16x8_add(a, b);
            v[i + j + 2] = i16x8_sub(a, b);
        }
    }
    for i in 0..4 {
        let a = v[i];
        let b = v[i + 4];
        v[i] = i16x8_add(a, b);
        v[i + 4] = i16x8_sub(a, b);
    }
}

#[cfg(target_arch = "wasm32")]
unsafe fn transpose_8x8_i16(v: &mut [v128; 8]) {
    let t0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[0], v[1]);
    let t1 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[2], v[3]);
    let t2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[4], v[5]);
    let t3 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[6], v[7]);

    let t4 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[0], v[1]);
    let t5 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[2], v[3]);
    let t6 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[4], v[5]);
    let t7 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[6], v[7]);

    let m0 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t0, t1);
    let m1 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t0, t1);
    let m2 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t2, t3);
    let m3 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t2, t3);

    let m4 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t4, t5);
    let m5 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t4, t5);
    let m6 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t6, t7);
    let m7 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t6, t7);

    v[0] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m0, m2);
    v[1] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m0, m2);
    v[2] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m1, m3);
    v[3] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m1, m3);

    v[4] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m4, m6);
    v[5] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m4, m6);
    v[6] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m5, m7);
    v[7] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m5, m7);
}

fn estimate_intra_costs(
    temp_plane: &mut Plane<u8>,
    frame: &Frame<u8>,
    _bit_depth: usize,
) -> Box<[u32]> {
    let plane = &frame.y_plane;
    let w_in_b = plane.width().get() / IMP_BLOCK_SIZE;
    let h_in_b = plane.height().get() / IMP_BLOCK_SIZE;

    let mut costs = Vec::with_capacity(w_in_b * h_in_b);

    for y in 0..h_in_b {
        for x in 0..w_in_b {
            predict_dc_intra(x, y, plane, temp_plane);
            let cost = get_satd_8x8(plane, temp_plane, x * IMP_BLOCK_SIZE, y * IMP_BLOCK_SIZE);
            costs.push(cost);
        }
    }
    costs.into_boxed_slice()
}

fn predict_dc_intra(bx: usize, by: usize, src: &Plane<u8>, dst: &mut Plane<u8>) {
    let x = bx * IMP_BLOCK_SIZE;
    let y = by * IMP_BLOCK_SIZE;

    let mut sum = 0u32;
    let mut count = 0u32;

    if x > 0 {
        for i in 0..IMP_BLOCK_SIZE {
            if let Some(row) = src.row(y + i) {
                sum += row[x - 1] as u32;
                count += 1;
            }
        }
    }
    if y > 0 {
        if let Some(row) = src.row(y - 1) {
            for i in 0..IMP_BLOCK_SIZE {
                sum += row[x + i] as u32;
                count += 1;
            }
        }
    }

    let dc = if count > 0 {
        (sum + (count >> 1)) / count
    } else {
        128
    } as u8;

    for i in 0..IMP_BLOCK_SIZE {
        if let Some(dst_row) = dst.row_mut(y + i) {
            for j in 0..IMP_BLOCK_SIZE {
                dst_row[x + j] = dc;
            }
        }
    }
}

fn estimate_importance_block_diff(frame1: &Frame<u8>, frame2: &Frame<u8>) -> f64 {
    let p1 = &frame1.y_plane;
    let p2 = &frame2.y_plane;
    let w = p1.width().get() / IMP_BLOCK_SIZE;
    let h = p1.height().get() / IMP_BLOCK_SIZE;

    let mut total_diff = 0u64;

    for y in 0..h {
        for x in 0..w {
            let sum1 = sum_block(p1, x * IMP_BLOCK_SIZE, y * IMP_BLOCK_SIZE);
            let sum2 = sum_block(p2, x * IMP_BLOCK_SIZE, y * IMP_BLOCK_SIZE);
            let count = (IMP_BLOCK_SIZE * IMP_BLOCK_SIZE) as i64;

            let mean1 = (sum1 + count / 2) / count;
            let mean2 = (sum2 + count / 2) / count;

            total_diff += (mean1 - mean2).abs() as u64;
        }
    }
    total_diff as f64 / (w * h) as f64
}

fn sum_block(p: &Plane<u8>, x: usize, y: usize) -> i64 {
    let mut sum = 0i64;
    for i in 0..IMP_BLOCK_SIZE {
        if let Some(row) = p.row(y + i) {
            for j in 0..IMP_BLOCK_SIZE {
                sum += row[x + j] as i64;
            }
        }
    }
    sum
}

fn hadamard_8x8(data: &mut [i16; 64]) -> u32 {
    for i in 0..8 {
        let off = i * 8;
        let a0 = data[off + 0];
        let a1 = data[off + 1];
        let a2 = data[off + 2];
        let a3 = data[off + 3];
        let a4 = data[off + 4];
        let a5 = data[off + 5];
        let a6 = data[off + 6];
        let a7 = data[off + 7];

        let b0 = a0 + a1;
        let b1 = a0 - a1;
        let b2 = a2 + a3;
        let b3 = a2 - a3;
        let b4 = a4 + a5;
        let b5 = a4 - a5;
        let b6 = a6 + a7;
        let b7 = a6 - a7;

        let c0 = b0 + b2;
        let c1 = b1 + b3;
        let c2 = b0 - b2;
        let c3 = b1 - b3;
        let c4 = b4 + b6;
        let c5 = b5 + b7;
        let c6 = b4 - b6;
        let c7 = b5 - b7;

        data[off + 0] = c0 + c4;
        data[off + 1] = c1 + c5;
        data[off + 2] = c2 + c6;
        data[off + 3] = c3 + c7;
        data[off + 4] = c0 - c4;
        data[off + 5] = c1 - c5;
        data[off + 6] = c2 - c6;
        data[off + 7] = c3 - c7;
    }

    for i in 0..8 {
        let a0 = data[0 * 8 + i];
        let a1 = data[1 * 8 + i];
        let a2 = data[2 * 8 + i];
        let a3 = data[3 * 8 + i];
        let a4 = data[4 * 8 + i];
        let a5 = data[5 * 8 + i];
        let a6 = data[6 * 8 + i];
        let a7 = data[7 * 8 + i];

        let b0 = a0 + a1;
        let b1 = a0 - a1;
        let b2 = a2 + a3;
        let b3 = a2 - a3;
        let b4 = a4 + a5;
        let b5 = a4 - a5;
        let b6 = a6 + a7;
        let b7 = a6 - a7;

        let c0 = b0 + b2;
        let c1 = b1 + b3;
        let c2 = b0 - b2;
        let c3 = b1 - b3;
        let c4 = b4 + b6;
        let c5 = b5 + b7;
        let c6 = b4 - b6;
        let c7 = b5 - b7;

        data[0 * 8 + i] = c0 + c4;
        data[1 * 8 + i] = c1 + c5;
        data[2 * 8 + i] = c2 + c6;
        data[3 * 8 + i] = c3 + c7;
        data[4 * 8 + i] = c0 - c4;
        data[5 * 8 + i] = c1 - c5;
        data[6 * 8 + i] = c2 - c6;
        data[7 * 8 + i] = c3 - c7;
    }

    data.iter().map(|&x| x.abs() as u32).sum::<u32>() / 2
}
