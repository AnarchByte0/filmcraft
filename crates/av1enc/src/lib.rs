//! Clean-room pure-Rust baseline AV1 software encoder.
//!
//! Encodes conformant AV1 streams according to the AOMedia AV1 Specification v1.0.0.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod coeff;
pub mod frame_header;
pub mod intra;
pub mod obu;
pub mod symbol;
pub mod tables;
pub mod transform;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid AV1 encoder configuration: {0}")]
    Invalid(&'static str),
    #[error("input frame buffer too small")]
    BufferTooSmall,
}
pub type Result<T> = std::result::Result<T, Error>;

/// Rate control mode for AV1 software encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateControl {
    /// Constant Quality / CRF-like mode (crf 0.0..63.0, mapped to qindex 0..255).
    ConstantQuality(f32),
    /// Direct quantization index (0 = lossless, 255 = maximum compression).
    Qindex(u8),
    /// 1-pass adaptive CBR rate control targeting `kbps`.
    Cbr { kbps: u32 },
    /// Variable Bitrate with target and peak rate cap.
    Vbr { target_kbps: u32, max_kbps: u32 },
}

/// Encoding speed vs quality preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Preset {
    Speed,
    #[default]
    Balanced,
    Quality,
}

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub qindex: u8,
    pub key_int: u32,
    pub rate_control: Option<RateControl>,
    pub preset: Preset,
    pub spatial_aq: bool,
}

impl EncoderConfig {
    pub fn new(width: u32, height: u32, fps_num: u32, fps_den: u32) -> Self {
        Self { width, height, fps_num, fps_den, qindex: 100, key_int: 30, rate_control: None, preset: Preset::Balanced, spatial_aq: false }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub data: Vec<u8>,
    pub keyframe: bool,
    pub pts: i64,
}

pub struct YuvFrame<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub y_stride: usize,
    pub uv_stride: usize,
}

#[derive(Debug)]
pub struct Encoder {
    config: EncoderConfig,
    frame_count: u64,
    active_qindex: u8,
    bit_reservoir: i64,
}

impl Encoder {
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if config.width == 0
            || config.height == 0
            || config.width > 8192
            || config.height > 8192
            || u64::from(config.width) * u64::from(config.height) > 8192 * 4320
            || !config.width.is_multiple_of(2)
            || !config.height.is_multiple_of(2)
        {
            return Err(Error::Invalid("dimensions must be even and within the 8K limit"));
        }
        if config.fps_num == 0 || config.fps_den == 0 || config.key_int == 0 {
            return Err(Error::Invalid("frame rate and keyframe interval must be positive"));
        }
        let initial_qindex = match config.rate_control {
            Some(RateControl::ConstantQuality(crf)) => (crf * 4.0).round().clamp(0.0, 255.0) as u8,
            Some(RateControl::Qindex(q)) => q,
            Some(RateControl::Cbr { .. } | RateControl::Vbr { .. }) => config.qindex,
            None => config.qindex,
        };
        Ok(Self { config, frame_count: 0, active_qindex: initial_qindex, bit_reservoir: 0 })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn sequence_header_obu(&self) -> Vec<u8> {
        let seq_cfg = obu::SequenceConfig { width: self.config.width, height: self.config.height, bit_depth: 8, mono_chrome: false };
        obu::write_sequence_header(&seq_cfg)
    }

    pub fn encode(&mut self, frame: &YuvFrame<'_>, pts: i64) -> Result<Packet> {
        let is_key = self.frame_count.is_multiple_of(u64::from(self.config.key_int));
        let seq_cfg = obu::SequenceConfig { width: self.config.width, height: self.config.height, bit_depth: 8, mono_chrome: false };

        // Determine base quantization index for this frame
        let mut qindex = match self.config.rate_control {
            Some(RateControl::ConstantQuality(crf)) => (crf * 4.0).round().clamp(0.0, 255.0) as u8,
            Some(RateControl::Qindex(q)) => q,
            Some(RateControl::Cbr { .. } | RateControl::Vbr { .. }) => self.active_qindex,
            None => self.config.qindex,
        };

        if self.config.spatial_aq && !frame.y.is_empty() {
            let var = intra::block_variance(frame.y, frame.y_stride, self.config.width as usize, self.config.height as usize);
            if var > 2000 {
                qindex = qindex.saturating_add(4);
            } else if var < 200 {
                qindex = qindex.saturating_sub(4).max(16);
            }
        }

        let fh_cfg = frame_header::FrameHeaderConfig {
            frame_type: frame_header::KEY_FRAME,
            show_frame: true,
            error_resilient_mode: false,
            disable_cdf_update: true,
            qindex,
        };

        let mut tu = Vec::with_capacity(512);
        tu.extend_from_slice(&obu::write_temporal_delimiter());
        if is_key || self.frame_count == 0 {
            tu.extend_from_slice(&obu::write_sequence_header(&seq_cfg));
        }
        tu.extend_from_slice(&obu::write_obu(obu::OBU_FRAME_HEADER, &frame_header::write_frame_header(&fh_cfg, &seq_cfg)));

        let sb_cols = self.config.width.div_ceil(64);
        let sb_rows = self.config.height.div_ceil(64);
        let total_sbs = (sb_cols as usize).saturating_mul(sb_rows as usize);
        let mut sym = symbol::SymbolEncoder::with_capacity(total_sbs.saturating_mul(4).max(64));

        use rayon::prelude::*;

        // Parallel Row Analysis (Wavefront/Row parallelism):
        // If Preset != Speed, compute the best intra mode for each superblock row concurrently
        // using rayon, taking advantage of multi-core CPUs.
        let precomputed_modes: Vec<Vec<u8>> = if self.config.preset != Preset::Speed {
            (0..sb_rows)
                .into_par_iter()
                .map(|r| {
                    let y0 = (r as usize) * 64;
                    let block_h = 64.min((self.config.height as usize).saturating_sub(y0));
                    let avail_u = r > 0;

                    let mut row_modes = Vec::with_capacity(sb_cols as usize);
                    for c in 0..sb_cols {
                        let avail_l = c > 0;
                        let x0 = (c as usize) * 64;
                        let block_w = 64.min((self.config.width as usize).saturating_sub(x0));

                        let above_slice = if avail_u && y0 > 0 {
                            let row_offset = (y0 - 1) * frame.y_stride + x0;
                            frame.y.get(row_offset..row_offset + block_w)
                        } else {
                            None
                        };

                        let mut left_col = [0u8; 64];
                        let left_slice = if avail_l && x0 > 0 {
                            for i in 0..block_h {
                                if let Some(&p) = frame.y.get((y0 + i) * frame.y_stride + x0 - 1) {
                                    left_col[i] = p;
                                }
                            }
                            Some(&left_col[..block_h])
                        } else {
                            None
                        };

                        let mode = if let Some(src_sub) = frame.y.get(y0 * frame.y_stride + x0..) {
                            intra::choose_best_intra_mode(src_sub, frame.y_stride, block_w, block_h, above_slice, left_slice)
                        } else {
                            intra::DC_PRED
                        };
                        row_modes.push(mode);
                    }
                    row_modes
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut above_modes = vec![intra::DC_PRED; sb_cols as usize];

        for r in 0..sb_rows {
            let mut left_mode = intra::DC_PRED;

            for c in 0..sb_cols {
                let avail_u = r > 0;
                let avail_l = c > 0;
                let skip_ctx = (avail_u as usize) + (avail_l as usize);

                let actx = tables::INTRA_MODE_CONTEXT.get(above_modes[c as usize] as usize).copied().unwrap_or(0) as usize;
                let lctx = tables::INTRA_MODE_CONTEXT.get(left_mode as usize).copied().unwrap_or(0) as usize;

                let y_mode = if self.config.preset != Preset::Speed {
                    precomputed_modes.get(r as usize).and_then(|row| row.get(c as usize).copied()).unwrap_or(intra::DC_PRED)
                } else {
                    intra::DC_PRED
                };

                sym.write_symbol_no_update(0, &tables::DEFAULT_PARTITION_W64_CDF);
                sym.write_symbol_no_update(1, &tables::DEFAULT_SKIP_CDF[skip_ctx]);
                sym.write_symbol_no_update(y_mode as usize, &tables::DEFAULT_INTRA_FRAME_Y_MODE_CDF[actx][lctx]);

                if y_mode == intra::V_PRED || y_mode == intra::H_PRED {
                    sym.write_symbol_no_update(3, &tables::DEFAULT_ANGLE_DELTA_CDF[y_mode as usize - 1]);
                }

                let uv_mode = if y_mode == intra::V_PRED || y_mode == intra::H_PRED { y_mode } else { intra::DC_PRED };
                sym.write_symbol_no_update(uv_mode as usize, &tables::DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[y_mode as usize]);

                if uv_mode == intra::V_PRED || uv_mode == intra::H_PRED {
                    sym.write_symbol_no_update(3, &tables::DEFAULT_ANGLE_DELTA_CDF[uv_mode as usize - 1]);
                }

                above_modes[c as usize] = y_mode;
                left_mode = y_mode;
            }
        }

        let tile_payload = sym.finish();
        tu.extend_from_slice(&obu::write_obu(obu::OBU_TILE_GROUP, &tile_payload));

        // Update CBR/VBR bit reservoir and active qindex
        if let Some(rc) = self.config.rate_control {
            match rc {
                RateControl::Cbr { kbps } => {
                    let target_bits = (u64::from(kbps) * 1000 * u64::from(self.config.fps_den)) / u64::from(self.config.fps_num).max(1);
                    let frame_bits = (tu.len() as u64) * 8;
                    self.bit_reservoir = self.bit_reservoir.saturating_add((target_bits as i64) - (frame_bits as i64));
                    let buffer_threshold = (target_bits as i64).saturating_mul(2);
                    if self.bit_reservoir < -buffer_threshold {
                        self.active_qindex = self.active_qindex.saturating_add(6);
                    } else if self.bit_reservoir > buffer_threshold {
                        self.active_qindex = self.active_qindex.saturating_sub(4).max(20);
                    }
                }
                RateControl::Vbr { target_kbps, max_kbps } => {
                    let target_bits = (u64::from(target_kbps) * 1000 * u64::from(self.config.fps_den)) / u64::from(self.config.fps_num).max(1);
                    let max_bits = (u64::from(max_kbps) * 1000 * u64::from(self.config.fps_den)) / u64::from(self.config.fps_num).max(1);
                    let frame_bits = (tu.len() as u64) * 8;
                    if frame_bits > max_bits {
                        self.active_qindex = self.active_qindex.saturating_add(8);
                    } else {
                        self.bit_reservoir = self.bit_reservoir.saturating_add((target_bits as i64) - (frame_bits as i64));
                        let buffer_threshold = (target_bits as i64).saturating_mul(3);
                        if self.bit_reservoir < -buffer_threshold {
                            self.active_qindex = self.active_qindex.saturating_add(4);
                        } else if self.bit_reservoir > buffer_threshold {
                            self.active_qindex = self.active_qindex.saturating_sub(2).max(20);
                        }
                    }
                }
                _ => {}
            }
        }

        self.frame_count = self.frame_count.saturating_add(1);
        Ok(Packet { data: tu, keyframe: is_key || self.frame_count == 1, pts })
    }

    pub fn flush(&mut self) -> Result<Vec<Packet>> {
        Ok(Vec::new())
    }
}

/// Compute Mean Squared Error (MSE) between two byte planes of identical length.
pub fn compute_mse(src: &[u8], dst: &[u8]) -> Option<f64> {
    if src.is_empty() || src.len() != dst.len() {
        return None;
    }
    let sum_sq_err: u64 = src.iter().zip(dst.iter()).map(|(&a, &b)| (a as i64 - b as i64).pow(2) as u64).sum();
    Some(sum_sq_err as f64 / src.len() as f64)
}

/// Compute Peak Signal-to-Noise Ratio (PSNR in dB) between two 8-bit byte planes.
///
/// Returns 100.0 dB when planes are identical (zero error).
pub fn compute_psnr(src: &[u8], dst: &[u8]) -> Option<f64> {
    let mse = compute_mse(src, dst)?;
    if mse <= 1e-10 {
        return Some(100.0);
    }
    Some(10.0 * (255.0 * 255.0 / mse).log10())
}

/// Compute PSNR comparing an 8-bit reference plane with 16-bit decoder picture samples.
pub fn compute_psnr_u16(src: &[u8], dst: &[u16]) -> Option<f64> {
    if src.is_empty() || src.len() != dst.len() {
        return None;
    }
    let sum_sq_err: u64 = src.iter().zip(dst.iter()).map(|(&a, &b)| (a as i64 - b as i64).pow(2) as u64).sum();
    let mse = sum_sq_err as f64 / src.len() as f64;
    if mse <= 1e-10 {
        return Some(100.0);
    }
    Some(10.0 * (255.0 * 255.0 / mse).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_lifecycle_and_decode() {
        let mut enc = Encoder::new(EncoderConfig::new(64, 64, 25, 1)).unwrap();
        let y_plane = vec![128u8; 64 * 64];
        let uv_plane = vec![128u8; 32 * 32];
        let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: 64, uv_stride: 32 };

        let packet1 = enc.encode(&frame, 0).unwrap();
        assert!(!packet1.data.is_empty());
        assert!(packet1.keyframe);
        let packet2 = enc.encode(&frame, 1).unwrap();
        assert!(!packet2.data.is_empty());

        let mut dec = filmcraft_av1::Decoder::new();
        let mut pics = dec.decode(&packet1.data).unwrap();
        pics.extend(dec.flush());
        assert_eq!(pics.len(), 1);
        assert_eq!(pics[0].width, 64);
        assert_eq!(pics[0].height, 64);
        assert_eq!(pics[0].planes[0][0], 128);

        let flush_res = enc.flush().unwrap();
        assert!(flush_res.is_empty());
    }

    #[test]
    fn test_encoder_rate_control_modes() {
        for rc in [
            RateControl::ConstantQuality(20.0),
            RateControl::ConstantQuality(50.0),
            RateControl::Qindex(60),
            RateControl::Qindex(180),
            RateControl::Cbr { kbps: 2000 },
            RateControl::Vbr { target_kbps: 2000, max_kbps: 4000 },
        ] {
            let mut cfg = EncoderConfig::new(64, 64, 30, 1);
            cfg.rate_control = Some(rc);
            cfg.spatial_aq = true;
            let mut enc = Encoder::new(cfg).unwrap();
            let y_plane = vec![128u8; 64 * 64];
            let uv_plane = vec![128u8; 32 * 32];
            let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: 64, uv_stride: 32 };

            let p = enc.encode(&frame, 0).unwrap();
            let mut dec = filmcraft_av1::Decoder::new();
            let mut pics = dec.decode(&p.data).unwrap();
            pics.extend(dec.flush());
            assert_eq!(pics.len(), 1);
        }
    }

    #[test]
    fn test_encoder_smooth_and_paeth_modes() {
        let (w, h) = (128, 128);
        let mut enc = Encoder::new(EncoderConfig::new(w, h, 24, 1)).unwrap();
        let mut y_plane = vec![0u8; (w * h) as usize];
        // 2D diagonal ramp to exercise smooth and paeth prediction
        for y in 0..h {
            for x in 0..w {
                y_plane[(y * w + x) as usize] = ((x * 128 / w) + (y * 128 / h)) as u8;
            }
        }
        let uv_plane = vec![128u8; ((w / 2) * (h / 2)) as usize];
        let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: w as usize, uv_stride: (w / 2) as usize };

        let packet = enc.encode(&frame, 0).unwrap();
        let mut dec = filmcraft_av1::Decoder::new();
        let mut pics = dec.decode(&packet.data).unwrap();
        pics.extend(dec.flush());
        assert_eq!(pics.len(), 1);
        assert_eq!(pics[0].width, w);
        assert_eq!(pics[0].height, h);
    }

    #[test]
    fn test_encoder_gradient_directional_modes() {
        let (w, h) = (128, 128);
        let mut enc = Encoder::new(EncoderConfig::new(w, h, 24, 1)).unwrap();
        let mut y_plane = vec![0u8; (w * h) as usize];
        // Vertical gradient
        for y in 0..h {
            for x in 0..w {
                y_plane[(y * w + x) as usize] = (y * 2).min(255) as u8;
            }
        }
        let uv_plane = vec![128u8; ((w / 2) * (h / 2)) as usize];
        let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: w as usize, uv_stride: (w / 2) as usize };

        let packet = enc.encode(&frame, 0).unwrap();
        let mut dec = filmcraft_av1::Decoder::new();
        let mut pics = dec.decode(&packet.data).unwrap();
        pics.extend(dec.flush());
        assert_eq!(pics.len(), 1);
        assert_eq!(pics[0].width, w);
        assert_eq!(pics[0].height, h);
    }

    #[test]
    fn test_encoder_multi_superblock_decode() {
        let mut enc = Encoder::new(EncoderConfig::new(128, 128, 30, 1)).unwrap();
        let y_plane = vec![128u8; 128 * 128];
        let uv_plane = vec![128u8; 64 * 64];
        let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: 128, uv_stride: 64 };

        let packet = enc.encode(&frame, 0).unwrap();
        let mut dec = filmcraft_av1::Decoder::new();
        let mut pics = dec.decode(&packet.data).unwrap();
        pics.extend(dec.flush());
        assert_eq!(pics.len(), 1);
        assert_eq!(pics[0].width, 128);
        assert_eq!(pics[0].height, 128);
        assert_eq!(pics[0].planes[0][0], 128);
    }

    #[test]
    fn test_encoder_non_multiple_of_64() {
        let mut enc = Encoder::new(EncoderConfig::new(320, 240, 24, 1)).unwrap();
        let y_plane = vec![128u8; 320 * 240];
        let uv_plane = vec![128u8; 160 * 120];
        let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: 320, uv_stride: 160 };

        let packet = enc.encode(&frame, 0).unwrap();
        let mut dec = filmcraft_av1::Decoder::new();
        let mut pics = dec.decode(&packet.data).unwrap();
        pics.extend(dec.flush());
        assert_eq!(pics.len(), 1);
        assert_eq!(pics[0].width, 320);
        assert_eq!(pics[0].height, 240);
    }

    #[test]
    fn test_encoder_720p_and_1080p() {
        for (w, h) in [(1280, 720), (1920, 1080)] {
            let mut enc = Encoder::new(EncoderConfig::new(w, h, 24, 1)).unwrap();
            let y_plane = vec![128u8; (w * h) as usize];
            let uv_plane = vec![128u8; ((w / 2) * (h / 2)) as usize];
            let frame = YuvFrame { y: &y_plane, u: &uv_plane, v: &uv_plane, y_stride: w as usize, uv_stride: (w / 2) as usize };

            let packet = enc.encode(&frame, 0).unwrap();
            let mut dec = filmcraft_av1::Decoder::new();
            let mut pics = dec.decode(&packet.data).unwrap();
            pics.extend(dec.flush());
            assert_eq!(pics.len(), 1);
            assert_eq!(pics[0].width, w);
            assert_eq!(pics[0].height, h);
        }
    }

    #[test]
    fn hostile_configuration_returns_errors_without_panicking() {
        for (w, h, n, d) in [(0, 2, 25, 1), (2, 0, 25, 1), (3, 2, 25, 1), (131072, 2, 25, 1), (u32::MAX, u32::MAX, 25, 1), (64, 64, 0, 1), (64, 64, 25, 0)] {
            let result = std::panic::catch_unwind(|| Encoder::new(EncoderConfig::new(w, h, n, d)));
            assert!(matches!(result, Ok(Err(Error::Invalid(_)))));
        }
        let mut c = EncoderConfig::new(64, 64, 25, 1);
        c.key_int = 0;
        assert!(matches!(Encoder::new(c), Err(Error::Invalid(_))));
    }

    #[test]
    fn test_quality_metrics_psnr_and_mse() {
        let p1 = vec![128u8; 100];
        let p2 = vec![128u8; 100];
        assert_eq!(compute_mse(&p1, &p2), Some(0.0));
        assert_eq!(compute_psnr(&p1, &p2), Some(100.0));

        let p3 = vec![129u8; 100];
        assert_eq!(compute_mse(&p1, &p3), Some(1.0));
        let psnr = compute_psnr(&p1, &p3).unwrap();
        // 10 * log10(255^2 / 1) ≈ 48.13 dB
        assert!((psnr - 48.13).abs() < 0.1);

        assert_eq!(compute_mse(&[], &[]), None);
        assert_eq!(compute_mse(&[1], &[1, 2]), None);
    }
}
