//! Clean-room transform coefficient entropy encoding for AV1 tiles (spec 5.11.37 / 7.13.3).

/// 4x4 zigzag scan order (spec Table 7.13.1).
pub const SCAN_4X4: [usize; 16] = [
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
];

/// Inverse scan mapping: gives the scan index (0..15) for a raster position (0..15).
pub const INV_SCAN_4X4: [usize; 16] = [
    0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15,
];

/// End-of-block position mapping for 4x4 blocks (multisize 0, eob in 1..=16).
/// Returns (eob_pt_sym, eob_extra_opt, literal_bits).
#[inline(always)]
pub fn map_eob_to_syntax(eob: usize) -> (usize, Option<usize>, u32, usize) {
    debug_assert!((1..=16).contains(&eob));
    match eob {
        1 => (0, None, 0, 0),
        2 => (1, None, 0, 0),
        3 => (2, Some(0), 0, 0),
        4 => (2, Some(1), 0, 0),
        5 => (3, Some(0), 0, 1),
        6 => (3, Some(0), 1, 1),
        7 => (3, Some(1), 0, 1),
        8 => (3, Some(1), 1, 1),
        9 => (4, Some(0), 0, 2),
        10 => (4, Some(0), 1, 2),
        11 => (4, Some(0), 2, 2),
        12 => (4, Some(0), 3, 2),
        13 => (4, Some(1), 0, 2),
        14 => (4, Some(1), 1, 2),
        15 => (4, Some(1), 2, 2),
        _ => (4, Some(1), 3, 2), // 16
    }
}

/// Find the End-of-Block (EOB) index in scan order for a 4x4 quantized block.
/// Returns 0 if all coefficients are zero.
#[inline(always)]
pub fn find_eob_4x4(quant: &[i16; 16]) -> usize {
    for i in (0..16).rev() {
        if quant[SCAN_4X4[i]] != 0 {
            return i + 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_eob_and_scan_symmetry() {
        let mut block = [0i16; 16];
        assert_eq!(find_eob_4x4(&block), 0);

        block[0] = 5; // DC coeff is scan pos 0
        assert_eq!(find_eob_4x4(&block), 1);

        block[15] = -2; // Last coeff in scan order is raster 15
        assert_eq!(find_eob_4x4(&block), 16);

        // Verify scan inversion
        for i in 0..16 {
            assert_eq!(INV_SCAN_4X4[SCAN_4X4[i]], i);
            assert_eq!(SCAN_4X4[INV_SCAN_4X4[i]], i);
        }
    }

    #[test]
    fn test_eob_syntax_mapping() {
        for eob in 1..=16 {
            let (pt, extra, lit_val, lit_bits) = map_eob_to_syntax(eob);
            let eob_pt = pt + 1;
            let mut decoded_eob = if eob_pt < 2 { eob_pt } else { (1 << (eob_pt - 2)) + 1 };
            let eob_shift = eob_pt as i32 - 3;
            if eob_shift >= 0 {
                if extra == Some(1) {
                    decoded_eob += 1 << eob_shift;
                }
                if lit_bits > 0 {
                    decoded_eob += (lit_val as usize) & ((1 << lit_bits) - 1);
                }
            }
            assert_eq!(decoded_eob, eob, "failed for eob {eob}");
        }
    }
}
