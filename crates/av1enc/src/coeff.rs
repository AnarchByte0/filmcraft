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

use crate::symbol::SymbolEncoder;

/// Default CDF for transform block skip (`txb_skip`, Section 9.3).
pub static DEFAULT_TXB_SKIP_CDF: [u16; 3] = [31849, 32768, 0];

/// Default CDF for end-of-block pointer for 4x4 blocks (multisize 0).
pub static DEFAULT_EOB_PT_16_CDF: [u16; 6] = [840, 1039, 1980, 4895, 32768, 0];

/// Default CDF for eob extra bits.
pub static DEFAULT_EOB_EXTRA_CDF: [u16; 3] = [16384, 32768, 0];

/// Write a transform block skip flag (`txb_skip`).
/// `all_zero = true` writes symbol 1, indicating all coefficients are zero.
pub fn write_txb_skip(sym: &mut SymbolEncoder, all_zero: bool) {
    sym.write_symbol_no_update(if all_zero { 1 } else { 0 }, &DEFAULT_TXB_SKIP_CDF);
}

/// Write EOB syntax for a 4x4 transform block into the symbol stream.
pub fn write_eob_4x4(sym: &mut SymbolEncoder, eob: usize) {
    debug_assert!((1..=16).contains(&eob));
    let (pt_sym, extra_opt, lit_val, lit_bits) = map_eob_to_syntax(eob);
    sym.write_symbol_no_update(pt_sym, &DEFAULT_EOB_PT_16_CDF);
    if let Some(extra_sym) = extra_opt {
        sym.write_symbol_no_update(extra_sym, &DEFAULT_EOB_EXTRA_CDF);
    }
    for i in (0..lit_bits).rev() {
        let bit = ((lit_val >> i) & 1) == 1;
        sym.write_bool(bit);
    }
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

    #[test]
    fn test_write_txb_skip_and_eob() {
        let mut sym = SymbolEncoder::new();
        write_txb_skip(&mut sym, false);
        write_eob_4x4(&mut sym, 4);
        let bytes = sym.finish();
        assert!(!bytes.is_empty());
    }
}
