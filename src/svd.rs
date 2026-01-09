use std::borrow::Cow;
use std::fs::File;
use std::str;

use memmap2::Mmap;
use nalgebra::{DMatrix, SymmetricEigen};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rand::Rng;

use crate::gfcore::{BedSnpIter, VcfSnpIter};

fn matmul_a_b(a: &[f64], m: usize, n: usize, b: &[f64], l: usize, block_rows: usize) -> Vec<f64> {
    let mut y = vec![0.0; m * l];
    let block = block_rows.max(1).min(m);
    let mut i = 0usize;
    while i < m {
        let i_end = (i + block).min(m);
        for row in i..i_end {
            let a_row = &a[row * n..(row + 1) * n];
            let y_row = &mut y[row * l..(row + 1) * l];
            for k in 0..n {
                let aik = a_row[k];
                if aik != 0.0 {
                    let b_row = &b[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += aik * b_row[j];
                    }
                }
            }
        }
        i = i_end;
    }
    y
}

fn matmul_at_b(a: &[f64], m: usize, n: usize, b: &[f64], l: usize, block_rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * l];
    let block = block_rows.max(1).min(m);
    let mut i = 0usize;
    while i < m {
        let i_end = (i + block).min(m);
        for row in i..i_end {
            let a_row = &a[row * n..(row + 1) * n];
            let b_row = &b[row * l..(row + 1) * l];
            for k in 0..n {
                let aik = a_row[k];
                if aik != 0.0 {
                    let out_row = &mut out[k * l..(k + 1) * l];
                    for j in 0..l {
                        out_row[j] += aik * b_row[j];
                    }
                }
            }
        }
        i = i_end;
    }
    out
}

fn matmul_qt_a(q: &[f64], m: usize, l: usize, a: &[f64], n: usize, block_rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * n];
    let block = block_rows.max(1).min(m);
    let mut i = 0usize;
    while i < m {
        let i_end = (i + block).min(m);
        for row in i..i_end {
            let q_row = &q[row * l..(row + 1) * l];
            let a_row = &a[row * n..(row + 1) * n];
            for j in 0..l {
                let qij = q_row[j];
                if qij != 0.0 {
                    let out_row = &mut out[j * n..(j + 1) * n];
                    for k in 0..n {
                        out_row[k] += qij * a_row[k];
                    }
                }
            }
        }
        i = i_end;
    }
    out
}

fn gram_b(b: &[f64], l: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; l * l];
    for i in 0..l {
        let row_i = &b[i * n..(i + 1) * n];
        for j in i..l {
            let row_j = &b[j * n..(j + 1) * n];
            let mut sum = 0.0;
            for k in 0..n {
                sum += row_i[k] * row_j[k];
            }
            c[i * l + j] = sum;
            c[j * l + i] = sum;
        }
    }
    c
}

fn dmatrix_to_rowmajor(mat: &DMatrix<f64>) -> Vec<f64> {
    let (m, n) = mat.shape();
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = mat[(i, j)];
        }
    }
    out
}

fn transpose_rowmajor(src: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut dst = vec![0.0; rows * cols];
    for i in 0..rows {
        let row = &src[i * cols..(i + 1) * cols];
        for j in 0..cols {
            dst[j * rows + i] = row[j];
        }
    }
    dst
}

trait GenoIter {
    fn n_samples(&self) -> usize;
    fn next_row(&mut self) -> Option<Vec<f32>>;
}

impl GenoIter for BedSnpIter {
    fn n_samples(&self) -> usize {
        BedSnpIter::n_samples(self)
    }

    fn next_row(&mut self) -> Option<Vec<f32>> {
        BedSnpIter::next_snp(self).map(|(row, _)| row)
    }
}

impl GenoIter for VcfSnpIter {
    fn n_samples(&self) -> usize {
        VcfSnpIter::n_samples(self)
    }

    fn next_row(&mut self) -> Option<Vec<f32>> {
        VcfSnpIter::next_snp(self).map(|(row, _)| row)
    }
}

#[inline]
fn row_mean_invstd(row: &[f32], center: bool, scale: bool) -> (f64, f64) {
    if !center && !scale {
        return (0.0, 1.0);
    }
    let sum: f64 = row.iter().map(|&v| v as f64).sum();
    let mean = sum / row.len() as f64;
    if !scale {
        return (mean, 1.0);
    }
    let p = (mean / 2.0).clamp(0.0, 1.0);
    let var = 2.0 * p * (1.0 - p);
    if var <= 0.0 {
        return (mean, 0.0);
    }
    (mean, 1.0 / var.sqrt())
}

fn compute_y_with_iter<I: GenoIter>(
    mut it: I,
    omega: &[f64],
    l: usize,
    center: bool,
    scale: bool,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    if omega.len() != n * l {
        return Err("omega size mismatch".into());
    }
    let mut y: Vec<f64> = Vec::new();
    while let Some(row) = it.next_row() {
        let base = y.len();
        y.resize(base + l, 0.0);
        let y_row = &mut y[base..base + l];
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    let omega_row = &omega[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * omega_row[j];
                    }
                }
            }
        } else {
            let (mean, inv_std) = row_mean_invstd(&row, center, scale);
            if scale && inv_std == 0.0 {
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    let omega_row = &omega[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * omega_row[j];
                    }
                }
            }
        }
    }
    Ok((y, n))
}

fn compute_at_y_with_iter<I: GenoIter>(
    mut it: I,
    y: &[f64],
    l: usize,
    center: bool,
    scale: bool,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    let mut z = vec![0.0f64; n * l];
    let mut row_idx: usize = 0;
    while let Some(row) = it.next_row() {
        let y_row = y
            .get(row_idx * l..(row_idx + 1) * l)
            .ok_or_else(|| "row count mismatch in A^T*Y".to_string())?;
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    let z_row = &mut z[k * l..(k + 1) * l];
                    for j in 0..l {
                        z_row[j] += val * y_row[j];
                    }
                }
            }
        } else {
            let (mean, inv_std) = row_mean_invstd(&row, center, scale);
            if scale && inv_std == 0.0 {
                row_idx += 1;
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    let z_row = &mut z[k * l..(k + 1) * l];
                    for j in 0..l {
                        z_row[j] += val * y_row[j];
                    }
                }
            }
        }
        row_idx += 1;
    }
    if row_idx * l != y.len() {
        return Err("row count mismatch in A^T*Y".into());
    }
    Ok((z, n))
}

fn compute_a_z_with_iter<I: GenoIter>(
    mut it: I,
    z: &[f64],
    l: usize,
    center: bool,
    scale: bool,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    if z.len() != n * l {
        return Err("Z size mismatch".into());
    }
    let mut y: Vec<f64> = Vec::new();
    while let Some(row) = it.next_row() {
        let base = y.len();
        y.resize(base + l, 0.0);
        let y_row = &mut y[base..base + l];
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    let z_row = &z[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * z_row[j];
                    }
                }
            }
        } else {
            let (mean, inv_std) = row_mean_invstd(&row, center, scale);
            if scale && inv_std == 0.0 {
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    let z_row = &z[k * l..(k + 1) * l];
                    for j in 0..l {
                        y_row[j] += val * z_row[j];
                    }
                }
            }
        }
    }
    Ok((y, n))
}

fn compute_qt_a_with_iter<I: GenoIter>(
    mut it: I,
    q: &[f64],
    l: usize,
    center: bool,
    scale: bool,
) -> Result<(Vec<f64>, usize), String> {
    let n = it.n_samples();
    let mut b = vec![0.0f64; l * n];
    let mut row_idx: usize = 0;
    while let Some(row) = it.next_row() {
        let q_row = q
            .get(row_idx * l..(row_idx + 1) * l)
            .ok_or_else(|| "row count mismatch in Q^T*A".to_string())?;
        if !center && !scale {
            for k in 0..n {
                let val = row[k] as f64;
                if val != 0.0 {
                    for j in 0..l {
                        b[j * n + k] += q_row[j] * val;
                    }
                }
            }
        } else {
            let (mean, inv_std) = row_mean_invstd(&row, center, scale);
            if scale && inv_std == 0.0 {
                row_idx += 1;
                continue;
            }
            for k in 0..n {
                let mut val = row[k] as f64 - mean;
                if scale {
                    val *= inv_std;
                }
                if val != 0.0 {
                    for j in 0..l {
                        b[j * n + k] += q_row[j] * val;
                    }
                }
            }
        }
        row_idx += 1;
    }
    if row_idx * l != q.len() {
        return Err("row count mismatch in Q^T*A".into());
    }
    Ok((b, n))
}

fn randomized_svd_streaming<I, F>(
    mut make_iter: F,
    k: usize,
    oversample: usize,
    n_iter: usize,
    center: bool,
    scale: bool,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize), String>
where
    I: GenoIter,
    F: FnMut() -> Result<I, String>,
{
    if k == 0 {
        return Err("k must be > 0".into());
    }
    let l = k.checked_add(oversample).ok_or("k + oversample overflow")?;

    let iter0 = make_iter()?;
    let n = iter0.n_samples();
    if n == 0 {
        return Err("no samples in genotype input".into());
    }
    if l == 0 || l > n {
        return Err("k + oversample must be in 1..=n_samples".into());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0; n * l];
    for i in 0..n {
        let row = &mut omega[i * l..(i + 1) * l];
        for j in 0..l {
            row[j] = rng.sample(StandardNormal);
        }
    }

    let (mut y, n_y) = compute_y_with_iter(iter0, &omega, l, center, scale)?;
    if n_y != n {
        return Err("sample count mismatch".into());
    }
    if y.len() % l != 0 {
        return Err("Y size mismatch".into());
    }
    let mut m_eff = y.len() / l;
    if m_eff == 0 {
        return Err("no variants left after QC".into());
    }

    for _ in 0..n_iter {
        let iter_z = make_iter()?;
        let (z, n_z) = compute_at_y_with_iter(iter_z, &y, l, center, scale)?;
        if n_z != n {
            return Err("sample count mismatch".into());
        }
        let iter_y = make_iter()?;
        let (y_new, n_y2) = compute_a_z_with_iter(iter_y, &z, l, center, scale)?;
        if n_y2 != n {
            return Err("sample count mismatch".into());
        }
        if y_new.len() % l != 0 {
            return Err("Y size mismatch".into());
        }
        let m_new = y_new.len() / l;
        if m_new != m_eff {
            return Err("inconsistent SNP filtering across passes".into());
        }
        y = y_new;
    }

    if l > m_eff {
        return Err("k + oversample exceeds retained variant count".into());
    }
    if k > m_eff.min(n) {
        return Err("k exceeds matrix rank limit".into());
    }

    let y_mat = DMatrix::from_row_slice(m_eff, l, &y);
    let q_full = y_mat.qr().q();
    let q = q_full.columns(0, l).into_owned();
    let q_row = dmatrix_to_rowmajor(&q);

    let iter_b = make_iter()?;
    let (b, n_b) = compute_qt_a_with_iter(iter_b, &q_row, l, center, scale)?;
    if n_b != n {
        return Err("sample count mismatch".into());
    }

    let c = gram_b(&b, l, n);
    let c_mat = DMatrix::from_row_slice(l, l, &c);
    let eig = SymmetricEigen::new(c_mat);

    let mut pairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = Vec::with_capacity(k);
    let mut u_b = DMatrix::<f64>::zeros(l, k);
    for (col, (val, idx)) in pairs.into_iter().take(k).enumerate() {
        let sigma = val.max(0.0).sqrt();
        s.push(sigma);
        let ucol = eig.eigenvectors.column(idx);
        u_b.set_column(col, &ucol);
    }

    let u = &q * &u_b;
    let u_row = dmatrix_to_rowmajor(&u);

    let mut vt = vec![0.0; k * n];
    for col in 0..k {
        let sigma = s[col];
        if sigma == 0.0 {
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let vt_row = &mut vt[col * n..(col + 1) * n];
        for r in 0..l {
            let coeff = u_b[(r, col)];
            if coeff != 0.0 {
                let b_row = &b[r * n..(r + 1) * n];
                for j in 0..n {
                    vt_row[j] += coeff * b_row[j];
                }
            }
        }
        for j in 0..n {
            vt_row[j] *= inv_sigma;
        }
    }

    Ok((u_row, s, vt, m_eff, n))
}

fn extract_quoted_value(header: &str, key: &str) -> Option<String> {
    let key_pos = header.find(key)?;
    let rest = &header[key_pos + key.len()..];
    let colon_pos = rest.find(':')?;
    let rest = rest[colon_pos + 1..].trim_start();
    let mut chars = rest.chars();
    let quote = chars.next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let end = chars.position(|c| c == quote)?;
    Some(rest[1..1 + end].to_string())
}

fn extract_bool_value(header: &str, key: &str) -> Option<bool> {
    let key_pos = header.find(key)?;
    let rest = &header[key_pos + key.len()..];
    let colon_pos = rest.find(':')?;
    let rest = rest[colon_pos + 1..].trim_start();
    if rest.starts_with("True") {
        Some(true)
    } else if rest.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let key_pos = header.find("shape")?;
    let rest = &header[key_pos + "shape".len()..];
    let colon_pos = rest.find(':')?;
    let rest = rest[colon_pos + 1..].trim_start();
    let start = rest.find('(')?;
    let end = rest[start + 1..].find(')')? + start + 1;
    let inner = &rest[start + 1..end];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let val = part.trim();
        if val.is_empty() {
            continue;
        }
        dims.push(val.parse::<usize>().ok()?);
    }
    Some(dims)
}

fn parse_npy_header(bytes: &[u8]) -> Result<(usize, String, bool, Vec<usize>), String> {
    if bytes.len() < 10 {
        return Err("npy file too small".into());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let _minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => {
            if bytes.len() < 10 {
                return Err("npy header truncated".into());
            }
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("npy header truncated".into());
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => return Err(format!("unsupported npy version: {major}")),
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err("npy header truncated".into());
    }
    let header = str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| "npy header not utf8".to_string())?;
    let descr = extract_quoted_value(header, "descr")
        .ok_or_else(|| "npy header missing descr".to_string())?;
    let fortran = extract_bool_value(header, "fortran_order")
        .ok_or_else(|| "npy header missing fortran_order".to_string())?;
    let shape = extract_shape(header)
        .ok_or_else(|| "npy header missing shape".to_string())?;
    Ok((header_end, descr, fortran, shape))
}

struct NpyView {
    mmap: Mmap,
    offset: usize,
    rows: usize,
    cols: usize,
    descr: String,
    fortran: bool,
}

impl NpyView {
    fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let (offset, descr, fortran, shape) = parse_npy_header(&mmap)?;
        if shape.len() != 2 {
            return Err("only 2D .npy arrays are supported".into());
        }
        let rows = shape[0];
        let cols = shape[1];
        Ok(Self {
            mmap,
            offset,
            rows,
            cols,
            descr,
            fortran,
        })
    }

    fn as_f64(&self) -> Result<Cow<[f64]>, String> {
        if self.fortran {
            return Err("fortran_order .npy not supported".into());
        }
        if self.descr.starts_with('>') {
            return Err("big-endian .npy not supported".into());
        }
        if self.descr != "<f8" && self.descr != "|f8" && self.descr != "f8" {
            return Err(format!("unsupported dtype: {}", self.descr));
        }

        let len_bytes = self.rows.checked_mul(self.cols)
            .and_then(|v| v.checked_mul(8))
            .ok_or("matrix too large")?;
        let data = self.mmap.get(self.offset..self.offset + len_bytes)
            .ok_or("npy data truncated")?;

        let (prefix, aligned, suffix) = unsafe { data.align_to::<f64>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(aligned));
        }

        let mut out = Vec::with_capacity(self.rows * self.cols);
        for chunk in data.chunks_exact(8) {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            out.push(f64::from_le_bytes(bytes));
        }
        Ok(Cow::Owned(out))
    }
}

fn randomized_svd_blocked(
    a: &[f64],
    m: usize,
    n: usize,
    k: usize,
    oversample: usize,
    n_iter: usize,
    block_rows: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if m == 0 || n == 0 {
        return Err("input matrix must be non-empty".into());
    }
    let min_dim = m.min(n);
    if k == 0 || k > min_dim {
        return Err(format!("k must be in 1..={min_dim}"));
    }
    let l = k.checked_add(oversample).ok_or("k + oversample overflow")?;
    if l == 0 || l > min_dim {
        return Err("k + oversample must be in 1..=min(m,n)".into());
    }
    if a.len() != m * n {
        return Err("input matrix size mismatch".into());
    }
    let block = if block_rows == 0 { m } else { block_rows.min(m) };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = vec![0.0; n * l];
    for k in 0..n {
        let row = &mut omega[k * l..(k + 1) * l];
        for j in 0..l {
            row[j] = rng.sample(StandardNormal);
        }
    }

    let mut y = matmul_a_b(a, m, n, &omega, l, block);
    for _ in 0..n_iter {
        let z = matmul_at_b(a, m, n, &y, l, block);
        y = matmul_a_b(a, m, n, &z, l, block);
    }

    let y_mat = DMatrix::from_row_slice(m, l, &y);
    let q_full = y_mat.qr().q();
    let q = q_full.columns(0, l).into_owned();
    let q_row = dmatrix_to_rowmajor(&q);

    let b = matmul_qt_a(&q_row, m, l, a, n, block);
    let c = gram_b(&b, l, n);
    let c_mat = DMatrix::from_row_slice(l, l, &c);
    let eig = SymmetricEigen::new(c_mat);

    let mut pairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut s = Vec::with_capacity(k);
    let mut u_b = DMatrix::<f64>::zeros(l, k);
    for (col, (val, idx)) in pairs.into_iter().take(k).enumerate() {
        let sigma = val.max(0.0).sqrt();
        s.push(sigma);
        let ucol = eig.eigenvectors.column(idx);
        u_b.set_column(col, &ucol);
    }

    let u = &q * &u_b;
    let u_row = dmatrix_to_rowmajor(&u);

    let mut vt = vec![0.0; k * n];
    for col in 0..k {
        let sigma = s[col];
        if sigma == 0.0 {
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let vt_row = &mut vt[col * n..(col + 1) * n];
        for r in 0..l {
            let coeff = u_b[(r, col)];
            if coeff != 0.0 {
                let b_row = &b[r * n..(r + 1) * n];
                for j in 0..n {
                    vt_row[j] += coeff * b_row[j];
                }
            }
        }
        for j in 0..n {
            vt_row[j] *= inv_sigma;
        }
    }

    Ok((u_row, s, vt))
}

#[pyfunction]
#[pyo3(signature = (a, k, oversample=10, n_iter=2, block_rows=4096, seed=0))]
/// Blocked randomized SVD for a dense matrix.
/// Returns (U, S, Vt) with shapes (m, k), (k,), (k, n).
/// Non-contiguous inputs are copied into a row-major buffer.
pub fn block_randomized_svd<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    k: usize,
    oversample: usize,
    n_iter: usize,
    block_rows: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let a_arr = a.as_array();
    let (m, n) = (a_arr.shape()[0], a_arr.shape()[1]);
    let a_data: Cow<[f64]> = match a.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(a_arr.iter().cloned().collect()),
    };

    let result = py.allow_threads(|| {
        randomized_svd_blocked(
            a_data.as_ref(),
            m,
            n,
            k,
            oversample,
            n_iter,
            block_rows,
            seed,
        )
    });
    let (u_vec, s_vec, vt_vec) =
        result.map_err(|e| PyRuntimeError::new_err(e))?;

    let u_arr = Array2::from_shape_vec((m, k), u_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vt_arr = Array2::from_shape_vec((k, n), vt_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_np = u_arr.into_pyarray_bound(py);
    let s_np = s_vec.into_pyarray_bound(py);
    let vt_np = vt_arr.into_pyarray_bound(py);

    Ok((u_np, s_np, vt_np))
}

#[pyfunction]
#[pyo3(signature = (path, k, oversample=10, n_iter=2, block_rows=4096, seed=0))]
/// Blocked randomized SVD for a float64 C-order .npy file.
pub fn block_randomized_svd_npy<'py>(
    py: Python<'py>,
    path: &str,
    k: usize,
    oversample: usize,
    n_iter: usize,
    block_rows: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let view = NpyView::open(path).map_err(PyRuntimeError::new_err)?;
    let a_data = view.as_f64().map_err(PyRuntimeError::new_err)?;
    let (m, n) = (view.rows, view.cols);

    let result = py.allow_threads(|| {
        randomized_svd_blocked(
            a_data.as_ref(),
            m,
            n,
            k,
            oversample,
            n_iter,
            block_rows,
            seed,
        )
    });
    let (u_vec, s_vec, vt_vec) =
        result.map_err(|e| PyRuntimeError::new_err(e))?;

    let u_arr = Array2::from_shape_vec((m, k), u_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vt_arr = Array2::from_shape_vec((k, n), vt_vec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_np = u_arr.into_pyarray_bound(py);
    let s_np = s_vec.into_pyarray_bound(py);
    let vt_np = vt_arr.into_pyarray_bound(py);

    Ok((u_np, s_np, vt_np))
}

#[pyfunction]
#[pyo3(signature = (prefix, k, oversample=10, n_iter=2, maf=0.0, miss=1.0, center=true, scale=false, impute=true, seed=0))]
/// Blocked randomized SVD for PLINK bed/bim/fam (SNP-major).
/// Returns (U, S, Vt) for the transposed orientation:
///   U: (n_samples, k), Vt: (k, n_snps_retained).
pub fn block_randomized_svd_bed<'py>(
    py: Python<'py>,
    prefix: &str,
    k: usize,
    oversample: usize,
    n_iter: usize,
    maf: f32,
    miss: f32,
    center: bool,
    scale: bool,
    impute: bool,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let center = center || scale;
    let prefix_s = prefix.to_string();
    let result = py.allow_threads(|| {
        randomized_svd_streaming::<BedSnpIter, _>(
            || BedSnpIter::new_with_fill(&prefix_s, maf, miss, impute),
            k,
            oversample,
            n_iter,
            center,
            scale,
            seed,
        )
    });
    let (u_vec, s_vec, vt_vec, m, n) = result.map_err(PyRuntimeError::new_err)?;
    let u_out = transpose_rowmajor(&vt_vec, k, n);
    let vt_out = transpose_rowmajor(&u_vec, m, k);

    let u_arr = Array2::from_shape_vec((n, k), u_out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vt_arr = Array2::from_shape_vec((k, m), vt_out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_np = u_arr.into_pyarray_bound(py);
    let s_np = s_vec.into_pyarray_bound(py);
    let vt_np = vt_arr.into_pyarray_bound(py);

    Ok((u_np, s_np, vt_np))
}

#[pyfunction]
#[pyo3(signature = (path, k, oversample=10, n_iter=2, maf=0.0, miss=1.0, center=true, scale=false, impute=true, seed=0))]
/// Blocked randomized SVD for VCF/VCF.GZ (SNP-major).
/// Returns (U, S, Vt) for the transposed orientation:
///   U: (n_samples, k), Vt: (k, n_snps_retained).
pub fn block_randomized_svd_vcf<'py>(
    py: Python<'py>,
    path: &str,
    k: usize,
    oversample: usize,
    n_iter: usize,
    maf: f32,
    miss: f32,
    center: bool,
    scale: bool,
    impute: bool,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let center = center || scale;
    let path_s = path.to_string();
    let result = py.allow_threads(|| {
        randomized_svd_streaming::<VcfSnpIter, _>(
            || VcfSnpIter::new_with_fill(&path_s, maf, miss, impute),
            k,
            oversample,
            n_iter,
            center,
            scale,
            seed,
        )
    });
    let (u_vec, s_vec, vt_vec, m, n) = result.map_err(PyRuntimeError::new_err)?;
    let u_out = transpose_rowmajor(&vt_vec, k, n);
    let vt_out = transpose_rowmajor(&u_vec, m, k);

    let u_arr = Array2::from_shape_vec((n, k), u_out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let vt_arr = Array2::from_shape_vec((k, m), vt_out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let u_np = u_arr.into_pyarray_bound(py);
    let s_np = s_vec.into_pyarray_bound(py);
    let vt_np = vt_arr.into_pyarray_bound(py);

    Ok((u_np, s_np, vt_np))
}
