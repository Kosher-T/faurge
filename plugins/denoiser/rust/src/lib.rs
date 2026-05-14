use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;
use std::sync::Mutex;

use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::prelude::*;

const TARGET_SAMPLE_RATE: usize = 48000;

pub struct BridgeState {
    tract: Mutex<DfTract>,
}

#[no_mangle]
pub extern "C" fn df_bridge_create(
    model_path: *const c_char,
    atten_limit: f32,
) -> *mut BridgeState {
    let lim_db = (atten_limit * 100.0).clamp(0.0, 100.0);
    let rp = RuntimeParams::default_with_ch(1)
        .with_atten_lim(lim_db);

    let params = if model_path.is_null() {
        DfParams::default()
    } else {
        let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                eprintln!("[df_bridge] Invalid model path");
                return ptr::null_mut();
            }
        };
        match DfParams::new(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[df_bridge] Failed to load model: {:?}", e);
                return ptr::null_mut();
            }
        }
    };

    let tract = match DfTract::new(params, &rp) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[df_bridge] Failed to create DfTract: {:?}", e);
            return ptr::null_mut();
        }
    };

    let bridge = Box::new(BridgeState {
        tract: Mutex::new(tract),
    });

    Box::into_raw(bridge)
}

#[no_mangle]
pub extern "C" fn df_bridge_process(
    handle: *mut BridgeState,
    samples: *mut f32,
    num_samples: usize,
) -> i32 {
    if handle.is_null() || samples.is_null() || num_samples == 0 {
        return -1;
    }

    let bridge = unsafe { &*handle };
    let mut tract = match bridge.tract.lock() {
        Ok(t) => t,
        Err(_) => return -1,
    };

    let frame_size = tract.hop_size;
    if frame_size == 0 {
        return -2;
    }

    let input_slice = unsafe { std::slice::from_raw_parts_mut(samples, num_samples) };
    let mut output_buf = vec![0.0f32; frame_size];

    let mut offset = 0;
    while offset + frame_size <= num_samples {
        let frame = &input_slice[offset..offset + frame_size];

        let noisy = ArrayView2::from_shape((1, frame_size), frame).unwrap();
        let enh = ArrayViewMut2::from_shape((1, frame_size), &mut output_buf).unwrap();

        if let Err(e) = tract.process(noisy, enh) {
            eprintln!("[df_bridge] process frame failed: {:?}", e);
            return -3;
        }

        input_slice[offset..offset + frame_size].copy_from_slice(&output_buf);
        offset += frame_size;
    }

    let remaining = num_samples - offset;
    if remaining > 0 {
        let mut pad_input = vec![0.0f32; frame_size];
        pad_input[..remaining].copy_from_slice(&input_slice[offset..]);
        let mut pad_output = vec![0.0f32; frame_size];

        let noisy = ArrayView2::from_shape((1, frame_size), &pad_input).unwrap();
        let enh = ArrayViewMut2::from_shape((1, frame_size), &mut pad_output).unwrap();

        if tract.process(noisy, enh).is_ok() {
            input_slice[offset..].copy_from_slice(&pad_output[..remaining]);
        }
    }

    0
}

#[no_mangle]
pub extern "C" fn df_bridge_get_frame_size(handle: *const BridgeState) -> usize {
    if handle.is_null() {
        return 0;
    }
    let bridge = unsafe { &*handle };
    match bridge.tract.lock() {
        Ok(tract) => tract.hop_size,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn df_bridge_get_sample_rate(_handle: *const BridgeState) -> usize {
    TARGET_SAMPLE_RATE
}

#[no_mangle]
pub extern "C" fn df_bridge_set_atten_limit(
    handle: *mut BridgeState,
    limit: f32,
) {
    if handle.is_null() {
        return;
    }
    let bridge = unsafe { &*handle };
    if let Ok(mut tract) = bridge.tract.lock() {
        let lim_db = (limit * 100.0).clamp(0.0, 100.0);
        tract.set_atten_lim(lim_db);
    }
}

#[no_mangle]
pub extern "C" fn df_bridge_destroy(handle: *mut BridgeState) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}
