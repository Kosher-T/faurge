#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "faurge/declipper.hpp"
#include "faurge/metrics.hpp"

#include <string>
#include <vector>

namespace py = pybind11;
namespace fg = faurge;

// Convert a DeclipResult to a Python dict
py::dict result_to_dict(const fg::DeclipResult& r) {
    py::dict d;
    d["success"]            = r.success;
    d["processing_time_ms"] = r.processingTimeMs;
    d["before_thdn_db"]     = r.beforeThdnDb;
    d["after_thdn_db"]      = r.afterThdnDb;
    d["error_message"]      = r.errorMessage;

    py::dict clip_report;
    clip_report["total_samples"]        = r.report.totalSamples;
    clip_report["total_clipped_samples"] = r.report.totalClippedSamples;
    clip_report["percent_clipped"]      = r.report.percentClipped;
    clip_report["num_regions"]          = r.report.regions.size();

    py::list regions;
    for (const auto& rm : r.regionMetrics) {
        py::dict reg;
        reg["index"]                 = rm.regionIndex;
        reg["length_samples"]        = rm.lengthSamples;
        reg["severity"]              = fg::severityToString(rm.severity);
        reg["estimated_overshoot_db"] = rm.estimatedOvershootDb;
        reg["reconstruction_snr_db"] = rm.reconstructionSnrDb;
        regions.append(reg);
    }
    clip_report["regions"] = regions;
    d["clip_report"] = clip_report;

    return d;
}

// Accept numpy array (float32) as input for zero-copy path
py::dict process_numpy(py::array_t<float> audio_arr, int sample_rate,
                       float clip_threshold, int min_clip_length,
                       int merge_gap, int anchor_size,
                       bool detect_soft_clip, float soft_clip_deriv_thr,
                       int hermite_max_len, int akima_max_len,
                       int ar_model_order, float peak_overshoot,
                       int crossfade_width, bool enable_anti_alias,
                       float dc_block_freq_hz) {

    fg::DeclipConfig cfg;
    cfg.clipThreshold    = clip_threshold;
    cfg.minClipLength    = min_clip_length;
    cfg.mergeGap         = merge_gap;
    cfg.anchorSize       = anchor_size;
    cfg.detectSoftClip   = detect_soft_clip;
    cfg.softClipDerivThr = soft_clip_deriv_thr;
    cfg.hermiteMaxLen    = hermite_max_len;
    cfg.akimaMaxLen      = akima_max_len;
    cfg.arModelOrder     = ar_model_order;
    cfg.peakOvershoot    = peak_overshoot;
    cfg.crossfadeWidth   = crossfade_width;
    cfg.enableAntiAlias  = enable_anti_alias;
    cfg.dcBlockFreqHz    = dc_block_freq_hz;
    cfg.verbose          = false;

    py::buffer_info buf = audio_arr.request();
    float* data = static_cast<float*>(buf.ptr);
    size_t n = buf.size;

    std::vector<float> audio(data, data + n);

    fg::Declipper declipper(cfg);
    fg::DeclipResult result = declipper.process(audio, sample_rate);

    // Copy result back to input array (in-place processing)
    std::copy(audio.begin(), audio.end(), data);

    return result_to_dict(result);
}

py::dict process_file_py(const std::string& input_path, const std::string& output_path,
                         // config params same as above
                         float clip_threshold, int min_clip_length,
                         int merge_gap, int anchor_size,
                         bool detect_soft_clip, float soft_clip_deriv_thr,
                         int hermite_max_len, int akima_max_len,
                         int ar_model_order, float peak_overshoot,
                         int crossfade_width, bool enable_anti_alias,
                         float dc_block_freq_hz) {
    // This is a stub — file I/O happens in Python.
    // Users should read WAV in Python, call process_numpy, then write WAV.
    py::dict d;
    d["success"] = false;
    d["error_message"] = "Use declip_audio() Python function instead — file I/O is in Python";
    return d;
}

void clip_numpy(py::array_t<float> audio_arr, float gain_db, bool soft_clip) {
    py::buffer_info buf = audio_arr.request();
    float* data = static_cast<float*>(buf.ptr);
    size_t n = buf.size;

    std::vector<float> audio(data, data + n);
    fg::clip_audio_inplace(audio, gain_db, soft_clip);
    std::copy(audio.begin(), audio.end(), data);
}

PYBIND11_MODULE(_faurge_declip_cpp, m) {
    m.doc() = "Faurge Portable Declipper — C++ accelerated backend";

    m.def("declip", &process_numpy,
          py::arg("audio").noconvert(),
          py::arg("sample_rate"),
          py::arg("clip_threshold")      = 0.9999f,
          py::arg("min_clip_length")     = 2,
          py::arg("merge_gap")           = 3,
          py::arg("anchor_size")         = 20,
          py::arg("detect_soft_clip")    = true,
          py::arg("soft_clip_deriv_thr") = 0.5f,
          py::arg("hermite_max_len")     = 16,
          py::arg("akima_max_len")       = 64,
          py::arg("ar_model_order")      = 14,
          py::arg("peak_overshoot")      = 1.15f,
          py::arg("crossfade_width")     = 8,
          py::arg("enable_anti_alias")   = true,
          py::arg("dc_block_freq_hz")    = 10.0f,
          "Process audio in-place: detect clipping, reconstruct, post-filter. "
          "Modifies the input numpy array and returns a result dict.");

    m.def("clip", &clip_numpy,
          py::arg("audio").noconvert(),
          py::arg("gain_db")  = 6.0f,
          py::arg("soft_clip") = false,
          "Apply gain and clip audio in-place. Modifies the input numpy array.");
}
