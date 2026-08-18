#include "faurge/reconstructor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace faurge {

Reconstructor::Reconstructor(const DeclipConfig& config) : cfg_(config) {}

void Reconstructor::reconstruct(float* audio, size_t numSamples,
                                std::vector<ClipRegion>& regions) const {
    for (auto& region : regions) {
        region.estimatedPeakAmplitude = estimatePeak(region);

        size_t len = region.length();
        if (static_cast<int>(len) <= cfg_.hermiteMaxLen) {
            hermiteReconstruct(audio, region);
        } else if (static_cast<int>(len) <= cfg_.akimaMaxLen) {
            akimaReconstruct(audio, region);
        } else {
            arReconstruct(audio, numSamples, region);
        }
    }
}

void Reconstructor::hermiteReconstruct(float* audio, ClipRegion& region) const {
    const auto& before = region.anchorsBefore;
    const auto& after  = region.anchorsAfter;

    if (before.size() < 2 || after.size() < 2) {
        float p0 = before.empty() ? 0.0f : before.back();
        float p1 = after.empty()  ? 0.0f : after.front();
        size_t len = region.length();
        for (size_t i = 0; i < len; ++i) {
            float t = static_cast<float>(i + 1) / static_cast<float>(len + 1);
            audio[region.startSample + i] = p0 + t * (p1 - p0);
        }
        return;
    }

    float p0 = before.back();
    float p1 = after.front();

    float m0 = 0.0f;
    {
        size_t nb = before.size();
        if (nb >= 3) {
            m0 = (before[nb - 1] - before[nb - 3]) / 2.0f;
        } else {
            m0 = before[nb - 1] - before[nb - 2];
        }
    }

    float m1 = 0.0f;
    {
        size_t na = after.size();
        if (na >= 3) {
            m1 = (after[2] - after[0]) / 2.0f;
        } else {
            m1 = after[1] - after[0];
        }
    }

    float intervalLen = static_cast<float>(region.length() + 1);
    m0 *= intervalLen;
    m1 *= intervalLen;

    size_t len = region.length();
    for (size_t i = 0; i < len; ++i) {
        float t = static_cast<float>(i + 1) / static_cast<float>(len + 1);
        float t2 = t * t;
        float t3 = t2 * t;

        float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 =         t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 =         t3 -        t2;

        float value = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;

        audio[region.startSample + i] =
            constrainSample(value, region.estimatedPeakAmplitude, region.polarity);
    }
}

void Reconstructor::akimaReconstruct(float* audio, ClipRegion& region) const {
    std::vector<float> knots;
    std::vector<float> xpos;

    const auto& before = region.anchorsBefore;
    const auto& after  = region.anchorsAfter;

    int anchorN = static_cast<int>(before.size());
    int clipLen = static_cast<int>(region.length());
    int afterN  = static_cast<int>(after.size());
    int totalKnots = anchorN + 2 + afterN;

    knots.resize(totalKnots);
    xpos.resize(totalKnots);

    int idx = 0;
    for (int i = 0; i < anchorN; ++i) {
        knots[idx] = before[i];
        xpos[idx]  = static_cast<float>(i - anchorN);
        ++idx;
    }
    knots[idx] = before.empty() ? audio[region.startSample] : before.back();
    xpos[idx]  = 0.0f;
    ++idx;
    knots[idx] = after.empty() ? audio[region.endSample] : after.front();
    xpos[idx]  = static_cast<float>(clipLen + 1);
    ++idx;
    for (int i = 0; i < afterN; ++i) {
        knots[idx] = after[i];
        xpos[idx]  = static_cast<float>(clipLen + 2 + i);
        ++idx;
    }

    int nk = static_cast<int>(knots.size());
    std::vector<float> slopes(nk, 0.0f);

    std::vector<float> dd(nk - 1, 0.0f);
    for (int i = 0; i < nk - 1; ++i) {
        float dx = xpos[i + 1] - xpos[i];
        dd[i] = (dx != 0.0f) ? (knots[i + 1] - knots[i]) / dx : 0.0f;
    }

    for (int i = 0; i < nk; ++i) {
        if (i < 2 || i >= nk - 2) {
            if (i < nk - 1) slopes[i] = dd[std::min(i, nk - 2)];
            else slopes[i] = dd[nk - 2];
        } else {
            float w1 = std::fabs(dd[i] - dd[i - 1]);
            float w2 = std::fabs(dd[i - 2] - dd[i - 1]);
            float totalW = w1 + w2;
            if (totalW < 1e-12f) {
                slopes[i] = 0.5f * (dd[i - 1] + dd[i]);
            } else {
                float wa = std::fabs(dd[std::min(i, nk - 2)] - dd[i - 1]);
                float wb = std::fabs(dd[i - 2] - dd[std::max(i - 3, 0)]);
                float tw = wa + wb;
                if (tw < 1e-12f) {
                    slopes[i] = 0.5f * (dd[i - 1] + dd[std::min(i, nk - 2)]);
                } else {
                    slopes[i] = (wa * dd[i - 1] + wb * dd[std::min(i, nk - 2)]) / tw;
                }
            }
        }
    }

    int knotLeft  = anchorN;
    int knotRight = anchorN + 1;
    float x0 = xpos[knotLeft];
    float x1 = xpos[knotRight];
    float dx = x1 - x0;
    float p0 = knots[knotLeft];
    float p1 = knots[knotRight];
    float s0 = slopes[knotLeft] * dx;
    float s1 = slopes[knotRight] * dx;

    for (int i = 0; i < clipLen; ++i) {
        float x = static_cast<float>(i + 1);
        float t = (x - x0) / dx;
        float t2 = t * t;
        float t3 = t2 * t;

        float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 =         t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 =         t3 -        t2;

        float value = h00 * p0 + h10 * s0 + h01 * p1 + h11 * s1;

        audio[region.startSample + i] =
            constrainSample(value, region.estimatedPeakAmplitude, region.polarity);
    }
}

void Reconstructor::arReconstruct(float* audio, size_t numSamples,
                                  ClipRegion& region) const {
    int order = cfg_.arModelOrder;
    int clipLen = static_cast<int>(region.length());
    int contextLen = order * 4;

    auto burgFit = [](const std::vector<float>& data, int p) -> std::vector<float> {
        int n = static_cast<int>(data.size());
        if (n <= p) {
            return std::vector<float>(p, 0.0f);
        }

        std::vector<float> a(p + 1, 0.0f);
        a[0] = 1.0f;

        std::vector<float> ef(data.begin(), data.end());
        std::vector<float> eb(data.begin(), data.end());

        for (int m = 0; m < p; ++m) {
            float num = 0.0f, den = 0.0f;
            for (int j = m + 1; j < n; ++j) {
                num += ef[j] * eb[j - 1];
                den += ef[j] * ef[j] + eb[j - 1] * eb[j - 1];
            }
            float km = (den > 1e-30f) ? (-2.0f * num / den) : 0.0f;

            std::vector<float> aNew(p + 1, 0.0f);
            aNew[0] = 1.0f;
            for (int i = 1; i <= m; ++i) {
                aNew[i] = a[i] + km * a[m + 1 - i];
            }
            aNew[m + 1] = km;
            a = aNew;

            std::vector<float> efNew(n, 0.0f);
            for (int j = m + 1; j < n; ++j) {
                efNew[j] = ef[j] + km * eb[j - 1];
            }
            for (int j = m + 1; j < n; ++j) {
                eb[j] = eb[j - 1] + km * ef[j];
            }
            ef = efNew;
        }

        return std::vector<float>(a.begin() + 1, a.end());
    };

    std::vector<float> fwdContext;
    {
        int start = std::max(0, static_cast<int>(region.startSample) - contextLen);
        for (int i = start; i < static_cast<int>(region.startSample); ++i) {
            fwdContext.push_back(audio[i]);
        }
    }

    std::vector<float> bwdContext;
    {
        size_t start = region.endSample + 1;
        size_t end   = std::min(numSamples, start + contextLen);
        for (size_t i = start; i < end; ++i) {
            bwdContext.push_back(audio[i]);
        }
    }

    auto fwdCoeffs = burgFit(fwdContext, order);
    std::vector<float> fwdPred(clipLen, 0.0f);
    {
        std::vector<float> buf(fwdContext);
        for (int i = 0; i < clipLen; ++i) {
            float val = 0.0f;
            int bLen = static_cast<int>(buf.size());
            for (int k = 0; k < order && k < bLen; ++k) {
                val -= fwdCoeffs[k] * buf[bLen - 1 - k];
            }
            fwdPred[i] = val;
            buf.push_back(val);
        }
    }

    std::vector<float> bwdContextRev(bwdContext.rbegin(), bwdContext.rend());
    auto bwdCoeffs = burgFit(bwdContextRev, order);
    std::vector<float> bwdPred(clipLen, 0.0f);
    {
        std::vector<float> buf(bwdContextRev);
        for (int i = 0; i < clipLen; ++i) {
            float val = 0.0f;
            int bLen = static_cast<int>(buf.size());
            for (int k = 0; k < order && k < bLen; ++k) {
                val -= bwdCoeffs[k] * buf[bLen - 1 - k];
            }
            bwdPred[i] = val;
            buf.push_back(val);
        }
        std::reverse(bwdPred.begin(), bwdPred.end());
    }

    for (int i = 0; i < clipLen; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(std::max(clipLen - 1, 1));
        float wFwd = 0.5f * (1.0f + std::cos(t * 3.14159265f));
        float wBwd = 1.0f - wFwd;

        float blended = wFwd * fwdPred[i] + wBwd * bwdPred[i];

        audio[region.startSample + i] =
            constrainSample(blended, region.estimatedPeakAmplitude, region.polarity);
    }
}

float Reconstructor::estimatePeak(const ClipRegion& region) const {
    const auto& before = region.anchorsBefore;
    const auto& after  = region.anchorsAfter;

    float slopeBefore = 0.0f;
    if (before.size() >= 2) {
        slopeBefore = before.back() - before[before.size() - 2];
    }

    float slopeAfter = 0.0f;
    if (after.size() >= 2) {
        slopeAfter = after[1] - after[0];
    }

    float avgSlope = (std::fabs(slopeBefore) + std::fabs(slopeAfter)) / 2.0f;
    float clipLen  = static_cast<float>(region.length());
    float rawPeak  = cfg_.clipThreshold + avgSlope * clipLen * 0.25f;

    float maxPeak = cfg_.clipThreshold * cfg_.peakOvershoot;
    return std::min(rawPeak, maxPeak);
}

float Reconstructor::constrainSample(float value, float peakEst,
                                     ClipPolarity polarity) const {
    if (polarity == ClipPolarity::Positive) {
        return std::min(value, peakEst);
    } else {
        return std::max(value, -peakEst);
    }
}

} // namespace faurge
