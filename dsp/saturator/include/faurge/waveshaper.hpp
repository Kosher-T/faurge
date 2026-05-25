#pragma once

namespace faurge {

float saturateTube(float x);
float saturateTape(float x);
float saturateDiode(float x);
float saturateAsymmetric(float x);

using WaveshaperFn = float(*)(float);
WaveshaperFn getWaveshaper(int type);

} // namespace faurge
