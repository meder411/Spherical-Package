#include <torch/extension.h>

#include "cuda/resample.cuh"

namespace spherical {

namespace cuda {

torch::Tensor VotingResampleToMap(torch::Tensor input,
                                  torch::Tensor sample_map, int outputHeight,
                                  int outputWidth, int numCandidates) {
  // Useful dimensions to have
  const int64_t batchSize   = input.size(0);
  const int64_t channels    = input.size(1);
  const int64_t inputHeight = input.size(2);
  const int64_t inputWidth  = input.size(3);

  // Initialize output and index mask
  torch::Tensor output = torch::zeros(
      {batchSize, channels, outputHeight, outputWidth}, input.options());

  // Call the CUDA kernel once per batch
  for (int b = 0; b < batchSize; b++) {
    torch::Tensor tmp = torch::zeros(
        {channels, outputHeight, outputWidth, numCandidates}, input.options());
    ResampleToMap2DVotingLauncher(input[b], sample_map, channels, inputHeight,
                                  inputWidth, outputHeight, outputWidth,
                                  numCandidates, tmp);

    // Compute the index with the most votes
    torch::Tensor argmax = tmp.argmax(-1);

    // Copy the selected indices to the output
    output[b].copy_(argmax);
  }

  return output;
}

}  // namespace cuda

}  // namespace spherical