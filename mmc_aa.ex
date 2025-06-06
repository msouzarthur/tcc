require PolyHok

PolyHok.defmodule MC do
  include CAS

  def mc(n_blocks, block_size, results, n_points) do

    PolyHok.spawn(
      &MC.gpu_mc/2,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [results, n_points]
    )

    results
  end

  defk gpu_mc(results, n_points) do

    idx = blockIdx.x * blockDim.x + threadIdx.x
    count = 0
    x = 0.0
    y = 0.2

    if x*x + y*y <= 1.0 do
      count = count + 1
    end

    results[idx] = count
  end

  def reduce(ref, initial, f) do

    {l} = PolyHok.get_shape_gnx(ref)
    type = PolyHok.get_type_gnx(ref)
    size = l

    result_gpu = PolyHok.new_gnx(Nx.tensor([initial], type: type))

    threadsPerBlock = 128
    blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
    numberOfBlocks = blocksPerGrid
    PolyHok.spawn(
      &MC.reduce_kernel/4,
      {numberOfBlocks,1,1},
      {threadsPerBlock,1,1},
      [ref,result_gpu, f, size]
    )

    result_gpu
  end

  defk reduce_kernel(a,  ref4, f,n) do

    __shared__ cache[128]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = ref4[0]

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

    __syncthreads()
    i = i/2
    end

    if (cacheIndex == 0) do
      current_value = ref4[0]
      while(!(current_value == atomic_cas(ref4,current_value,f(cache[0],current_value)))) do
        current_value = ref4[0]
      end
    end
  end
end

[arg] = System.argv()
n_points = String.to_integer(arg)
block_size = 128
n_blocks = ceil(n_points/block_size)

h_results = Nx.broadcast(Nx.tensor(0.0), {n_points})
results = PolyHok.new_gnx(h_results)

prev = System.monotonic_time()

hits = MC.mc(n_blocks, block_size, results, n_points)
  |> MC.reduce(0, PolyHok.phok fn(a, b) -> a + b end)
  |> PolyHok.get_gnx()
  |> Nx.squeeze()
  |> Nx.to_number()

next = System.monotonic_time()

pi_estimate = 4.0 * hits / n_points

IO.puts("""
  tempo #{System.convert_time_unit(next-prev,:native,:millisecond)}ms
  pontos #{n_points}
  pi #{:io_lib.format("~.15f", [pi_estimate])}
  erro #{:io_lib.format("~.6f", [abs(pi_estimate-:math.pi())/:math.pi()*100])}%
""")
