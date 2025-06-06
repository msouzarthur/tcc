require PolyHok

PolyHok.defmodule MC do

  defk gpu_mc(results, n_points) do
    idx = blockIdx.x * blockDim.x + threadIdx.x

    count = 0

    x = 1.0
    y = 2.0
    
    if x*x + y*y <= 1.0 do
    count = count + 1
    end
    
    results[idx] = count
  end

end

# Configuração
[arg] = System.argv()
n_points = String.to_integer(arg)
block_size = 128
n_blocks = ceil(n_points/block_size)

# Armazenar o resultado
h_results = Nx.broadcast(Nx.tensor(0), {n_points})
results = PolyHok.new_gnx(h_results)

prev = System.monotonic_time()

PolyHok.spawn(
    &MC.gpu_mc/2,
    {n_blocks, 1, 1},
    {block_size, 1, 1},
    [results, n_points]
)

next = System.monotonic_time()

hits = PolyHok.get_gnx(results) 
  |> Nx.sum() 
  |> Nx.to_number()

pi_estimate = 4.0 * hits / n_points

IO.puts("""
tempo #{System.convert_time_unit(next-prev,:native,:millisecond)}ms
pontos #{n_points}
pi #{:io_lib.format("~.15f", [pi_estimate])}
erro #{:io_lib.format("~.6f", [abs(pi_estimate-:math.pi())/:math.pi()*100])}%
""")