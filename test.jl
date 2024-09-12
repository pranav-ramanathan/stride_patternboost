
using Random

N = 4

# Example small adjacency matrix (4x4)
adjmat = [1 0 0 1;
          0 1 0 0;
          0 0 1 0;
          0 0 0 1]

println("Original adjacency matrix:")
for row in eachrow(adjmat)
    println(row)
end

# Apply four random permutations
N = size(adjmat, 1)  # Get the size of the matrix (should be 4 in this example)
permuted_adjmats = []
for _ in 1:4
    perm = randperm(N)  # Generate a random permutation
    permuted_adjmat = adjmat[perm, perm]  # Apply the permutation to rows and columns
    push!(permuted_adjmats, permuted_adjmat)
end

# Print the permuted matrices in a square format
for i in 1:4
    println("\nPermuted adjacency matrix $i:")
    for row in eachrow(permuted_adjmats[i])
        println(row)
    end
end




function convert_adjmat_to_string(adjmat::Matrix{Int})::String
    entries = []

    # Collect entries from the upper diagonal of the matrix
    for i in 1:N-1
        for j in i+1:N
            push!(entries, string(adjmat[i, j]))
        end
    end

    # Join all entries into a single string
    return join(entries)
end

println( [convert_adjmat_to_string(permuted_adjmat) for permuted_adjmat in permuted_adjmats])
