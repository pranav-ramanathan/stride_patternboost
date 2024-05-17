include("constants.jl")

"""
Best possible constructions: https://arxiv.org/pdf/1402.5646

f(2) = 2
f(3) = 4
f(4) = 8
f(5) = 16
f(6) <= 30

(This program works with negative scores, so the goal for K=6 is to achieve -30. It is possible with N=8, SINGLETONS_IN_A1=4, but very hard)


"""

const K = 5
const SINGLETONS_IN_A1 = 4

"""
A0 = emptyset, Ak-1 = full set
A1 = singletons and rest, Ak-2 = complements of A1
k-4 families remain

Every obj will consist of k-4 blocks of 2^n zeros and 1s, separated by commas

Good if:
    each Ai from i=2 to k-3 is an antichain
    each Ai from i=2 to k-3 is saturated
    the Ais are disjoint
    the sequence is layered. Need to check A2->A1, A3->A2, A4->A3, ..., Ak-2->Ak-3

Fixing a transformer construction:
    Don't?


"""





function greedy_search_from_startpoint(obj::OBJ_TYPE)::OBJ_TYPE
    chars = collect(obj)
    # Get the length of the string
    n = length(chars)
    # Generate 10 unique random indices within the string length
    indices = rand(1:n, 10)
    
    # Iterate over each index
    for idx in indices
        # Check if the character at the index is '0'
        if chars[idx] == '0'
            # Replace '0' with '1'
            chars[idx] = '1'
        end
    end
    
    # Convert the character array back to a string
    return String(chars)
end


function extract_indices_from_blocks(input_string::String)
    # Split the input string by commas to get the blocks of bits
    blocks = split(input_string, ',')
    

    # Initialize a vector of vectors to store indices of ones for each block
    indices_of_ones = Vector{Vector{Int}}()

    # Iterate over each block
    for block in blocks
        # Check if the block length is 2^N, to ensure correct input format
        if length(block) != 2^N
            error("Each block must be exactly 2^N bits long.")
        end

        # Find indices where the character is '1'
        current_indices = findall(c -> c == '1', block)
        
        # Append the indices to the main list
        push!(indices_of_ones, current_indices)
    end

    return indices_of_ones
end


# Generate all subsets of the set {1, 2, ..., N}
function generate_subsets(N)
    subsets = Vector{Vector{Int}}()
    for i = 0:(2^N - 1)
        push!(subsets, findall(j -> (i & (1 << (j-1))) != 0, 1:N))
    end

    return subsets
end

# Create the adjacency matrix
function create_adjacency_matrix(subsets)
    len = length(subsets)
    adjmat = zeros(Int, len, len)
    for i in 1:len
        for j in 1:len
            if all(elem -> elem in subsets[j], subsets[i])
                adjmat[i, j] = 1
            elseif all(elem -> elem in subsets[i], subsets[j])
                adjmat[i, j] = -1
            else
                adjmat[i, j] = 0
            end
        end
    end
    return adjmat
end


function antichain_scores(adjacency_matrix, set_families)
    scores = Int[]  # Initialize an empty array to store scores for each family

    # Iterate over each family
    for family in set_families
        score = 0  # Initialize score for the current family

        # Loop over all pairs of elements in the family
        for i in 1:length(family)
            for j in (i+1):length(family)
                # Check the adjacency matrix for subset relationships
                if adjacency_matrix[family[i], family[j]] != 0 
                    score += 1  # Increment score if they are subsets of each other
                end
            end
        end

        # Append the score to the scores list
        push!(scores, score)
    end

    return scores
end


function saturation_scores(adjacency_matrix, set_families)
    num_sets = size(adjacency_matrix, 1)
    scores = Int[]  # Initialize an empty array to store scores for each family

    # Iterate over each family
    for family in set_families
        score = 0  # Initialize score for the current family

        # Check each set outside the family
        for potential_set in 1:num_sets
            if potential_set in family
                continue  # Skip if the set is already in the family
            end

            # Check if this set can be added without violating saturation properties
            is_valid = true  # Assume it can be added until proven otherwise

            # Loop over all sets in the family
            for set_in_family in family
                if adjacency_matrix[potential_set, set_in_family] != 0 
                    is_valid = false  # Can't add this set if it is a subset or superset of any in the family
                    break
                end
            end

            # Increment score if this set can be added
            if is_valid
                score += 1
            end
        end

        # Append the score to the scores list
        push!(scores, score)
    end

    return scores
end


# Convert an index to a subset based on binary representation
function index_to_subset(index::Int)
    # Subtract 1 to adjust for 1-based indexing in Julia
    index -= 1

    subset = []
    elem = 0

    # Iterate through each bit of the index
    while index > 0
        # Move to the next bit position
        elem += 1
        
        # Check if the least significant bit is set
        if index & 1 == 1
            push!(subset, elem)
        end
        
        # Right-shift the index to check the next bit
        index >>= 1
    end

    return subset
end

# Convert a subset to its index based on binary representation
function subset_to_index(subset)
    index = 0
    for elem in subset
        index += 1 << (elem - 1)
    end
    return index + 1  # +1 to adjust for 1-based indexing in Julia
end


function define_families(subsets)
    A0 = [Vector{Int}()]
    A1 = [findall(x -> x == i, 1:N) for i in 1:SINGLETONS_IN_A1]
    push!(A1, collect((SINGLETONS_IN_A1+1):N))

    B1 = [setdiff(1:N, s) for s in A1]
    B0 = [collect(1:N)]

    # Find indices
    indices_A0 = [subset_to_index(s) for s in A0]
    indices_A1 = [subset_to_index(s) for s in A1]
    indices_B1 = [subset_to_index(s) for s in B1]
    indices_B0 = [subset_to_index(s) for s in B0]

    return indices_A0, indices_A1, indices_B1, indices_B0
end


function measure_collisions(families)
    index_counts = Dict{Int, Int}()  # Dictionary to store the occurrence count of each index

    # Count occurrences of each index across all families
    for family in families
        for index in family
            if haskey(index_counts, index)
                index_counts[index] += 1
            else
                index_counts[index] = 1
            end
        end
    end

    # Calculate collisions: sum up all counts where an index appears more than once
    collisions = sum(value > 1 ? value - 1 : 0 for value in values(index_counts))

    return collisions
end


function measure_layeredness(families, adjacency_matrix)
    total_failures = 0  # Initialize the count of dominance failures

    # Loop through each family, except the last one
    for i in 1:length(families)-1
        current_family = families[i]
        next_family = families[i+1]
        failures = 0  # Count failures for the current pair of families

        # Check dominance condition for each element in the next family
        for x in next_family
            dominated = false  # Flag to check if x is dominated by any element in current_family

            # Check if there's at least one y in current_family that dominates x
            for y in current_family
                if adjacency_matrix[x, y] == -1
                    dominated = true
                    break  # Stop checking once we find a dominating element
                end
            end

            # If x is not dominated by any element in current_family, increment failures
            if !dominated
                failures += 1
                #print(index_to_subset(x))
                #print(" ")
            end
        end

        # Add failures for this pair to the total failures
        total_failures += failures
    end

    return total_failures
end


function reward_calc(obj::OBJ_TYPE; verbose=false)::REWARD_TYPE
    subsets = generate_subsets(N)
    adjacency_matrix = create_adjacency_matrix(subsets)
    families = extract_indices_from_blocks(obj)

    antichain_score = antichain_scores(adjacency_matrix, families)
    saturation_score = saturation_scores(adjacency_matrix, families)

    A0, A1, Akm2, Akm1 = define_families(subsets)
    pushfirst!(families, A1)  
    pushfirst!(families, A0)  
    push!(families, Akm2)      
    push!(families, Akm1) 

    collision_score = measure_collisions(families)
    layeredness_score = measure_layeredness(families, adjacency_matrix)
    #println(subsets)

    if verbose
        println("Reward calculation:")
        println(families)
        for family in families
            for item in family
                print(index_to_subset(item))
                print(", ")
            end
            println()
        end
        println("Scores:")
        println(layeredness_score)
        println(collision_score)
        println(saturation_score)
        println(antichain_score)
        println(sum(length(subvector) for subvector in families))
    end
    
    

    
    return -10*(300*layeredness_score + 1000*collision_score + sum(saturation_score) + sum(antichain_score)) - sum(length(subvector) for subvector in families) 
end


function empty_starting_point()::OBJ_TYPE
    """
    If there is no input file, the search starts always with this object
    (E.g. empty graph, all zeros matrix, etc)
    """
    blocks = ["0"^(2^N) for _ in 1:(K-4)]
    
    # Join the blocks with a comma
    result = join(blocks, ",")
    return result
end
