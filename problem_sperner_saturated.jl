using Combinatorics
using Dates: now, DateFormat
include("constants.jl")


# On Saturated k-Sperner Systems -- Natasha Morrison, Jonathan A. Noel, and Alex Scott

# Idea: A0 only contains empty set, A1 is a couple singletons and the rest
# Ak-2 is the complement of A1, and Ak-1 is just the whole set (as in the paper)
# Ai_small will live on level i, like in the paper
# The program chooses A2_small, A3_small, ... Ak-3_small on their respective layers
# We check how close this system is to being layered.
# Each family is an antichain, since they are uniform
# We make them saturated by including all maximal stable sets
# Important to check that the stable sets don't overlap across families,
# since then Lemma 15 from the paper wouldn't hold
# Stable sets are calculated by backtracking. It runs surprisingly fast!

# This reward function is a bit of a mess, sorry. Please don't try to read it.

# This local search doesn't work with multithreading (see last line of greedy function), need to fix 


# Best results I was able to achieve (N, K, SINGLETONS_IN_A1)
# (6,6,4): 0.0233 this was the best known construction
# (8,7,5): 0.0385
# (8,7,6): 0.0284
# (7,7,5): bad
# (7,7,6): bad 

# Potential parameters that could give even bigger improvements
# K = 8, N = 9, 10, 11, Singletons = 6, 7
# K = 9, N a bit bigger, Singletons a bit smaller than K

# most important parameters to set up:
const N::Int64 = 8
const K::Int64 = 7
const SINGLETONS_IN_A1 = 5 # Not sure this is the right value, we should try K-1, K-2, K-3 probably




const M::Int64 = sum(binomial(N,i) for i=2:K-3)

function try_flipping_some_bits(db, best_obj, best_rew, max_flip_count; max_improvement_count=-1)
    improved = true
    copy_obj = copy(best_obj)
    improvement_count = 0
    objects = Vector{OBJ_TYPE}(undef, 0)
    rewards = Vector{REWARD_TYPE}(undef, 0)
    while improved && improvement_count!=max_improvement_count
        improved = false
        for flip_count in 1:max_flip_count
            all_combinations = shuffle(collect(combinations(1:length(best_obj), flip_count)))
            for comb in all_combinations
                # Flip the bits at the indices specified in comb
                for index in comb
                    copy_obj[index] = copy_obj[index] == '1' ? '0' : '1'
                end
                rew, new = reward(db, String(copy_obj))
                if new
                    push!(objects, String(copy_obj))
                    push!(rewards, rew)
                end
                if rew > best_rew
                    best_rew = rew
                    best_obj = copy(copy_obj)
                    improved = true
                end
                for index in comb
                    copy_obj[index] = copy_obj[index] == '1' ? '0' : '1'
                end
            end
            copy_obj = best_obj
        end
        improvement_count += 1
        #print(improvement_count)
    end
    #println(best_rew)
    
    # Can't do it like this with multithreading!!
    add_db!(db, objects, rewards)

    return best_rew, best_obj
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::OBJ_TYPE
    chars = collect(obj)
    best_rew = reward_calc(obj)
    
    best_obj = copy(chars)
    # Try flipping one bit in all possible ways, and keep going until reward improves
    best_rew, best_obj = try_flipping_some_bits(db, best_obj, best_rew, 1; max_improvement_count = -1)

    # Everything below makes local search better, but it's too slow for large N on my PC
    """
    # If reward is big, try all possible ways to flip 2 bit while we can
    if best_rew >= -3
        if rand() < 0.1
            #println("Trying all double flips.")
            best_rew, best_obj = try_flipping_some_bits(db, best_obj, best_rew, 2; max_improvement_count = 1)
        end
    end

    # Occasionally try all possible 3 flips
    if best_rew >= 0
        if rand() < 0.01
            println("Trying all triple flips. Let's hope this is not too slow..")
            best_rew, best_obj = try_flipping_some_bits(db, best_obj, best_rew, 3; max_improvement_count = 1)
        end
    end
    """
    

    if best_rew > 0
        print_nicely(String(best_obj))
    end
    
    return String(best_obj)
end





function print_nicely(obj_string; to_file=true, filename="output.txt")
    obj = [c == '0' ? 1 : 2 for c in obj_string]
    println(Int.(obj))
    rew = reward_calc(obj_string)
    if rew > -6
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        unique_id = string(threadid())  # Assuming each thread has a unique thread ID
        directory = "constructions"
        mkpath(directory)  # Ensure the directory exists

        full_filename = joinpath(directory, string(rew, "_N=", N, "_K=", K, "_SIA1=", SINGLETONS_IN_A1, "_", unique_id, "_", timestamp, "_", filename))

        open(full_filename, "w") do f
            redirect_stdout(f) do
                reward_calc(obj_string; verbose=true)
            end
        end
    end
end


function generate_subsets(N)
    subsets = Vector{Vector{Int}}()
    for i = 0:(2^N - 1)
        subset = findall(j -> (i & (1 << (j-1))) != 0, 1:N)
        push!(subsets, subset)
    end

    # Sorting subsets by length
    sort!(subsets, by=length)
    return subsets
end


function create_subset_index_map(subsets)
    index_map = Dict{Vector{Int}, Int}()
    for (index, subset) in enumerate(subsets)
        index_map[subset] = index
    end
    return index_map
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

const subsets = generate_subsets(N)
const subset_index_map = create_subset_index_map(subsets)
const adjacency_matrix = create_adjacency_matrix(subsets)

const subsets_big = generate_subsets(N+2)
const subset_index_map_big = create_subset_index_map(subsets_big)
const adjacency_matrix_big = create_adjacency_matrix(subsets_big)

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


function find_maximal_stable_sets(N, family_indices)
    maximal_sets = Int[]
    family_indices_big = [subset_index_map_big[subsets[item]] for item in family_indices]

    function backtrack(current_set::Vector{Int}, index::Int)
        if index > N
            # Convert current_set to index via subset_index_map for efficient checking
            current_index = subset_index_map_big[Vector{Int}(current_set)]
            if !any(adjacency_matrix_big[current_index, idx] == 1 for idx in maximal_sets)
                push!(maximal_sets, current_index)
            end
            return
        end

        # Include the element if it does not make the set contain a forbidden subset
        next_element_set = union(current_set, Set([index]))

        next_element_index = subset_index_map_big[Vector{Int}(next_element_set)]
        if index >= N-1 || all(adjacency_matrix_big[family_idx, next_element_index] != 1 for family_idx in family_indices_big)
            backtrack(next_element_set, index + 1)
        end

        # Also explore the option without including the element
        backtrack(current_set, index + 1)
    end

    backtrack(Int[], 1)

    # Convert indices back to sets for final output
    stable = [subsets_big[idx] for idx in maximal_sets]# if !any(adjacency_matrix[idx, other_idx] == 1 for other_idx in maximal_sets if idx != other_idx)]
    stable_idx = Int[]
    for item in stable
        if maximum(item) <= N - 2
            
            push!(stable_idx, subset_index_map[item])
        else
            push!(stable_idx, 2^(N-2) + subset_index_map_big[item])
        end
    end
    return stable_idx
end




function define_families()
    A1 = [findall(x -> x == i, 1:N) for i in 1:SINGLETONS_IN_A1]
    push!(A1, collect((SINGLETONS_IN_A1+1):N+2))
    B1 = [setdiff(1:N+2, s) for s in A1]

    # Find indices
    indices_A1 = Int[]
    for s in A1
        if maximum(s) <= N 
            push!(indices_A1, subset_index_map[s])
        else
            push!(indices_A1, 2^N + subset_index_map_big[s])
        end
    end
    indices_B1 = Int[]
    for s in B1
        if maximum(s) <= N 
            push!(indices_B1, subset_index_map[s])
        else
            push!(indices_B1, 2^N + subset_index_map_big[s])
        end
    end

    return indices_A1, indices_B1
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
            if x > 2^N
                dominated = true
            else 
                # Check if there's at least one y in current_family that dominates x
                for y in current_family
                    if y < 2^N && adjacency_matrix[x, y] == -1
                        dominated = true
                        break  # Stop checking once we find a dominating element
                    end
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


function extract_families_from_blocks(obj)
    families = Vector{Vector{Int}}()  # To store indices for each block
    start_index = 1

    # Loop over each size from 2 to K-3
    for size in 2:(K-3)
        block_size = binomial(N, size)
        end_index = start_index + block_size - 1  # Determine the end of the current block in obj

        # Collect indices of '2's in the current block
        current_family = findall(c -> c == 2, obj[start_index:end_index])

        # Adjust indices to be relative to the whole string, not just the block
        push!(families, current_family .+ (start_index - 1 + N + 1))

        # Update start_index for the next block
        start_index = end_index + 1
    end
    #println(obj)
    #println(families)
    #exit()

    return families
end    
        


function reward_calc(obj_string::OBJ_TYPE; verbose=false)::REWARD_TYPE
    obj = [c == '0' ? 1 : 2 for c in obj_string]
    families = extract_families_from_blocks(obj)

    antichain_score = antichain_scores(adjacency_matrix, families)
    total = 0
    index_counts = Dict{Int, Int}()
    small_counter = 0
    
    for family in families
        for index in family
            if haskey(index_counts, index)
                index_counts[index] += 1
            else
                index_counts[index] = 1
            end
            small_counter += 1
        end
        stable = find_maximal_stable_sets(N+2, family)
        total += length(stable)
        for index in stable
            if haskey(index_counts, index)
                index_counts[index] += 1
            else
                index_counts[index] = 1
            end
        end
        if verbose 
            print("Family: ")
            println([subsets[item] for item in family])
            print(" Stable set: ")
            println([subsets_big[item-2^N] for item in stable])
        end
    end
    big_counter = total + SINGLETONS_IN_A1 + 1 + 1

    A1, Akm2 = define_families()
    small_counter += SINGLETONS_IN_A1 + 1 + 1
    for index in A1
        if haskey(index_counts, index)
            index_counts[index] += 1
        else
            index_counts[index] = 1
        end
    end
    for index in Akm2
        if haskey(index_counts, index)
            index_counts[index] += 1
        else
            index_counts[index] = 1
        end
    end
    pushfirst!(families, A1)  
    push!(families, Akm2)  
    collisions = sum(value > 1 ? value - 1 : 0 for value in values(index_counts))
    

    layeredness_score = measure_layeredness(families, adjacency_matrix)
    #println(subsets)

    if verbose
        println("Reward calculation:")
        println(index_counts)
        println(families)
        for family in families
            for item in family
                if item < 2^N 
                    print(subsets[item])
                else
                    print(subsets_big[item - 2^N])
                end
                print(", ")
            end
            println()
        end
        println("Scores:")
        println(collisions)
        println(layeredness_score)
        println(antichain_score)
        println(sum(length(subvector) for subvector in families) + total + 2) 
        #for family in families
        #    println(find_maximal_stable_sets(N+2, family))
        #end
        println()
        println()
        println("The final family is:\n")
        println("A0: Only the empty set.")
        println("A1:")
        for item in A1
            if item < 2^N 
                print(subsets[item])
            else
                print(subsets_big[item - 2^N])
            end
            print(", ")
        end
        for i in 2:length(families)-1
            println("\nA" * string(i) * ":")
            for item in families[i]
                if item < 2^N 
                    print(subsets[item])
                else
                    print(subsets_big[item - 2^N])
                end
                print(", ")
            end
            print("Stable sets: ")
            stable = find_maximal_stable_sets(N+2, families[i])
            for item in stable 
                if item < 2^N 
                    print(subsets[item])
                else
                    print(subsets_big[item-2^N])
                end
                print(", ")
            end
            println()
        end
        println("\nA"*string(K-2)*":")
        for item in Akm2
            if item < 2^N 
                print(subsets[item])
            else
                print(subsets_big[item - 2^N])
            end
            print(", ")
        end
        println()
        println("\nA"*string(K-1)*":")
        print("Only the entire set.\n")
        println("\nSmall sets: " * string(small_counter) * ", big sets: " * string(big_counter))
        println("\nTotal number of sets: " * string(sum(length(subvector) for subvector in families) + total + 2))
    end
    
    if layeredness_score == 0 && collisions == 0 && sum(antichain_score) == 0
        # Found a valid construction!!!
        if max(small_counter, big_counter) >= 2^(K-2)
            # Unfortunately this doesn't improve trivial bound
            return 2^(K-2) - max(small_counter, big_counter)
        else
            # We improved the trivial bound! Let's see by how much exactly, when K goes to infitiny
            # This can be compared across differing K and N
            # Goal: make this number as big as possible
            return 1 - log(2,max(small_counter, big_counter)) / (K-2)
        end
    end
    
    return -10*(100*layeredness_score + 10*collisions  + sum(antichain_score)) - max(small_counter, big_counter) + 2^(K-2)
    
end


function empty_starting_point()::OBJ_TYPE
    """
    If there is no input file, the search starts always with this object
    (e.g., empty graph, all zeros matrix, etc).
    Creates a string of length M with all zeros and changes up to 5 random positions to ones.
    """
    # Create an all-zero string of length M
    obj_string = fill('0', M)

    # Determine the number of positions to change (up to 5)
    num_changes = rand(5:30)

    # Select unique random positions to change
    positions = randperm(M)[1:num_changes]

    # Change selected positions to '1'
    for pos in positions
        obj_string[pos] = '1'
    end

    # Convert array of characters back to string
    return join(obj_string)
end
