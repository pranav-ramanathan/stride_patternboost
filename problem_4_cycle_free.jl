include("constants.jl")
using ArgParse


"""
Best possible constructions: https://oeis.org/A006855/list

N:    1  2  3  4  5  6  7  8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34   35   36   37   38   39   40    
f(N): 0, 1, 3, 4, 6, 7, 9, 11, 13, 16, 18, 21, 24, 27, 30, 33, 36, 39, 42, 46, 50, 52, 56, 59, 63, 67, 71, 76, 80, 85, 90, 92, 96, 102, 106, 110, 113, 117, 122, 127

"""

#function parse_args()
#    s = ArgParseSettings()
#    @add_arg_table s begin
#        "-N", "--number"
#        help = "specifies the value of N"
#        arg_type = Int
#        default = 20  # default value if -N is not provided
#    end
#    return parse_args(s)
#end

#args = parse_args()
const N = 33 #get(args, :number, 20)



function find_all_four_cycles(adjmat::Matrix{Int})
    N = size(adjmat, 1)
    four_cycles = Vector{Tuple{Int8, Int8, Int8, Int8}}()

    # Loop over all quadruples (a, b, c, d) where a < b < c < d
    for a in 1:N
        for b in a+1:N
            for c in a+1:N
                for d in b+1:N
                    if adjmat[a, b] == 1 && adjmat[b, c] == 1 && adjmat[c, d] == 1 && adjmat[d, a] == 1
                        push!(four_cycles, (a, b, c, d))
                    end
                end
            end
        end
    end

    return four_cycles
end




function convert_adjmat_to_string(adjmat::Matrix{Int})::String
    entries = []

    # Collect entries from the upper diagonal of the matrix
    for i in 1:N-1
        for j in i+1:N
            push!(entries, string(adjmat[i, j]))
        end
        push!(entries,"2")
    end

    # Join all entries into a single string
    return join(entries)
end

function ordered(edge)::Tuple{Int, Int}
    if edge[1] <= edge[2]
        return edge
    end
    return (edge[2], edge[1])
end



function greedy_search_from_startpoint(db, obj::OBJ_TYPE, additional_loops=0)::Vector{OBJ_TYPE}
    """
    Main greedy search algorithm. 
    It starts and ends with some construction 
    
    E.g. input: a graph which may or may not have triangles in it (these are the outputs of the transformer)
    Greedily remove edges to destroy all triangles, then greedily add edges without creating triangles
    Returns final maximal triangle-free graph
    """
    num_commas = count(c -> c == ',', obj)
    if num_commas != N - 1
        return greedy_search_from_startpoint(db, empty_starting_point())
    end

    adjmat = zeros(Int, N, N)

    # Fill the upper triangular matrix
    index = 1
    for i in 1:N-1
        for j in i+1:N
            while obj[index] == ','
                index += 1
            end
            #println(obj[index])
            adjmat[i, j] = parse(Int, obj[index])  # Convert character to integer
            adjmat[j, i] = adjmat[i, j]  # Make the matrix symmetric
            index += 1
        end
    end

    four_cycles = find_all_four_cycles(adjmat)


    # Delete worst edge until no triangles are left
    while !isempty(four_cycles)
        # Count frequency of each edge in four_cycles
        edge_count = Dict()
        for (i, j, k, l) in four_cycles
            for edge in [(i, j), (j, k), (k, l), (i, l)]
                edge_count[ordered(edge)] = get(edge_count, ordered(edge), 0) + 1
            end
        end

        # Find the most frequent edge
        _, most_frequent_edge = findmax(edge_count)


        # Remove this edge from the adjacency matrix
        i, j = most_frequent_edge
        adjmat[i, j] = 0
        adjmat[j, i] = 0

        # Update four_cycles by removing any that contain the most frequent edge
        four_cycles = filter(t -> !(most_frequent_edge in [(t[1], t[2]), ordered((t[2], t[3])), ordered((t[3], t[4])), (t[1], t[4])]), four_cycles)
        #println(length(four_cycles), most_frequent_edge, four_cycles)
    end


    #Now keep adding random edges without creating four_cycles, until stuck
    allowed_edges = Vector{Tuple{Int, Int}}()
    adjmat3 = adjmat * adjmat * adjmat 

    # Initial allowed edges calculation
    for i in 1:N-1
        for j in i+1:N
            if adjmat[i, j] == 0 && adjmat3[i, j] == 0
                push!(allowed_edges, (i, j))
            end
        end
    end

    # Continue until no allowed edges are left
    while !isempty(allowed_edges)
        # Randomly select an edge to add
        edge = allowed_edges[rand(1:length(allowed_edges))]
        i, j = edge
        adjmat[i, j] = 1
        adjmat[j, i] = 1

        # Recalculate allowed edges (slow version)
        new_allowed_edges = Vector{Tuple{Int, Int}}()
        adjmat3 = adjmat * adjmat * adjmat
        for (x, y) in allowed_edges
            if adjmat[x, y] == 0 && adjmat3[x, y] == 0
                push!(new_allowed_edges, (x, y))
            end
        end
        allowed_edges = new_allowed_edges
    end

    return [convert_adjmat_to_string(adjmat)]


    # Now that we have 'adjmat', sample four random permutations
    permuted_adjmats = []
    for _ in 1:4
        perm = randperm(N)  # Generate a random permutation
        permuted_adjmat = adjmat[perm, perm]  # Apply the permutation to rows and columns
        push!(permuted_adjmats, permuted_adjmat)
    end


    return [convert_adjmat_to_string(permuted_adjmat) for permuted_adjmat in permuted_adjmats]
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    """
    Function to calculate the reward of a final construction
    (E.g. number of edges in a graph, etc)
    """
    return count(isequal('1'), obj)
end


function empty_starting_point()::OBJ_TYPE
    """
    If there is no input file, the search starts always with this object
    (E.g. empty graph, all zeros matrix, etc)
    """

    adjmat = zeros(Int, N, N)

    return convert_adjmat_to_string(adjmat)
end
