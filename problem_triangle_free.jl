include("constants.jl")


function find_all_triangles(adjmat::Matrix{Int})
    N = size(adjmat, 1)
    triangles = []

    # Loop over all triples (i, j, k) where i < j < k
    for i in 1:N-2
        for j in i+1:N-1
            for k in j+1:N
                if adjmat[i, j] == 1 && adjmat[j, k] == 1 && adjmat[i, k] == 1
                    push!(triangles, (i, j, k))
                end
            end
        end
    end

    return triangles
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

function greedy_search_from_startpoint(obj::OBJ_TYPE)::OBJ_TYPE
    """
    Main greedy search algorithm. 
    It starts and ends with some construction 
    
    E.g. input: a graph which may or may not have triangles in it (these are the outputs of the transformer)
    Greedily remove edges to destroy all triangles, then greedily add edges without creating triangles
    Returns final maximal triangle-free graph
    """

    adjmat = zeros(Int, N, N)

    # Fill the upper triangular matrix
    index = 1
    for i in 1:N-1
        for j in i+1:N
            adjmat[i, j] = parse(Int, obj[index])  # Convert character to integer
            adjmat[j, i] = adjmat[i, j]  # Make the matrix symmetric
            index += 1
        end
    end

    triangles = find_all_triangles(adjmat)


    # Delete worst edge until no triangles are left
    while !isempty(triangles)
        # Count frequency of each edge in triangles
        edge_count = Dict()
        for (i, j, k) in triangles
            for edge in [(i, j), (j, k), (i, k)]
                edge_count[edge] = get(edge_count, edge, 0) + 1
            end
        end

        # Find the most frequent edge
        _, most_frequent_edge = findmax(edge_count)

        #println(triangles)
        #println(most_frequent_edge)
        #println(findmax(edge_count))
        #println(edge_count)

        # Remove this edge from the adjacency matrix
        i, j = most_frequent_edge
        adjmat[i, j] = 0
        adjmat[j, i] = 0

        # Update triangles by removing any that contain the most frequent edge
        triangles = filter(t -> !(most_frequent_edge in [(t[1], t[2]), (t[2], t[3]), (t[1], t[3])]), triangles)
    end


    #Now keep adding random edges without creating triangles, until stuck
    allowed_edges = Vector{Tuple{Int, Int}}()
    adjmat2 = adjmat * adjmat

    # Initial allowed edges calculation
    for i in 1:N-1
        for j in i+1:N
            if adjmat[i, j] == 0 && adjmat2[i, j] == 0
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
        """
        new_allowed_edges = Vector{Tuple{Int, Int}}()
        adjmat2 = adjmat * adjmat
        for (x, y) in allowed_edges
            if adjmat[x, y] == 0 && adjmat2[x, y] == 0
                push!(new_allowed_edges, (x, y))
            end
        end
        allowed_edges = new_allowed_edges
        """

        # Recalculate allowed edges faster
        # Update allowed_edges by filtering out edges that would form triangles with the new edge
        new_allowed_edges = Vector{Tuple{Int, Int}}()
        for edge in allowed_edges
            a, b = edge

            # Check if this edge shares a vertex with the new edge and forms a triangle
            if (a == i && adjmat[b, j] == 1) || (a == j && adjmat[b, i] == 1) ||
            (b == i && adjmat[a, j] == 1) || (b == j && adjmat[a, i] == 1)
                continue  # This edge would form a triangle, skip it
            end
            
            # Also remove the newly added edge itself if it's still in the list
            if (a == i && b == j) || (a == j && b == i)
                continue
            end

            # If it passes all checks, it remains in allowed_edges
            push!(new_allowed_edges, edge)
        end

        # Replace the old list with the new, filtered list
        allowed_edges = new_allowed_edges


    end
    return convert_adjmat_to_string(adjmat)
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
    return "0" ^ (N * (N - 1) ÷ 2 )
end
