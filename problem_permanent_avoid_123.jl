include("constants.jl")

"""
Definitely use multithreading for this problem! The permanent calculation is the bottleneck.
    julia -t 8 search.jl
    (replace 8 by desired thread count)

    


Best possible constructions: https://oeis.org/A343844

f(1) = 1
f(2) = 2
f(3) = 4
f(4) = 8
f(5) = 16
f(6) = 32
f(7) = 64
f(8) = 120

f(9) >= 225
f(10) >= 424
f(11) >= 795
f(12) >= 1484
f(13) >= 2809
f(16) >= 18488
f(19) >= 122256
f(20) >= 227264
f(22) >= 794910
f(25) >= 5200384

"""

const N = 16


function ryser(A::AbstractMatrix)
    """computes the permanent of A using ryser with Gray ordering"""
        # code from https://discourse.julialang.org/t/matrix-permanent/10766
        # see Combinatorial Algorithms for Computers and Calculators, Second Edition (Computer Science and Applied Mathematics) by Albert Nijenhuis, Herbert S. Wilf (z-lib.org)
        # chapter 23 for the permanent algorithm
        # chapter 1 for the gray code

    function grayBitToFlip(n::Int)
        n1 = (n-1) ⊻ ((n-1)>>1)
        n2 = n ⊻ (n>>1)
        d = n1 ⊻ n2
        j = trailing_zeros(d) + 1 #returns the position to flip
        s = iszero(n1 & d) #returns the bit to be flipped
        j, s
    end

    n,m = size(A)
    if (n == m)
        D = true
        v = sum(A, dims = 2)
        v = Float64.(v) ./ 2.0 # Convert `v` to Float64 before dividing
        p = prod(v)
        @inbounds for i = 1:(2^(n-1)-1)
            a,s = grayBitToFlip(i)
            if s
                @simd for j=1:n
                    v[j] -= A[j,a]
                end
            else
                @simd for j=1:n
                    v[j] += A[j,a]
                end
            end
            pv = one(typeof(p))
            @simd for j=1:n
                pv *= v[j] #most expensive part
            end
            (D = !D) ? (p += pv) : (p -= pv)
        end
        return p * 2.0
    else
        throw(ArgumentError("perm: argument must be a square matrix"))
    end
end


function test_ordered_312(a,b,c)
    if a[2] < b[2] < c[2] && c[1] < a[1] < b[1]
        return true
    end
    return false
end

function is_312(a, b, c)
    if test_ordered_312(a,b,c) || test_ordered_312(a,c,b) || test_ordered_312(b,a,c) || test_ordered_312(b,c,a) || test_ordered_312(c,a,b) || test_ordered_312(c,b,a) 
        return true
    end
    return false
end



function convert_matrix_to_string(adjmat::Matrix{Int8})::String
    entries = []

    # Collect entries from the upper diagonal of the matrix
    for i in 1:N
        for j in 1:N
            push!(entries, string(adjmat[i, j]))
        end
        push!(entries, ",")
    end

    # Join all entries into a single string
    return join(entries)
end


const POINT_SET::Vector{Tuple{Int64, Int64}} = [(i, j) for i in 1:N for j in 1:N]


function find_all_312s(points)
    forb_patterns = Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Tuple{Int, Int}}}()

    for i in 1:length(points)
        for j in i+1:length(points)
            for k in j+1:length(points)
                if is_312(points[i],points[j],points[k])                    
                    push!(forb_patterns, (points[i], points[j], points[k]))
                end
            end
        end
    end

    return forb_patterns
end


function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    points = Vector{Tuple{Int64, Int64}}(undef, 0)
    num_commas = count(c -> c == ',', obj)
    if num_commas != N 
        return greedy_search_from_startpoint(db, empty_starting_point())
    end
    
    counter::Int64 = 1
    for (i,j) in POINT_SET::Vector{Tuple{Int64, Int64}} 
        while obj[counter] == ","
            counter += 1
        end
        if obj[counter] == '1'
            append!(points,[(i,j)])
        end
        counter += 1
    end
    
    forb_patterns = find_all_312s(points)

    # Delete worst edge until no triangles are left
    while !isempty(forb_patterns)
        # Count frequency of each edge in four_cycles
        point_count = Dict()
        for pattern in forb_patterns
            for point in pattern
                point_count[point] = get(point_count, point, 0) + 1
            end
        end

        # Find the most frequent point
        _, most_frequent_point = findmax(point_count)


        # Remove this point from the adjacency matrix
        i, j = most_frequent_point
        # Find the index of the element (i, j)
        index = findfirst(x -> x == (i, j), points)

        # If the element is found, delete it
        if index !== nothing
            deleteat!(points, index)
        end

        # Update contained 312s by removing any that contain the most frequent point
        forb_patterns = filter(t -> !(most_frequent_point in t), forb_patterns)
    end




    counter = 1
    points_added_counter = 0
    good::Bool = true
    random_point_set = shuffle(POINT_SET)
    for (i,j) in random_point_set
        if obj[counter] == '0'
            good = true
            for a in 1:length(points)
                for b in a+1:length(points)
                    if is_312(points[a],points[b],(i,j))
                        good = false
                        break
                    end
                end
                if good == false
                    break
                end
            end
            if good
                append!(points,[(i,j)])
                points_added_counter += 1
            end
        end
        counter += 1
    end

    adjmat = zeros(Int8, N, N)
    counter = 1
    for (i,j) in POINT_SET::Vector{Tuple{Int64, Int64}} 
        if (i,j) in points
            adjmat[i,j] = 1
        end
        counter += 1
    end

    return [convert_matrix_to_string(adjmat)]
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    adjmat = zeros(Int8, N, N)
    counter::Int = 1
    for (i,j) in POINT_SET::Vector{Tuple{Int64, Int64}} 
        if obj[counter] == '1'
            adjmat[i,j] = 1
        end
        counter += 1
    end
    
    return ryser(adjmat)
end


function empty_starting_point()::OBJ_TYPE
    """
    If there is no input file, the search starts always with this object
    (E.g. empty graph, all zeros matrix, etc)
    """
    adjmat = zeros(Int8, N, N)
    for i in 1:N 
        adjmat[i,i] = 1
    end
    return convert_matrix_to_string(adjmat)
end
