using LightGraphs
using Base.Threads
using Random
using LinearAlgebra
using Statistics
import StatsBase: countmap 
using Dictionaries
using Printf
using Plots




const N::Int = 20
const NB_LOCAL_SEARCHES::Int64 = 1200  # number of local searches to do at once. probably best to set it divisible by number of threads
const NUM_INITIAL_EMPTY_OBJECTS = 100_000 #on the first run, this many rollouts will be done from empty graphs/objects
const FINAL_DATABASE_SIZE = 100_000

const OBJ_TYPE = String  # type for vectors encoding constructions
const REWARD_TYPE = Float32  # type for rewards 



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

        # Recalculate allowed edges
        new_allowed_edges = Vector{Tuple{Int, Int}}()
        adjmat2 = adjmat * adjmat
        for (x, y) in allowed_edges
            if adjmat[x, y] == 0 && adjmat2[x, y] == 0
                push!(new_allowed_edges, (x, y))
            end
        end
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



#########################################################################################


function find_next_available_filename(base::String, extension::String)
    i = 1
    while true
        filename = @sprintf("%s_%d.%s", base, i, extension)
        if !isfile(filename)
            return filename
        end
        i += 1
    end
end

function write_output_to_file(db)
    rewards = [ rew for rew in keys(db.rewards) ]
    sort!(rewards, rev=true)
    base_name = "greedy_output"
    extension = "txt"
    filename = find_next_available_filename(base_name, extension)
    curr_rew_index = 1
    curr_rew = rewards[1]
    lines_written::Int = 0
    open(filename, "w") do file
        while lines_written < FINAL_DATABASE_SIZE
            curr_rew = rewards[curr_rew_index]
            for obj in db.rewards[curr_rew][1:min(FINAL_DATABASE_SIZE - lines_written, length(db.rewards[curr_rew]))]
                write(file, obj * "\n")
            end
            lines_written += length(db.rewards[curr_rew])
            curr_rew_index += 1
        end
        
    end
    println("Data written to $(filename)")
end

function write_plot_to_file(db)
    rewards = [ rew for rew in keys(db.rewards) ]
    sort!(rewards, rev=true)
    reward_counts = [ length(db.rewards[rew]) for rew in rewards ]

    # Create the plot
    bar(rewards, reward_counts, xlabel="Scores", ylabel="Count", title="Score Distribution", legend=false)
    
    # Find a filename for saving the plot
    base_name = "plot"
    extension = "png"
    filename = find_next_available_filename(base_name, extension)
    
    # Save the plot to file
    savefig(filename)
    println("Plot saved to $(filename)")
end


function new_db()
    return Database(Dictionary{OBJ_TYPE, REWARD_TYPE}(), Dictionary{REWARD_TYPE, Vector{OBJ_TYPE}}(), Dictionary{REWARD_TYPE, UInt}())
end


function initial_lines()
    input_file = ""
    for arg in ARGS
        if arg == "-i" || arg == "--input"
            input_file_index = findfirst(==(arg), ARGS) + 1
            if input_file_index <= length(ARGS)
                input_file = ARGS[input_file_index]
            end
            break
        end
    end
    println("Input file: ", input_file)  # Debug print

    lines = String[]  # Create an empty vector of strings
    if input_file != ""
        println("Using input file")
        open(input_file, "r") do file
            for line in eachline(file)
                if length(line) == length(empty_starting_point())
                    push!(lines, line)  # Add each line to the vector
                end
            end
        end
    else 
        println("No input file provided")
        for _ in 1:NUM_INITIAL_EMPTY_OBJECTS
            push!(lines, empty_starting_point())
        end
    end
    return lines
end



function reward(obj)
    # version without dictionary
    # computes the reward for obj 
    return reward_calc(obj)
end

function reward(db, obj)
    # version with dictionary
    # returns reward of obj and a bool indicating whether it's a new graph
    if haskey(db.objects, obj)
        return db.objects[obj], false
    end
    # obj is not in the dictionary yet    
    return reward(obj), true
end

function local_search_on_object(db, obj)
    # naive version of local search on object  
    #allows for multiple rollouts from a single starting point 
    objects = Vector{OBJ_TYPE}(undef, 0)
    rewards = Vector{REWARD_TYPE}(undef, 0)
    
    
    greedily_expanded_obj = greedy_search_from_startpoint(obj)
    rew, new = reward(db, greedily_expanded_obj)
    if new
        push!(objects, greedily_expanded_obj)
        push!(rewards, rew)
    end
    return objects, rewards
end


function print_db(db)
    nb_top = 20
    println("Database:")
    rewards = [ rew for rew in keys(db.rewards) ]
    sort!(rewards, rev=true)
    max_size_for_r = 5
    db_size = 0
    for r in rewards 
        r_round = round(r, digits=4)
        s = "$r_round:"
        max_size_for_r = max(max_size_for_r, sizeof(s))
        db_size += length(db.rewards[r])
    end
    println(" - $db_size objects") 
    println(" - Distribution for top $nb_top rewards:")
    top_size = 0
    for r in rewards[1:min(nb_top, length(rewards))]
        top_size += length(db.rewards[r])
    end
    for r in rewards[1:min(nb_top, length(rewards))]
        length_line = 75                
        multiplicity = length(db.rewards[r])
        num = Int(round( (multiplicity/top_size )*length_line ))
        r_round = round(r, digits=4)
        pad = repeat(" ", max_size_for_r - sizeof("$r_round"))
        s = "$r_round:" * pad * "[" * repeat("=", num) * repeat(".", length_line - num) * "] $multiplicity"
        println(s)
    end     
end



function local_search!(db, lines, start_ind, nb=NB_LOCAL_SEARCHES)
    local_search_results_threads = []
    for j=1:nthreads()
        push!(local_search_results_threads, [[],[]])
    end
    # prepare local search pool
    count = 0
    pool = OBJ_TYPE[]
    append!(pool, lines[start_ind:min(start_ind + nb - 1,length(lines))])
            

    # we perform the local searches
    @threads for obj in pool
        list_obj, list_rew = local_search_on_object(db, obj)
        append!(local_search_results_threads[threadid()][1], list_obj)
        append!(local_search_results_threads[threadid()][2], list_rew)
    end
    # we update the dictionaries
    for j=1:nthreads()
        # we consider all new graphs found by j-th thread
        # Remark: a tiny number of graphs could be found by multiple threads, this is not a problem, the function add! will add each graph only once
        add_db!(db, local_search_results_threads[j][1], local_search_results_threads[j][2])
    end
    return nothing
end


struct Database
    # encapsulates dictionaries that are used in various places
    objects::Dictionary{OBJ_TYPE, REWARD_TYPE}
    rewards::Dictionary{REWARD_TYPE, Vector{OBJ_TYPE}}
    local_search_indices::Dictionary{REWARD_TYPE, UInt}  # encodes last indices for which local search has been performed, per reward (unused for now)
end

function add_db!(db, list_obj, list_rew = nothing)
    # add all objects in list_obj to the database (if not already there)
    # computes the rewards if not provided
    # returns the number of new objects added to the database    
    rewards_new_objects = []
    
    if list_rew != nothing
        for i in 1:length(list_obj)
            obj = list_obj[i]
            if !haskey(db.objects, obj)       
                rew = list_rew[i]         
                push!(rewards_new_objects, rew)
                #db.objects[obj] = rew
                set!(db.objects, obj, rew)
                if !haskey(db.rewards, rew)
                    insert!(db.rewards,rew,[obj])
                    insert!(db.local_search_indices, rew, 0)
                else
                    push!(db.rewards[rew], obj)
                end
            end
        end
    else 
        # compute rewards using multithreading if rewards are not provided
        # first we identify the new objects
        list_indices = Int[]
        for i in 1:length(list_obj)
            obj = list_obj[i]
            if !haskey(db.objects, obj)
                push!(list_indices, i)
            end
        end
        # next we compute the rewards using multithreading
        list_rew = zeros(Float32, length(list_obj))
        @threads for i in list_indices
            list_rew[i] = reward(list_obj[i])
        end
        # finally we update the db 
        for i in list_indices
            obj = list_obj[i]          
            rew = list_rew[i] 
            push!(rewards_new_objects, list_rew[i])
            set!(db.objects, obj,rew)
            if !haskey(db.rewards, rew)
                insert!(db.rewards,rew,[obj])
                insert!(db.local_search_indices, rew, 0)
            else
                push!(db.rewards[rew], obj)
            end            
        end
    end
    return rewards_new_objects
end


function main()
    db = new_db()
    
    lines = initial_lines()
    #add_db!(db, lines)

    start_idx = 1

    steps::Int = 0
    while start_idx < length(lines)
        time_local_search = @elapsed local_search!(db, lines, start_idx)
        start_idx += NB_LOCAL_SEARCHES
        steps += 1
        time_local_search = round(time_local_search, digits=2)
        print_db(db)
        println("\nTime elapsed: local search = $time_local_search s. \n")
    end
    print_db(db)
    write_output_to_file(db)
    write_plot_to_file(db)
end

main()





