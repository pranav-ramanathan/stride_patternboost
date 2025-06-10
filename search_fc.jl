using Base.Threads
using Random
using LinearAlgebra
using Statistics
import StatsBase: countmap 
using Dictionaries
using Printf
using Plots
using Combinatorics
using Dates 

# Choose the problem to work on here!

include("problem_triangle_free.jl")  
# include("problem_4_cycle_free.jl")
#include("problem_permanent_avoid_123.jl")


#########################################################################################

include("constants.jl")

if Threads.nthreads() > 1 
    BLAS.set_num_threads(1)   # this seems to help a bit
end

println("Using ", nthreads(), " thread(s)")

function find_next_available_filename(base::String, extension::String)
    i = 1
    while true
        filename = @sprintf("%s/%s_%d.%s", write_path, base, i, extension)
        if !isfile(filename)
            return filename
        end
        i += 1
    end
end

function write_output_to_file(db)
    rewards = [ rew for rew in keys(db.rewards) ]
    sort!(rewards, rev=true)
    base_name = "search_output"
    extension = "txt"
    filename = find_next_available_filename(base_name, extension)
    curr_rew_index = 1
    curr_rew = rewards[1]
    lines_written::Int = 0
    open(filename, "w") do file
        while lines_written < final_database_size && curr_rew_index <= length(rewards)
            curr_rew = rewards[curr_rew_index]
            for obj in db.rewards[curr_rew][1:min(final_database_size - lines_written, length(db.rewards[curr_rew]))]
                write(file, obj * "\n")
            end
            lines_written += length(db.rewards[curr_rew])
            curr_rew_index += 1
        end
        
    end
    println("Data written to $(filename)")
    println("An example of an object with maximum reward (" * string(rewards[1]) * "):")
    println(db.rewards[rewards[1]][1])
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

    # Create the .txt file and write the score distribution
    txt_filename = filename = @sprintf("%s/%s.%s", write_path, "distribution", "txt")
    open(txt_filename, "w") do f
        for (rew, count) in zip(rewards, reward_counts)
            println(f, "Score: $rew, Count: $count")
        end
    end
    println("Score distribution saved to $(txt_filename)")

    # Print the training set as well
    cumulative_count = 0
    filtered_rewards = []
    filtered_counts = []
    
    # Filter rewards to only consider up to the best final_database_size objects
    for (rew, count) in zip(rewards, reward_counts)
        if cumulative_count >= final_database_size
            break
        end
        next_count = min(count, final_database_size - cumulative_count)
        push!(filtered_rewards, rew)
        push!(filtered_counts, next_count)
        cumulative_count += next_count
    end

    # Create the plot with filtered data
    bar(filtered_rewards, filtered_counts, xlabel="Scores", ylabel="Count", title="Score Distribution", legend=false)
    
    # Find a filename for saving the plot
    base_name = "plot_training"
    extension = "png"
    filename = find_next_available_filename(base_name, extension)
    
    # Save the plot to file
    savefig(filename)
    println("Plot saved to $(filename)")

    # Create the .txt file for the filtered data
    txt_filename = @sprintf("%s/%s.%s", write_path, "training_distribution", "txt")
    open(txt_filename, "w") do f
        for (rew, count) in zip(filtered_rewards, filtered_counts)
            println(f, "Score: $rew, Count: $count")
        end
    end
    println("Filtered score distribution saved to $(txt_filename)")
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
        for _ in 1:num_initial_empty_objects
            push!(lines, empty_starting_point())
        end
    end
    return lines
end


# The below functions are a modification of a local search code originally written by Gwenael Joret

function reward(obj)
    return reward_calc(obj)
end

function reward(db, obj)
    if haskey(db.objects, obj)
        return db.objects[obj], false
    end
    return reward(obj), true
end

function local_search_on_object(db, obj)
    objects = Vector{OBJ_TYPE}(undef, 0)
    rewards = Vector{REWARD_TYPE}(undef, 0) 
    greedily_expanded_obj::OBJ_TYPE = greedy_search_from_startpoint(db, obj) 
    rew, new = reward(db, greedily_expanded_obj)
    if new
        push!(objects, greedily_expanded_obj)
        push!(rewards, rew)
    end
    return objects, rewards
end


function print_db(db)
    rewards = [ rew for rew in keys(db.rewards) ]
    sort!(rewards, rev=true)
    db_size = 0
    for r in rewards 
        db_size += length(db.rewards[r])
    end
    # shrink database if necessary
    if db_size > 2*target_db_size
        println(" - Shrinking database to $target_db_size best objects")
        shrink!(db)
        rewards = [ rew for rew in keys(db.rewards) ]
        sort!(rewards, rev=true)
    end  
end


function local_search!(db, lines, start_ind, nb=nb_local_searches)
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


function shrink!(db)
    # shrinks the database to the target number of objects
    count = 0        
    rewards = [ rew for rew in keys(db.rewards) ]
    sort!(rewards, rev=true) 
    for rew in rewards
        if count < target_db_size
            lg = length(db.rewards[rew])
            count += lg 
            if count > target_db_size
                k = count - target_db_size
                for obj in db.rewards[rew][lg-k+1:end]                  
                    try
                        delete!(db.objects, obj)
                    catch e 
                        println("whoopsie")
                    end
                end
                db.rewards[rew] = db.rewards[rew][1:lg-k]
                db.local_search_indices[rew] = min(db.local_search_indices[rew], lg-k)
            end
        else
            for obj in db.rewards[rew]
                unset!(db.objects, obj)
            end
            delete!(db.rewards, rew)
            delete!(db.local_search_indices, rew)
        end
    end    
    return nothing
end

function main()
    db = new_db() 
    lines = initial_lines()
    println(length(lines))
    println(length(Set(lines)))
    print("Using ")
    print(nthreads())
    println(" threads")
    #add_db!(db, lines)
    start_idx = 1
    steps::Int = 0
    time_since_previous_output = 0
    while start_idx < length(lines)
        time_local_search = @elapsed local_search!(db, lines, start_idx)
        time_since_previous_output += time_local_search
        start_idx += nb_local_searches
        steps += 1
        time_local_search = round(time_local_search, digits=2)
        print_db(db)        
    end
    print_db(db)
    write_output_to_file(db)
    write_plot_to_file(db)
end


write_path = ARGS[1]
nb_local_searches = parse(Int,ARGS[2]) 
num_initial_empty_objects = parse(Int,ARGS[3])
final_database_size = parse(Int,ARGS[4])
target_db_size = parse(Int,ARGS[5])
main()





