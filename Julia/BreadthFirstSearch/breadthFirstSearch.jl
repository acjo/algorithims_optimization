# breadthFristSearch.jl


module BreadthFirstSeach

using DataStructures, Graphs
export MyGraph, addNode!, println, addEdge!, removeNode!, removeEdge!, traverse, shortestPath
import Base.println

struct MyGraph
    adjacency::Dict{Any, Set}
end

MyGraph() = MyGraph( Dict{Any, Set}( ) )

function println( io::IO, G::MyGraph )

    println( G.adjacency )

    return nothing
end

function addNode!( G::MyGraph, n )

    if !( n in keys( G.adjacency ) )
        G.adjacency[ n ] = Set( )
    end
    return nothing
end

function addEdge!( G::MyGraph, u, v )


    addNode!( G, u )
    addNode!( G, v )


    push!( G.adjacency[ u ], v )
    push!( G.adjacency[ v ], u )

    return nothing
end

function removeNode!( G::MyGraph, n )

    if !( n in keys( G.adjacency ) )
        raise(KeyError("node is not contained in graph"))
    end

    for node in keys( G.adjacency )
        if n in G.adjacency[ node ]
            pop!( G.adjacency[ node ], n )
        end
    end

    pop!( G.adjacency, n )

    return nothing
end

function removeEdge!( G::MyGraph, u, v )

    if !( u in keys( G.adjacency ) ) || !( v in keys( G.adjacency ) )
        throw( KeyError( "At least one of the given nodes is not in the graph." ) )
    end

    pop!( G.adjacency[ u ], v )
    pop!( G.adjacency[ v ], u )

    return nothing
end

function traverse( G::MyGraph, source )

    if !( source in keys( G.adjacency ) )
        throw(KeyError( "source node is not present in graph." ) )
    end

    V = [ ]
    M = Set( source )
    Q = Deque{Any}( )
    push!( Q, source )

    while !isempty( Q )
        node = popfirst!( Q )
        append!( V, node )
        for neighbor in G.adjacency[ node ]
            if !( neighbor in M )
                push!( M, neighbor )
                pushfirst!( Q, neighbor )
            end
        end
    end

    return V
end

function shortestPath( G::MyGraph, source, target )

    if !(source in keys( G.adjacency ) ) || !( target in keys( G.adjacency ) ) 
        throw( KeyError( "either source or target node is not present in Graph." ) )
    end

    V = [ ]
    M = Set( source )
    Q = Deque{Any}( )
    push!( Q, source )
    all_paths = Dict{Any, Any}( )

    while !isempty( Q )
        current = popfirst!( Q )
        append!( V, current )
        for neighbor in G.adjacency[ current ]
            if !( neighbor in M )
                if neighbor == target
                    all_paths[ neighbor ] = current
                    final_path = Deque{Any}( )
                    pushfirst!( final_path, neighbor )
                    while neighbor in keys( all_paths )
                        pushfirst!( final_path, all_paths[ neighbor ] )
                        neighbor = all_paths[ neighbor ]
                    end

                    return final_path
                end
                pushfirst!( Q, neighbor )
                push!( M, neighbor )
                all_paths[ neighbor ] = current
            end
        end
    end
end


struct MovieGraph
    # filename for the data
    fileName::Union{String, Nothing}
    # the network
    network::Union{Graph, Nothing}
    # tracks graph index to the movie name
    indexToMovieMap::Union{Dict{Int, String}, Nothing}
    # tracks graph index to the actor name
    indexToActorMap::Union{Dict{Int, String}, Nothing} 
    # tracks movie title to the graph index
    movieToIndexMap::Union{Dict{String, Int}, Nothing}
    # tracks actor name to the graph index
    actorToIndexMap::Union{Dict{String, Int}, Nothing}
    # tracks what movie has what actors in it
    movieToActorMap::Union{Dict{String, Vector{String}}, Nothing} 
    # tracks which actor is in what movies
    actorToMovieMap::Union{Dict{String, Vector{String}}, Nothing}
end

MovieGraph( ) = MovieGraph( nothing, nothing, nothing, nothing )

function MovieGraph( fileName )

    # open file, read in the lines, and close the file
    io = open( fileName, "r" )
    S = readlines( io )
    close( io )

    # instantiate graph
    G = Graph( )

    # tracks graph index to the movie name
    indexToMovieMap = Dict{Int, String}( )
    # tracks graph index to the actor name
    indexToActorMap = Dict{Int, String}( )
    # tracks movie title to the graph index
    movieToIndexMap = Dict{String, Int}( )
    # tracks actor name to the graph index
    actorToIndexMap = Dict{String, Int}( )
    # tracks what movie has what actors in it
    movieToActorMap = Dict{ String, Vector{ String}}( )
    # tracks which actor is in what movies
    actorToMovieMap = Dict{String, Vector{String}}( )

    # intialize graph vertex index 
    ll = 1
    # iterate through all movies
    for ii in axes( S, 1 )

        # get the current movie and split by forward slash
        currentMovie = strip( S[ ii ] )
        currentMovie = split( S[ ii ], "/" )

        # extract the current title
        currentTitle = currentMovie[ 1 ]

        # if the current title is not contained in the movie dictionaries
        if !( haskey( movieToActorMap, currentTitle ) )
            # create a key value pair where the key is the movie and the value are the actors
            movieToActorMap[ currentTitle ] = currentMovie[ 2:end ]
            # creat a key value pair where the key is the movie and the value is the graph index
            movieToIndexMap[ currentTitle ] = ll
            # create a key value pair where the key is the graph index and the value is the movie title
            indexToMovieMap[ ll ] = currentTitle
            # add one vertex to G
            add_vertex!( G )
            # increment the graph index to match the next vertex we add to G
            ll += 1
        # if the current title is contained in the movie dictionaries
        else
            # get the old actors this movie pointed to 
            old = movieToActorMap[ currentTitle ]
            # instantiate a new empty vector of strings
            combined = Vector{String}()
            # add all the old actors
            append!( combined, old )
            # add all the new actors
            append!( combined, currentMovie[ 2:end ] )
            # map the movie title to the unqiue actors of the movie 
            movieToActorMap[ currentTitle ] = unique( combined )
        end

        # iterate throughe each actor of the current movie
        for jj in eachindex( currentMovie[ 2:end ] )

            # extract the current actor
            currentActor = currentMovie[ jj + 1 ]
            
            # if the current actor is not contained in the actor dictionaries
            if !( haskey( actorToMovieMap, currentActor ) )
                # create a key value pair where the key is the current actor and the value is the current movie title 
                actorToMovieMap[ currentActor ] = [ currentTitle ] 
                # create a key value pair where the key is the current actor and the value is the graph index
                actorToIndexMap[ currentActor ] = ll
                # create a key value pair where the key is the graph index and the value is the current actor
                indexToActorMap[ ll ] = currentActor
                # add one vertex to G
                add_vertex!( G )
                # incrrement the graph index to match the next vertex we add to G
                ll += 1
            # if the current actor is contained in the acotr dictionaries
            else
                # get the old movie title(s) mapped to by this actor
                old = actorToMovieMap[ currentActor ]
                # instantiate an empty vector of strings
                combined = Vector{String}( )
                # add the old title(s)
                append!( combined, old )
                # add the new title
                append!( combined, [ currentTitle ] )
                # map the current actor to the unique movie titles they have acted in
                actorToMovieMap[ currentActor ] = unique( combined )
            end

            # add edge from current actor <==> current movie 
            add_edge!( G, actorToIndexMap[ currentActor ], movieToIndexMap[ currentTitle ] )
        end
    end

    return MovieGraph( fileName, G, indexToMovieMap, indexToActorMap, movieToIndexMap, actorToIndexMap, movieToActorMap, actorToMovieMap ) 
end


function pathToActor( G::MovieGraph, source, target )

    if typeof( source ) === String
        sourceIndex = G.actorToIndexMap[ source ]
    else
        sourceIndex = source
    end

    if typeof( target ) === String
        targetIndex = G.actorToIndexMap[ target ]
    else
        targetIndex = target
    end

    ds = dijkstra_shortest_paths( G.network, sourceIndex )


    return nothing
end

MV = MovieGraph( "movie_data.txt" )


end