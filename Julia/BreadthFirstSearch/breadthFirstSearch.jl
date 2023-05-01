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


mutable struct MovieGraph
    fileName::Union{String, Nothing}
    network::Union{Set, }
    movieTitles::Union{Set, Nothing}
    actorNames::Union{Set, Nothing}
end

MovieGraph( ) = MovieGraph( nothing, nothing, nothing, nothing )

function MovieGraph( fileName )

    io = open( fileName, "r" )
    S = readlines( io )
    close( io )

    movieTitles = Set{String}( )
    actorNames = Set{String}( )

    indexToMovieMap = Dict{Int, String}( )
    indexToActorMap = Dict{Int, String}( )
    movieToIndexMap = Dict{String, Int}( )
    actorToIndexMap = Dict{String, Int}( )
    duplicates = [ ]

    kk = length( S ) + 1
    ll = 1
    for ii in axes( S, 1 )
        currentMovie = strip( S[ ii ] )
        currentMovie = split( S[ ii ], "/" )

        currentTitle = currentMovie[ 1 ]
        push!( movieTitles, currentTitle )

        if !( haskey( movieToIndexMap, currentTitle ) )
            indexToMovieMap[ ll ] = currentTitle
            movieToIndexMap[ currentTitle ] = ll
            ll += 1
        else
            append!( duplicates, currentMovie )

        end

        for jj in eachindex( currentMovie )
            if jj == 1
                continue
            end
            currentActor = currentMovie[ jj ]
            push!( actorNames, currentActor )

            if !(haskey( actorToIndexMap, currentActor ) )
                actorToIndexMap[ currentActor ] = kk
                indexToActorMap[ kk ] = currentActor
                kk += 1
            end
        end
    end




    println( length( actorToIndexMap) )
    println( length( movieToIndexMap  ) )
    println( )
    println( length( indexToActorMap ) )
    println( length( indexToMovieMap ) )
    println( )
    println( length( actorNames ) )
    println( length( movieTitles ) )


    println( maximum( keys( indexToMovieMap ) ) )
    println( minimum( keys( indexToMovieMap )))
    println( maximum( keys( indexToActorMap) ) -  minimum( keys( indexToActorMap ) ) )

    kk = maximum( keys( indexToMovieMap ) ) + 1
    keys = deepcopy( indexToActorMap )

    # reindex dictionary to account for duplicated movie titles 
    for key in keys
        actor = pop!( indexToActorMap, key )
        indexToActorMap[ kk ] = actor
        actorToIndexMap[ actor ] = kk
        kk += 1
    end





    return nothing
end


MV = MovieGraph( "movie_data.txt" )


end