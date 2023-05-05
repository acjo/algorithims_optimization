# breadthFristSearch.jl


module BreadthFirstSeach

using Graphs
using Plots
using DataFrames
using Statistics: mean
using DataStructures

export MyGraph, addNode!, println, repr, addEdge!, removeNode!, removeEdge!, traverse, shortestPath, MovieGraph, pathToActor, averageNumber 
import Base.println, Base.repr

struct MyGraph
    adjacency::Dict{Any, Set}
end

MyGraph() = MyGraph( Dict{Any, Set}( ) )

function println( io::IO, G::MyGraph )

    println( MyGraph )
    println( G.adjacency )

    return nothing
end

function repr( G::MyGraph )

    stringRep = repr( MyGraph )
    stringRep *= "\n"
    stringRep *= repr( G.adjacency )

    return stringRep
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

    V = Any[ ]
    M = Set{Any}( ) 
    push!( M, source )
    Q = Deque{Any}( )
    push!( Q, source )

    while !isempty( Q )
        currentNode = popfirst!( Q )
        append!( V, [ currentNode ] )
        for neighbor in G.adjacency[ currentNode ]
            if !( neighbor in M )
                push!( M, neighbor )
                push!( Q, neighbor )
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
    M = Set{Any}( )
    push!( M, source )
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
    # contains movie titles
    movieTitles::Union{Set{String}, Nothing}
    # contains actor names
    actorNames::Union{Set{String}, Nothing}
    # maps name (of actor or movie) to to network node index
    nameToIndex::Union{Dict{String, Int}, Nothing}
    # maps index of network node to name (of actor or movie)
    indexToName::Union{Dict{Int, String}, Nothing}
end

MovieGraph( ) = MovieGraph( nothing, nothing, nothing, nothing, nothing, nothing )

function MovieGraph( fileName )

    io = open( fileName, "r" )
    S = readlines( io )
    close( io )


    G = Graph( )
    movieTitles = Set{String}( )
    actorNames = Set{String}( )
    nameToIndex = Dict{String,Int}( )
    indexToName = Dict{Int,String}( )

    ll = 1
    for ii in eachindex( S )
        currentMovie = split( strip( S[ ii ] ), "/" )
        currentTitle = currentMovie[ 1 ]

        push!( movieTitles, currentTitle )
        if !haskey( nameToIndex, currentTitle )
            nameToIndex[ currentTitle ] = ll
            indexToName[ ll ] = currentTitle
            add_vertex!( G )
            ll += 1
        end
        movieIndex = nameToIndex[ currentTitle ]

        
        for jj in eachindex( currentMovie[ 2:end ] )
            currentActor = currentMovie[ jj+1 ]
            push!( actorNames, currentActor )
            if !haskey( nameToIndex, currentActor )
                nameToIndex[ currentActor ] = ll
                indexToName[ ll ] = currentActor
                add_vertex!( G )
                ll += 1
            end
            actorIndex = nameToIndex[ currentActor ]

            add_edge!( G, actorIndex, movieIndex )

        end
    end

    return MovieGraph( fileName, G, movieTitles, actorNames, nameToIndex, indexToName )
end


function pathToActor( G::MovieGraph, source::Union{String, Int}, target::Union{String,Int} )

    if typeof( source ) === String
        sourceIndex = G.nameToIndex[ source ]
    else
        if source in keys( G.indexToName )
            sourceIndex = source
        else
            throw( KeyError( source ) )
        end
    end

    if typeof( target ) === String
        targetIndex = G.nameToIndex[ target ]
    else
        if target in keys( G.indexToName )
            targetIndex = target
        else
            throw( KeyError( target ) )
        end
    end

    ds = desopo_pape_shortest_paths( G.network, sourceIndex )

    spath( x, r, s ) = x == s ? x : [ spath( r.parents[ x ], r, s ) x ]

    indexPath = spath( targetIndex, ds, sourceIndex )

    path = String[ ]
    for ii in indexPath
        append!( path, [ G.indexToName[ ii ] ] )
    end
    
    return path, length( path ) - 1
end

function averageNumber( G::MovieGraph, target::Union{String,Int} )

    if typeof( target ) === String
        targetIndex = G.nameToIndex[ target ]
    else
        if target in keys( G.indexToName )
            targetIndex = target
        else
            throw( KeyError( target ) )
        end
    end

    actorDistances = []
    ds = desopo_pape_shortest_paths( G.network, targetIndex )

    actorDistances = [ Int( floor( ds.dists[ G.nameToIndex[ name ] ] / 2) ) for name in G.actorNames ]


    avgDist = mean( actorDistances )


    p = histogram( actorDistances; xlabel="actor Distances", normalize=:pdf, bins=collect( 1:maximum( actorDistances ) ).- 0.5 )
    display( p )

    return avgDist 
end


end