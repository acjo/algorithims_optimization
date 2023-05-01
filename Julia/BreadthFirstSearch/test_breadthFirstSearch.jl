include("breadthFirstSearch.jl")

using .BreadthFirstSeach
using DataStructures


function test_graph()

    G = MyGraph( )
    @assert G.adjacency == Dict{Any, Set}() 
    return nothing
end

function test_println( )

    d = Dict( 1 => Set( 2 ) )

    G = MyGraph( d )

    println( G )


    return nothing
end

function test_addNode!( )

    G = MyGraph( )

    addNode!(G, 1)

    @assert G.adjacency == Dict{Any, Set}( 1 => Set() )


    d = Dict( 'a' => Set( 'b' ) )
    G = MyGraph( d )
    addNode!( G, 'u' )

    @assert G.adjacency == Dict{Any, Set}( 'a' => Set('b'), 'u' => Set( ) )

    return nothing

end

function test_addEdge!( )

    G = MyGraph( )
    addEdge!( G, 'u', 'v')
    @assert G.adjacency ==  Dict{Any, Set}( 'u' => Set( 'v' ), 'v' => Set( 'u' ) )
    addEdge!( G, 'u', 'c')
    @assert G.adjacency ==  Dict{Any, Set}( 'u' => Set( ( 'v', 'c') ), 'v' => Set( 'u' ), 'c' => Set( 'u' ) )
    return nothing
end

function test_removeNode!( )

    d = Dict{Any, Set}( 'A' => Set( ( 'B', 'D' ) ), 
                        'B' => Set( ( 'A', 'D' ) ), 
                        'C' => Set( 'D' ), 
                        'D' => Set( ( 'A', 'B', 'C' ) ) )

    D = Dict{Any, Set}( 'B' => Set( 'D' ), 
                        'C' => Set( 'D' ), 
                        'D' => Set( ( 'B', 'C' ) ) )

    G = MyGraph( d )

    removeNode!( G, 'A' )

    @assert G.adjacency == D 

    return nothing
end

function test_removeEdge!( )

    d = Dict{Any, Set}( 'A' => Set( ( 'B', 'D' ) ), 
                        'B' => Set( ( 'A', 'D' ) ), 
                        'C' => Set( 'D' ), 
                        'D' => Set( ( 'A', 'B', 'C' ) ) )

    D = Dict{Any, Set}( 'A' => Set( 'D' ), 
                        'B' => Set( 'D' ), 
                        'C' => Set( 'D' ), 
                        'D' => Set( ( 'A', 'B', 'C' ) ) )

    G = MyGraph( d )
    removeEdge!( G, 'A', 'B' )

    @assert G.adjacency == D
    return nothing
end

function test_traverse( )

    d = Dict{Any, Set}( 'A' => Set( ( 'B', 'D' ) ), 
                        'B' => Set( ( 'A', 'D' ) ), 
                        'C' => Set( 'D' ), 
                        'D' => Set( ( 'A', 'B', 'C' ) ) )

    G = MyGraph( d )

    visited = traverse( G, 'A' )

    @assert visited == [ 'A', 'B', 'D', 'C' ]
    return nothing
end

function test_shortestPath( ) 

    d = Dict{Any, Set}( 'A' => Set( ( 'B', 'D' ) ), 
                        'B' => Set( ( 'A', 'D' ) ), 
                        'C' => Set( 'D' ), 
                        'D' => Set( ( 'A', 'B', 'C' ) ) )

    G = MyGraph( d )

    path = shortestPath( G, 'A', 'C' ) 

    D = Deque{Any}( )

    push!( D, 'A')
    push!( D, 'D' )
    push!( D, 'C' )

    @assert path == D

    return nothing
end

test_graph( )
test_println( )
test_addEdge!( )
test_addNode!( )
test_removeNode!( )
test_removeEdge!( )
test_traverse( )
test_shortestPath( )

printstyled( "All tests passed"; color=:green )