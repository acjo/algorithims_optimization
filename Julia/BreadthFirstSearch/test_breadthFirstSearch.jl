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

    representation = repr( G )

    @assert representation == "MyGraph\nDict{Any, Set}(1 => Set([2]))"

    return nothing
end

function test_addNode!( )

    G = MyGraph( )

    addNode!(G, 1)
    @assert G.adjacency == Dict{Any, Set}( 1 => Set( ) )
    d = Dict( "a" => Set( [ "b" ] ) )
    G = MyGraph( d )
    addNode!( G, "u" )
    @assert G.adjacency ==  Dict{Any, Set}( "a" => Set( [ "b" ]), "u" => Set( ) )

    return nothing

end

function test_addEdge!( )

    G = MyGraph( )
    addEdge!( G, "u", "v")

    @assert G.adjacency ==  Dict{Any, Set}( "u" => Set( [ "v" ] ), "v" => Set( [ "u" ] ) )
    addEdge!( G, "u", "c")
    @assert G.adjacency ==  Dict{Any, Set}( "u" => Set( ( "v", "c") ), "v" => Set( [ "u" ] ), "c" => Set( [ "u" ] ) )
    return nothing
end

function test_removeNode!( )

    d = Dict{Any, Set}( "A" => Set( ( "B", "D" ) ), 
                        "B" => Set( ( "A", "D" ) ), 
                        "C" => Set( [ "D" ] ), 
                        "D" => Set( ( "A", "B", "C" ) ) )

    D = Dict{Any, Set}( "B" => Set( [ "D" ] ), 
                        "C" => Set( [ "D" ] ), 
                        "D" => Set( ( "B", "C" ) ) )

    G = MyGraph( d )

    removeNode!( G, "A" )

    @assert G.adjacency == D 

    return nothing
end

function test_removeEdge!( )

    d = Dict{Any, Set}( "A" => Set( ( "B", "D" ) ), 
                        "B" => Set( ( "A", "D" ) ), 
                        "C" => Set( [ "D" ] ), 
                        "D" => Set( ( "A", "B", "C" ) ) )

    D = Dict{Any, Set}( "A" => Set( [ "D" ] ), 
                        "B" => Set( [ "D" ] ), 
                        "C" => Set( [ "D" ] ), 
                        "D" => Set( ( "A", "B", "C" ) ) )

    G = MyGraph( d )
    removeEdge!( G, "A", "B" )


    @assert G.adjacency == D
    return nothing
end

function test_traverse( )

    d = Dict{Any, Set}( "A" => Set( ( "B", "D" ) ), 
                        "B" => Set( ( "A", "D" ) ), 
                        "C" => Set( [ "D" ] ), 
                        "D" => Set( ( "A", "B", "C" ) ) )

    G = MyGraph( d )

    visited = traverse( G, "A" )
    @assert visited == [ "A", "B", "D", "C" ]
    return nothing
end

function test_shortestPath( ) 

    d = Dict{Any, Set}( "A" => Set( ( "B", "D" ) ), 
                        "B" => Set( ( "A", "D" ) ), 
                        "C" => Set( "D" ), 
                        "D" => Set( ( "A", "B", "C" ) ) )

    G = MyGraph( d )

    path = shortestPath( G, "A", "C" ) 

    D = Deque{Any}( )

    push!( D, "A" )
    push!( D, "D" )
    push!( D, "C" )

    @assert path == D

    return nothing
end

function test_movieGraph( )

    t = @timed MV = MovieGraph( "Julia\\BreadthFirstSearch\\movie_data.txt" )

    @assert t.time < 20 

    @assert MV.fileName == "Julia\\BreadthFirstSearch\\movie_data.txt"
    @assert length( MV.actorNames ) == 930_717
    @assert length( MV.movieTitles ) == 137_018
    @assert length( MV.indexToName ) == 930_717 + 137_018 
    @assert length( MV.nameToIndex ) == 930_717 + 137_018 
    @assert MV.network.ne == 3_186_529

    return nothing
end

function test_pathToActor()

    MV = MovieGraph( "Julia\\BreadthFirstSearch\\movie_data.txt" )

    answers = [ (["Samuel L. Jackson", "Pulp Fiction (1994)", "Frank Whaley", "JFK (1991)", "Kevin Bacon"], 4),
                (["Ewan McGregor", "Star Wars: The Force Awakens (2015)", "Greg Grunberg", "Hollow Man (2000)", "Kevin Bacon"], 4),
                (["Jennifer Lawrence", "X-Men: First Class (2011)", "Kevin Bacon"], 2),
                (["Mark Hamill", "Star Wars: Episode V - The Empire Strikes Back (1980)", "John Ratzenberger", "She's Having a Baby (1988)", "Kevin Bacon"], 4) ]

    for ( ( actor1, actor2 ), (correctPath, correctL) ) in zip( [ ( "Samuel L. Jackson", "Kevin Bacon" ), ( "Ewan McGregor", "Kevin Bacon"), ("Jennifer Lawrence", "Kevin Bacon"), ( "Mark Hamill", "Kevin Bacon" ) ], answers )
        path, L = pathToActor( MV, actor1, actor2 )

        @assert L == correctL
        @assert path == correctPath

    end

    return nothing
end

function test_averageNumber( )

    MV = MovieGraph( "Julia\\BreadthFirstSearch\\movie_data_small.txt" )
    @assert averageNumber( MV, "Kevin Bacon" ) ≈ 2.637441287054702

    return nothing
end

test_graph( )
printstyled( "Test 1 passed\n"; color=:green )
test_println( )
printstyled( "Test 2 passed\n"; color=:green )
test_addNode!( )
printstyled( "Test 3 passed\n"; color=:green )
test_addEdge!( )
printstyled( "Test 4 passed\n"; color=:green )
test_removeNode!( )
printstyled( "Test 5 passed\n"; color=:green )
test_removeEdge!( )
printstyled( "Test 6 passed\n"; color=:green )
test_traverse( )
printstyled( "Test 7 passed\n"; color=:green )
test_shortestPath( )
printstyled( "Test 8 passed\n"; color=:green )
test_movieGraph( ) 
printstyled( "Test 9 passed\n"; color=:green )
test_pathToActor( )
printstyled( "Test 10 passed\n"; color=:green )
test_averageNumber( )
printstyled( "Test 11 passed\n"; color=:green )

printstyled( "All tests passed"; color=:green )