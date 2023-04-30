include("nearestNeighbor.jl")

using .NearestNeighbor, NearestNeighbors, Distances, NPZ

function test_exhaustiveSearch( m::Int,k::Int )

    X = npzread("testX.npy")
    z = npzread("testz.npy")

    
    X1 = npzread( "testX.npy ")
    X = rand( Float64, ( m,k ) )
    z = rand( Float64, k )

    minvec, mindist = exhaustiveSearch( X, z )

    kdt = KDTree( X' )

    idx, dist = nn( kdt, z )


    @assert mindist ≈ dist
    @assert minvec ≈ X[ idx, 1:end ]' 

    return nothing
end

function test_KDTNode( )

    node = KDTNode( [ 1, 2, 3 ] )

    @assert node.value == [ 1,2,3 ]
    @assert node.index === nothing
    @assert node.pivot === nothing
    @assert node.left === nothing 
    @assert node.right === nothing


    node = KDTNode( [1, 2, 3], 1 )
    @assert node.value == [1,2,3]
    @assert node.index == 1
    @assert node.pivot === nothing
    @assert node.left === nothing
    @assert node.right === nothing

    node = KDTNode( [1, 2, 3], 1, 1 )
    @assert node.value == [1,2,3]
    @assert node.index == 1
    @assert node.pivot === 1
    @assert node.left === nothing
    @assert node.right === nothing


    return nothing

end

function test_KDTree( )

    kdt = KDT( )

    @assert kdt.root === nothing
    @assert kdt.k === nothing

    node = KDTNode( [ 1,2,3 ], 0, 0 )
    kdt = KDT( node )
    @assert kdt.k == 3
    @assert kdt.root.value == [ 1,2,3 ]

    return nothing
end

function test_find( )

    kdt = KDT( )
    try
        find( kdt, [ 1,2,3 ] )
    catch e
        if isa( e, KeyError )
            @assert true
        else
            throw( e )
        end
    end

    node = KDTNode( [ 1,2,3 ], 1 )
    kdt = KDT( node, 3 )

    foundNode = find( kdt, [ 1,2,3 ] )

    @assert foundNode === node

    try
        find( kdt, [ 3,4,5 ] )
    catch e
        if isa( e, KeyError )
            @assert true
        else
            throw( e )
        end
    end

    return nothing

end


test_exhaustiveSearch( 100, 10 )

test_KDTNode( )

test_KDTree( )

test_find( )