# markovChains.jl

module MarkovChains

using Distributions: Multinomial
using LinearAlgebra: norm
using Random: rand, seed!
export MarkovChain, transition, walk, path, steadyState, SentenceGenerator, babble

abstract type Chain end

struct MarkovChain <: Chain
    transitionMatrix::Matrix{<:Number}
    labels::Vector{String}
    mapping::Union{Dict{Int, Int},  Dict{String, Int}}
end

function MarkovChain( transitionMatrix )

    if !( length( axes( transitionMatrix )) == 2) || !( size(transitionMatrix, 1) == size(transitionMatrix, 2) )
        throw( ArgumentError( "The transition matrix needs to be square and 2 dimensional." ) )

    elseif !( sum( transitionMatrix; dims=2 ) ≈ ones( ( size(transitionMatrix,1), 1) ) )
        throw(ArgumentError("The transition matrix is not row stochastic"))
    end

    labels = collect( 1:size( transitionMatrix, 1) )
    mapping = Dict{Int, Int}( )
    for label in labels
        mapping[ label ] = label
    end

    return MarkovChain( float.( deepcopy( transitionMatrix ) ), labels, mapping )
end

function MarkovChain( transitionMatrix, labels )

    if !( length( axes( transitionMatrix )) == 2) || !( size(transitionMatrix, 1) == size(transitionMatrix, 2) )
        throw( ArgumentError( "The transition matrix needs to be square and 2 dimensional." ) )

    elseif !( sum( transitionMatrix; dims=2 ) ≈ ones( ( size(transitionMatrix,1), 1) ) )
        throw(ArgumentError("The transition matrix is not row stochastic"))
    end

    mapping = Dict{String, Int}( )
    for ( ii, label ) in enumerate( labels )
        mapping[ label ] = ii
    end

    return MarkovChain( float.( deepcopy( transitionMatrix ) ), labels, mapping )

end

function transition( MC::Chain, state::Union{String, Int} )

    draw = rand( Multinomial( 1, MC.transitionMatrix[ MC.mapping[ state ], : ] ) )

    index = argmax( draw )

    return MC.labels[ index ]

end

function walk( MC::Chain, start::Union{String, Int}, N::Int )

    if N <= 0
        throw( ArgumentError( "The walk length must be greater than 0." ) )
    end
    states = [ ]

    append!( states, [ start ] )

    for i=1:N-1
        newState = transition( MC, states[ end ] )
        append!( states, [ newState ] )
    end
    return states
end

function path( MC::Chain, source::Union{String,Int}, target::Union{String,Int} )


    states = [ ]
    append!( states, [ source ] )

    while states[ end ] != target 
        newState = transition( MC, states[ end ] )
        append!( states, [ newState] )
    end


    return states
end

function steadyState( MC::Chain; tol::Float64=1e-12, maxiter::Int=40 )

    xOld = rand( Float64, ( 1, size( MC.transitionMatrix, 1 ) ) )
    xOld ./= sum( xOld )

    for i=1:maxiter

        xNew = xOld * MC.transitionMatrix

        if norm( xNew - xOld ) < tol
            return xNew
            break
        end

        xOld = deepcopy( xNew )

    end

    throw(ArgumentError("A^k does not converge"))
end


struct SentenceGenerator <: Chain
    fileName::Union{String}
    transitionMatrix::Matrix{<:Number}
    labels::Union{Vector{String}, Nothing}
    mapping::Union{Dict{Int, Int},  Dict{String, Int}}
end


function SentenceGenerator( fileName::String )

    # read in text separating by lines
    io = open( fileName )
    lines = readlines( io )
    close( io )

    # get unique state labels
    stateLabels = String[ ]
    append!( stateLabels, [ "\$tart" ] )
    for line in lines
        words = split( strip( line) )
        for word in words 
            append!( stateLabels, [ string( word ) ] )
        end
    end
    append!( stateLabels, [ "\$top" ] )

    unique!( stateLabels )

    # initizlize the transition matrix
    transitionMatrix = zeros( Float64, ( length( stateLabels ), length( stateLabels ) ) )

    # the tranistion matrix is such that p( S_{t+1} = $top | S_{t} = $Stop ) = 1 
    transitionMatrix[ end, end ] = 1

    # algorithmically construct the transition matrix
    # The element transitionMatrix[ii, jj] is the probability that word jj follows word ii.

    for jj in eachindex( stateLabels[ 1:end-1 ] ) 
    # for jj in eachindex( stateLabels )
        for i in eachindex( stateLabels[ 1:end-1 ] )
            ii = i+1
        # for ii in eachindex( stateLabels ) 
            # nothing ever follows stop
            # if jj == length( stateLabels )
            #     break

            # nothing is ever followed by start
            # elseif ii == 1
            #     continue
            # find out what follows start
            if jj == 1
                currentPhrase = stateLabels[ ii ]
                # for each line find out which one of them has the current 
                # phrase as the first word
                for line in lines
                    if split( line )[ 1 ] == currentPhrase
                        transitionMatrix[ ii, jj ] += 1
                    end
                end
            # find out what is followed by stop
            elseif ii == length( stateLabels )
                currentPhrase = stateLabels[ jj ]

                # for each line find out which one of them
                # has the current phrase as the last word
                for line in lines
                    if split( line )[ end ] == currentPhrase
                        transitionMatrix[ ii, jj ] +=1
                    end
                end
            # found what words follow each other
            else
                currentPhrase = join( [ stateLabels[ jj ] stateLabels[ ii ] ], " " )
                # initizlize the curernt count of the prhase in all lines to zero
                countPhrase = 0

                for line in lines
                    countPhrase += count( currentPhrase, line )
                end

                if countPhrase != 0
                    transitionMatrix[ ii, jj ] = countPhrase
                end
            end
        end
    end

    # println( transitionMatrix )
    # make transitionMatrix row stochastic
    transitionMatrix ./= sum( transitionMatrix; dims=1 )

    # create state to label mapping
    mapping = Dict{String, Int}( )
    for ( ii, label ) in enumerate( stateLabels )
        mapping[ label ] = ii
    end

    return SentenceGenerator( fileName, Matrix( transitionMatrix' ), stateLabels, mapping )

end


function babble( SG::SentenceGenerator )


    randomBable = path( SG, "\$tart", "\$top" )
    replace!( randomBable, "\$tart"=>"")
    replace!( randomBable, "\$top"=>"")


    return join( randomBable, " " )
end



end

