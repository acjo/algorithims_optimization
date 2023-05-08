include("markovChains.jl")

using .MarkovChains
using Random: seed!
using NPZ


function test_sentenceGenerator( )

    SG = SentenceGenerator( "Julia\\MarkovChains\\yoda.txt" )
    TM = npzread( "Julia\\MarkovChains\\TM.npy" )
    @assert SG.transitionMatrix ≈ TM'

    return nothing
end



function test_babble( )
    seed!( 3 ) 

    SG = SentenceGenerator( "Julia\\MarkovChains\\yoda.txt" )
    println( babble( SG ) )

    input = readline( )

    @assert input == "true"


    return nothing
end


test_sentenceGenerator( )
printstyled( "Test 1 passed.\n"; color=:green )
test_babble( )
printstyled( "Test 2 passed.\n"; color=:green )
printstyled( "All tests passed.\n"; color=:green )



