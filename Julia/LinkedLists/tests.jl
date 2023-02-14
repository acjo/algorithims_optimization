# tests.jl

module tests

include("linked_lists.jl")
using .LinkedLists

function testLL()

    testLL1 = LinkedList(nothing,nothing,0)
    LinkedLists.append!(testLL1, "A")
    LinkedLists.append!(testLL1, "B")
    LinkedLists.append!(testLL1, "C")
    LinkedLists.append!(testLL1, "A")
    LinkedLists.append!(testLL1, "D")
    LinkedLists.append!(testLL1, "D")
    LinkedLists.append!(testLL1, [1,2,3,])
    LinkedLists.append!(testLL1, π)
    LinkedLists.append!(testLL1, 3)
    LinkedLists.append!(testLL1, "D")
    LinkedLists.append!(testLL1, "D")

    corretStringRep = "[\"A\", \"B\", \"C\", \"A\", \"D\", \"D\", [1, 2, 3], π, 3, \"D\", \"D\"]"

    @assert corretStringRep == LinkedLists.repr(testLL1)

    remove!(testLL1, "A")
    testLL2 = LinkedList(nothing,nothing,0)
    LinkedLists.append!(testLL2, "B")
    LinkedLists.append!(testLL2, "C")
    LinkedLists.append!(testLL2, "A")
    LinkedLists.append!(testLL2, "D")
    LinkedLists.append!(testLL2, "D")
    LinkedLists.append!(testLL2, [1,2,3,])
    LinkedLists.append!(testLL2, π)
    LinkedLists.append!(testLL2, 3)
    LinkedLists.append!(testLL2, "D")
    LinkedLists.append!(testLL2, "D")

    @assert testLL1 == testLL2

    try
        println("here")
        remove!(testLL, "A")
    catch KeyError
        @assert true
    end

    return

end


function testD()

    return

end


testLL()

end
