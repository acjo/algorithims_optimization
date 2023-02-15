# tests.jl

module tests

include("linked_lists.jl")
using .LinkedLists

function testLL()

    LLN1 = LinkedListNode("A",nothing,nothing)
    testLL = LinkedList(LLN1,LLN1,"A")



    # test constructor & append function
    testLL1 = LinkedList(nothing,nothing,0)
    # test append function (first block)
    append!(testLL1, "A")
    @assert testLL1 == testLL

    remove!(testLL1, "A")
    testLL = LinkedList(nothing,nothing,0)
    @assert testLL1 == testLL

    # add a bunch more nodes and test the append function (second block)
    append!(testLL1, "A")
    append!(testLL1, "B")
    append!(testLL1, "C")
    append!(testLL1, "A")
    append!(testLL1, "D")
    append!(testLL1, "D")
    append!(testLL1, [1,2,3,])
    append!(testLL1, π)
    append!(testLL1, 3)
    append!(testLL1, "D")
    append!(testLL1, "D")


    append!(testLL, "A")
    append!(testLL, "B")
    append!(testLL, "C")
    append!(testLL, "A")
    append!(testLL, "D")
    append!(testLL, "D")
    append!(testLL, [1,2,3,])
    append!(testLL, π)
    append!(testLL, 3)
    append!(testLL, "D")
    append!(testLL, "D")

    @assert testLL1 == testLL


    # test the find function
    testLL = LinkedList(nothing,nothing,0)

    # first block
    try
        find(testLL, 3)
    catch e
        if isa(e,KeyError)
            @assert true
        else
            throw(e)
        end
    end


    append!(testLL, 3)
    node = find(testLL, 3)
    @assert node.data == 3
    @assert isequal(node.next, nothing)
    @assert isequal(node.prev, nothing)

    append!(testLL, 3)
    try
        node = find(testLL, 4)
    catch e
        if isa(e, KeyError)
            @assert true
        else
            throw(e)
        end
    end


    # correct string representation test
    correctStringRep = "[\"A\", \"B\", \"C\", \"A\", \"D\", \"D\", [1, 2, 3], π, 3, \"D\", \"D\"]"
    @assert correctStringRep == repr(testLL1)

    # test removal function
    remove!(testLL1, "A")
    testLL2 = LinkedList(nothing,nothing,0)
    append!(testLL2, "B")
    append!(testLL2, "C")
    append!(testLL2, "A")
    append!(testLL2, "D")
    append!(testLL2, "D")
    append!(testLL2, [1,2,3,])
    append!(testLL2, π)
    append!(testLL2, 3)
    append!(testLL2, "D")
    append!(testLL2, "D")

    @assert testLL1 == testLL2

    try
        remove!(testLL, "A")
    catch e
        if isa(e,KeyError)
            @assert true
        else
            throw(e)
        end
    end

    testLL1 = LinkedList(nothing,nothing,0)
    testLL2 = LinkedList(nothing,nothing,0)

    append!(testLL1, "A")
    insert!(testLL2, 1, "A")

    @assert testLL1 == testLL2


    return true
end


function testD()

    return true

end


function allTests()


    @assert testLL()
    @assert testD()

    println("All tests passed!")

end

allTests()


end
