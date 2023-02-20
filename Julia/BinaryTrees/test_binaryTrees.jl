# test_binaryTrees.jl

include("binaryTrees.jl")
using .BinaryTrees, Random, Plots


function testSLL()
end

function testBST()
    root = BSTNode(5);

    tree = BST(root);

    insert!(tree, 2);
    insert!(tree, 7);
    insert!(tree, 1);
    insert!(tree, 6);
    insert!(tree, 8);


    draw(tree);
end

function testAVL()
end


function prob4()

    inputFile = "english.txt"

    io = open(inputFile, "r")

    input = read(io, String)

    close(io)

    lines = split(input, "\n")[1:end-1]

    numElements = [2^n for n=3:14]


    # singly linked list times
    sllLoadTimes = []
    sllFindTimes = []

    # bst times
    bstLoadTimes = []
    bstFindTimes = []

    # avl times
    avlLoadTimes = []
    avlFindTimes = []

    for n in numElements
        println(n)

        # random indices of the vector containing the strings from the input file
        randomIndexes = randperm(MersenneTwister(1234), length(lines))[1:n]

        # random elments
        randomElements = lines[randomIndexes]

        # random indices of the vector containing n random elemnts from the input file
        randomIndexes = randperm(MersenneTwister(1234), n)[1:5]
        subset = randomElements[randomIndexes]

        # create data structures
        sll = SinglyLinkedList()
        bst = BST()
        avl = AVL()

        ###################
        # timing creation times
        ###################
        # SLL


        totalTime = 0
        for element in randomElements
            t = @timed append!(sll, string(element));
            totalTime += t.time
        end

        append!(sllLoadTimes, totalTime)

        # BST
        totalTime = 0
        for element in randomElements
            t = @timed insert!(bst, string(element))
            totalTime += t.time
        end

        append!(bstLoadTimes, totalTime)

        # AVL
        totalTime = 0
        for element in randomElements
            t = @timed insert!(avl, string(element))
            totalTime += t.time
        end
        append!(avlLoadTimes, totalTime)


        ###################
        # timing finding
        ###################
        # SLL
        totalTime = 0
        for element in subset
            t = @timed iterativeFind(sll, string(element))
            totalTime += t.time
        end
        append!(sllFindTimes, totalTime)

        # timing finding
        # BST
        totalTime = 0
        for element in subset
            t = @timed find(bst, string(element))
            totalTime += t.time
        end
        append!(bstFindTimes, totalTime)

        # timing finding
        # AVL
        totalTime = 0
        for element in subset
            t = @timed find(avl, string(element))
            totalTime += t.time
        end
        append!(avlFindTimes, totalTime)

    end
    # take absolute value of arrays
    #=
    sllLoadTimes = abs.(sllLoadTimes)
    bstLoadTimes = abs.(bstLoadTimes)
    avlLoadTimes = abs.(avlLoadTimes)

    sllFindTimes = abs.(sllFindTimes)
    bstFindTimes = abs.(bstFindTimes)
    avlFindTimes = abs.(avlFindTimes)
    =#

    combinedLoadTimes = [sllLoadTimes; bstLoadTimes; avlLoadTimes]
    combinedFindTimes = [sllFindTimes; bstFindTimes; avlFindTimes]

    p1 = plot(numElements, sllLoadTimes; label="SLL", linewidth=3, xscale=:log2, yscale=:log2, minorgrid=true)
    plot!(numElements, bstLoadTimes; label="BST", linewidth=3, xscale=:log2, yscale=:log2, minorgrid=true)
    plot!(numElements, avlLoadTimes; label="AVL", linewidth=3, xscale=:log2, yscale=:log2, minorgrid=true)
    title!("Node Insertion Times")
    # p1 = plot(numElements,[sllLoadTimes, bstLoadTimes, avlLoadTimes]; label=["SLL", "BST", "AVL"], title="Load Times", linewidth=3, xscale=:log2, yscale=:log2, minorgrid=true)
    xlims!(minimum(numElements), maximum(numElements))
    ylims!(minimum(combinedLoadTimes), maximum(combinedLoadTimes))
    xlabel!("Number of Elements in Data Structure")
    ylabel!("Time")
    display(p1)
    println("Press 'Enter' to continue...")
    readline()

    p2 = plot(numElements, sllFindTimes; label="SLL", linewidth=3, xscale=:log2, yscale=:log2, legend=true,minorgrid=true)
    plot!(numElements, bstFindTimes; label="BST", linewidth=3, xscale=:log2, yscale=:log2, legend=true,minorgrid=true)
    plot!(numElements, avlFindTimes; label="AVL", linewidth=3, xscale=:log2, yscale=:log2, legend=true,minorgrid=true)
    title!("Node Search Times")
    xlims!(minimum(numElements), maximum(numElements))
    ylims!(minimum(combinedFindTimes), maximum(combinedFindTimes))
    xlabel!("Number of Elements in Data Structure")
    ylabel!("Time")
    display(p2)
    println("Press 'Enter' to continue...")
    readline()

    plt = plot(p1,p2,layout=(2,1))
    display(plt)
    println("Press 'Enter' to continue...")
    readline()
    #=
    # p2 = plot(numElements,[sllFindTimes, bstFindTimes, avlFindTimes]; label=["SLL", "BST", "AVL"], title="Find Times", linewidth=3, xscale=:log2, yscale=:log2, legend=true, minorgrid=true)
    readline()
    =#
    #=
    plt = plot(p1, p2, layout=(2,1))
    display(plt)
    =#

end

prob4()
