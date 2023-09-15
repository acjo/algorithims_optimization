include("nearestNeighbor.jl")

using .NearestNeighbor, NearestNeighbors, Distances, NPZ

function test_exhaustiveSearch(m::Int, k::Int)

    X = npzread("testX.npy")
    z = npzread("testz.npy")

    
    X = rand(Float64, (m,k))
    z = rand(Float64, k)

    minvec, mindist = exhaustiveSearch(X, z)

    kdt = KDTree(X')

    idx, dist = nn(kdt, z)

    @assert mindist ≈ dist
    @assert minvec ≈ X[idx, 1:end]' 

    return nothing
end

function test_KDTNode()

    node = KDTNode([1, 2, 3])

    @assert node.value == [1,2,3]
    @assert node.pivot === nothing
    @assert node.left === nothing 
    @assert node.right === nothing


    node = KDTNode([1, 2, 3], 1)
    @assert node.value == [1,2,3]
    @assert node.pivot == 1
    @assert node.left === nothing
    @assert node.right === nothing


    return nothing

end

function test_KDTree()

    kdt = KDT()

    @assert kdt.root === nothing
    @assert kdt.k === nothing

    node = KDTNode([1,2,3], 0)
    kdt = KDT(node)
    @assert kdt.k == 3
    @assert kdt.root.value == [1,2,3]

    return nothing
end

function test_find()

    kdt = KDT()
    try
        find(kdt, [1,2,3])
    catch e
        if isa(e, KeyError)
            @assert true
        else
            throw(e)
        end
    end

    node = KDTNode([1,2,3], 1)
    kdt = KDT(node, 3)
    foundNode = find(kdt, [1,2,3])
    @assert foundNode === node

    try
        find(kdt, [3,4,5])
    catch e
        if isa(e, KeyError)
            @assert true
        else
            throw(e)
        end
    end

    return nothing

end

function test_insert()

    function _test(dataset, fileName)

        kdt = KDT()

        for entry in dataset
            insert!(kdt, entry)
        end

        myStrRep = string(kdt)
        io = open(fileName, "r")
        lines = readlines(io)
        close(io)
        cStrRep = join(lines, "\n")

        @assert myStrRep == cStrRep

        return nothing
    end

    check1 = [[2, 3], 
               [1, 4]]

    check2 = [[2, 3, 4], 
               [5, 6, 7], 
               [7, 1, 9], 
               [3, 4, 8]]

    check3 = [[5, 5], 
               [2, 2], 
               [8, 8], 
               [3, 3], 
               [4, 4], 
               [1, 1], 
               [6, 6], 
               [7, 7], 
               [9,9]]

    check4 = [[5, 5], 
               [2, 4], 
               [8, 3], 
               [3, 2], 
               [4, 6], 
               [1, 7], 
               [6, 8], 
               [7, 9], 
               [9, 1]]

    check5 = [[3, 1, 4], 
               [1, 2, 7], 
               [4, 3, 5], 
               [2, 0, 3], 
               [2, 4, 5], 
               [6, 1, 4], 
               [1, 4, 4], 
               [0, 5, 7], 
               [5, 2, 5]]

    for (data, fileName) in zip([check1, check2, check3, check4, check5], 
                                 ["check1.txt", "check2.txt", "check3.txt", "check4.txt", "check5.txt"])
        _test(data, fileName)
    end

    return nothing
end



function test_query()

    function neighbor(m::Int, k::Int)
        # get random pints
        X = rand(Float64, (m, k)) 
        y = rand(Float64, k)

        # build trees
        myKDT = KDT()
        for ii in axes(X, 1)
            insert!(myKDT, X[ii, :])
        end
        kdt = KDTree(X')

        # query for the distances
        point, distance = query(myKDT, y)
        idx, dist = nn(kdt, y)

        closest = X[idx, :]


        @assert distance ≈ dist
        @assert closest ≈ point

        return nothing

    end

    dims = [(10, 10), 
             (100, 10), 
             (10, 100), 
             (100, 100), 
             (100, 100)]


    for (m, k) in dims
        neighbor(m, k)
    end
    return nothing
end


function test_knnclassifier()


    function _test(m,k, nNeighbors)

        # generate random data
        data = rand(Float64, (k, m))
        # generate random target
        target = rand(Float64, k)
        # do binary classification
        labels = rand((1:2), m)

        # classify 
        classifier = KNeighborsClassifier(nNeighbors)

        fit!(classifier, data, labels)

        stu = predict(classifier, target)

        return nothing
    end

    for m in [10, 20, 50, 100]
        for (k, nNeighbors) in [(5, 3), (10, 5)]
            _test(m, k, nNeighbors)
        end
    end

    return nothing

end

function test_mnist()


    acc = NearestNeighbor.mnist(1)

    @assert acc > 0.88

    return

end


test_exhaustiveSearch(100, 10)

test_KDTNode()

test_KDTree()

test_find()

test_insert()

test_query() 

test_knnclassifier()

test_mnist()


printstyled("All tests pass\n", color=:blue)