# nearestNeighbor.jl

module NearestNeighbor

using LinearAlgebra, DataStructures, NPZ, NearestNeighbors
using StatsBase: mode
import Base.println, Base.insert!, Base.string

export exhaustiveSearch, KDTNode, KDT, find, insert!, println, string, query, KNeighborsClassifier, fit!, predict
function exhaustiveSearch(X, z)
    """
    This function solves th enearest neighbor search problem with an exhaustive search.
    Parameters:
        X ((m,k)): a matrix of m k-dimensional points
        z ((k,)): a k-dimensional target point.
    Returns:
        min_vec ((k)) the row of Z that is nearest to z.
        min_dist (float) The Euclidean distance from the nearest neighbor to z.
    """


    z = vec(z)
    # array broadcast to get the difference
    difference = X .- z'
    distances = [norm(difference[ii, 1:end], 2) for ii in axes(X, 1)]

    minimumDistance = minimum(distances)
    mask = distances .== minimumDistance
    nearestNeighbor = X[mask, 1:end]

    return nearestNeighbor, minimumDistance
end


mutable struct KDTNode
    """
    Node class for K-D Trees.

    Attributes:
        value ((k,)) : coordinate in k-dimensional space.
        index (int): the index of the value within a data set. If any
        pivot (int); the dimension of the value to make comparison on
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this nodes' right child.
    """

    value::Vector{<:Number}
    pivot::Union{Nothing, Int}
    left::Union{Nothing, KDTNode}
    right::Union{Nothing,KDTNode}

    function KDTNode(value, pivot, left, right)
        """
        Constructor if everything is provided. i.e. KDTNode(value, index,left,right,pivot)
        """

        return new(value, pivot, left, right)
    end
end
# constructor if only the value is provided
KDTNode(value) = KDTNode(value, nothing, nothing, nothing)
# constructor if only the value and pivot are provided
KDTNode(value, pivot) = KDTNode(value, pivot, nothing, nothing)


mutable struct KDT
    """
    A k-dimensional binary tree for solving the nearest neighbor problem.
    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """

    root::Union{KDTNode, Nothing}
    k::Union{Int,Nothing}

    # inner constructor
    function KDT(root, k)
        """
        Constructor if everything is provided. i.e. KDT(root,k)
        """
        return new(root, k)
    end
end
# outter constructors for default values
KDT() = KDT(nothing, nothing)
KDT(root) = KDT(root, length(root.value))


function find(tree::KDT, data)

    """
    return the node containing the data. If there is no such node
    in the tree, or if the tree is emtpy, throw a Key Error
    """

    function _step(currentNode)
        """
        Recursively step through the tree until finding the node
        containing the data. If there is no such node, throw a KeyError.
        """
        # data not found in the tree
        if isequal(currentNode, nothing)
            throw(KeyError(data))
        # node found!
        elseif data ≈ currentNode.value
            return currentNode
        # recurisvely search left
        elseif (currentNode.pivot !== nothing) && data[currentNode.pivot] < currentNode.value[currentNode.pivot]
            return _step(currentNode.left)
        # recursively search right
        else
            return _step(currentNode.right)
        end
    end
    node = _step(tree.root)
    return node
end

function insert!(tree::KDT, data; index=nothing)
    """Insert a new node containing the specified data.

        Parameters:
            tree: a KD tree
            data: a k-dimensional point to insert into the tree.
            index: the index of the node in the tree
        Raises:
            ArgumentError: if data does not have the same dimensions as other
            values in the tree.
            KeyError: If the data is already in the tree.
        """

    data = vec(data)
    newNode = KDTNode(data)

    if !isequal(tree.root,nothing)
        if tree.k ≠ size(data,1)
            throw(ArgumentError("The input data is not of the same dimension as other values in the tree."))
        end
    end

    function _step(currentNode)
        # we need to place the new node on the left if this is true
        if data[currentNode.pivot] < currentNode.value[currentNode.pivot]
            # if there is no left node, insert the data here
            if isequal(currentNode.left, nothing)
                # pont the parent to it
                currentNode.left = newNode
                # if the parents pivot is k set child's pivot to 1
                if currentNode.pivot == tree.k
                    newNode.pivot = 1
                # set the child's pivot to k+1
                else
                    newNode.pivot = currentNode.pivot + 1
                end
           # otherwise recursive call on the left node
            else
                _step(currentNode.left)
            end
            # if the data at the pivot is greater than orequal to the node value do the same thing
        else
            if isequal(currentNode.right, nothing)
                currentNode.right = newNode
                if currentNode.pivot == tree.k 
                    newNode.pivot = 1
                else
                    newNode.pivot = currentNode.pivot + 1
                end
            else
                _step(currentNode.right)
            end
        end
        return nothing
    end

    # try to find the data
    try
        find(tree, data)
    catch e
        # if the data isn't in the tree
        if isa(e, KeyError)
            # if the tree is empty set the root node, k, and the pivot
            if isequal(tree.root, nothing)
                tree.root = newNode
                tree.root.pivot = 1
                tree.k = size(data,1)
            # otherwise recurisvely step through starting with the root node
            else
                _step(tree.root)
            end
        # rethrow error if we caught the wrong one
        else
            throw(e)
        end
    # if no error was caught, that means the data is aready in the tree.
    else
        throw(ArgumenError("Data already contained in tree, no duplicates allowed!"))
    end

    return

end

function println(io::IO, tree::KDT)
    #=
    """String representation: a hierarchical list of nodes and their axes.
        Example:                            KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0
    """
    =#

    if isequal(tree.root,nothing)
        println("Empty KDT")
    end

    nodes, strs = Queue{KDTNode}(), String[]
    enqueue!(nodes, tree.root)

    while length(nodes) > 0
        currentNode = dequeue!(nodes)
        currentString = string(currentNode.value) * "\tpivot = " * string(currentNode.pivot)
        push!(strs, currentString)
        for childNode in [currentNode.left, currentNode.right]
            if !isequal(childNode, nothing)
                enqueue!(nodes, childNode)
            end
        end

    end

    str = "KDT(k=" * string(tree.k) *")\n" * join(strs, "\n")
    println(str)
    return nothing
end

function string(tree::KDT)
    #=
    """String representation: a hierarchical list of nodes and their axes.
        Example:                            KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0
    """
    =#

    if isequal(tree.root,nothing)
        return "Empty KDT"
    end

    nodes, strs = Queue{KDTNode}(), String[]
    enqueue!(nodes, tree.root)

    while length(nodes) > 0
        currentNode = dequeue!(nodes)
        currentString = string(currentNode.value) * "\tpivot = " * string(currentNode.pivot)
        push!(strs, currentString)
        for childNode in [currentNode.left, currentNode.right]
            if !isequal(childNode, nothing)
                enqueue!(nodes, childNode)
            end
        end

    end

    str = "KDT(k=" * string(tree.k) *")\n" * join(strs, "\n")
    return str
end

function query(tree::KDT, z)
    """This function finds the value in the tree that is nearest to z.

        Parameters:
            z : a k-dimensional target point.

        Returns:
            the node in the tree that is nearest to z.
            The Euclidean distance from the nearest neighbor to z.
    """

    function KDSearch(currentNode, nearestNode, d)
        """
        Recursive function used to find the nearest node
        """
        if isequal(currentNode, nothing)
            return nearestNode, d
        end

        x = currentNode.value
        i = currentNode.pivot

        currentDistance = norm(x-z,2)

        if currentDistance < d # check if current is closer to z than the nearest
            nearestNode = currentNode
            d = currentDistance
        end

        # search to the left
        if z[i] < x[i]
            nearestNode, d = KDSearch(currentNode.left, nearestNode, d)
            # search to the right if needed
            if z[i] + d >= x[i]
                nearestNode, d = KDSearch(currentNode.right, nearestNode, d)
            end
        # search to the right
        else
            nearestNode, d = KDSearch(currentNode.right, nearestNode, d)
            # search to the left if needed
            if z[i] - d <= x[i]
                nearestNode, d = KDSearch(currentNode.left, nearestNode, d)
            end
        end

        return nearestNode, d
    end

    node, distance = KDSearch(tree.root, tree.root, norm(tree.root.value-z, 2))

    return node.value, distance

end


mutable struct KNeighborsClassifier
    """
    A k-nearest neighbors classifier that uses our KDTRee to solve the
    nearest neighbor problem.
    """

    nNeighbors::Int
    tree::Union{KDTree, Nothing}
    labels::Union{Vector{Union{String, Int, Float64}}, Nothing}

    function KNeighborsClassifier(nNeighbors,tree,labels)
        return new(nNeighbors,tree,labels)
    end
end

KNeighborsClassifier(nNeighbors) = KNeighborsClassifier(nNeighbors, nothing, nothing) 

function fit!(classifier::KNeighborsClassifier, X, y)
    """
    creates the tree and labels attributes
    Paramaters:
        X: ((k,m) matrix) the training set
        y:((m,), vector) the training entries
    """

    if !isequal(classifier.tree, nothing)
        @warn "KD trees are static, rebuilding entire tree with new data"

        if size(X, 1) != size(KDT.data[1], 1)
            throw(ArgumentError("The number of rows represents the dimension of the search space, \"X\" does not have the correct number of dimensions.")) 
        end
        oldX = reduce(hcat, KDT.data)
        combinedX = hcat(oldX, X)

        for x in classifier.tree.data
            combinedX = [combinedX, x]
        end

        kdt = KDTree(combinedX)

    else
        kdt = KDTree(X)
    end
    classifier.tree = kdt
    classifier.labels = float.(vec(y))
end

function predict(classifier::KNeighborsClassifier, z)
    """
    returns the most common label
    Parameters:
    z (k Vector) elements whose label is to be predicted
    i.e. there are n data points of size k.
    """

    idxs, dists = knn(classifier.tree, z, classifier.nNeighbors)

    listOfLabels = [classifier.labels[idx] for idx in idxs]

    return mode(listOfLabels)

end

function mnist(nNeighbors; fileName ="mnist_subset.npz")

    data = npzread(fileName)
    xTrain = float.(data["X_train"])
    xTrain = transpose(xTrain)
    yTrain = Int.(data["y_train"])
    xTest = float.(data["X_test"])
    xTest = transpose(xTest)
    yTest = Int.(data["y_test"])

    classifier = KNeighborsClassifier(nNeighbors)
    fit!(classifier, xTrain, yTrain)

    classificationMatch = 0
    for ii in axes(xTest, 2)
        predictedLabel = predict(classifier, xTest[:, ii])
        if predictedLabel == yTest[ii] 
            classificationMatch += 1
        end
    end

    accuracy = classificationMatch / size(xTest, 2)
    return accuracy

end

end
