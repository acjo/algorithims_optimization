#nearest_neighbor.jl

module NearestNeighbor

using LinearAlgebra, DataStructures

import Base.println, Base.insert!

export exhaustiveSearch, KDTNode, KDT, find, insert!, println, query

function exhaustiveSearch(X, z)
    """
    This function solves th enearest neighbor search problem with an exhaustive search.
    Parameters:
        X ((m,k)): a matrix of m k-dimensional points
        z ((k, )): a k-dimensional target point.
    Returns:
        min_vec ((k)) the row of Z that is nearest to z.
        min_dist (float) The Euclidean distance from the nearest neighbor to z.
    """


    z = vec(z)
    difference = X .- z
    distances = []
    for ii=1:size(x,1)
        currentDistance = norm(difference[ii, 1:end], 2)
        append!(distances, currentDistance)
    end

    minimumDistance = minimum(distances)

    mask = distances == minimumDistance

    nearestNeighbor = X[mask, 1:end]

    return nearestNeighbor, minimumDistance
end


mutable struct KDTNode
    """
    Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this nodes' right child.
        value ((k, )) : coordinate in k-dimensional space.
        pivot (int); the dimension of the value to make comparison on
        index (int): the index of the value within a data set. If any
    """

    value::Vector{<:Number}
    index::Union{Nothing, Int}
    left::Union{Nothing, KDTNode}
    right::Union{Nothing,KDTNode}
    pivot::Union{Nothing, Int}

    function KDTNode(value; index=nothing, left=nothing, right=nothing, pivot=nothing)
        """
        Constructor if only the value is provided. i.e. KDTNode(value)
        """

        return new(value,index,left,right,pivot)
    end

    function KDTNode(value, index; left=nothing, right=nothing, pivot=nothing)
        """
        Constructor if only the value and index are provided provided. i.e. KDTNode(value,index)
        """

        return new(value,index, left,right,pivot)
    end

    function KDTNode(value, index, left, right, pivot)
        """
        Constructor if everything is provided. i.e. KDTNode(value, index,left,right,pivot)
        """

        return new(value,index,left,right,pivot)
    end
end


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

    function KDT(;root=nothing, k=nothing)
        """
        Constructor if nothing is provided. I.e. KDT()
        """
        return new(root, k)
    end

    function KDT(root, k=nothing)
        """
        Constructor if only the root is provided. I.e KDT(root)
        """
        k = length(root.value)
        return new(root, k)
    end

    function KDT(root, k)
        """
        Constructor if everything is provided. i.e. KDT(root,k)
        """
    end
end


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
        elseif data[currentNode.pivot] < currentNode.value[currentNode.pivot]
            return _step(currentNode.left)
        # recursively search right
        else
            return _step(currentNode.right)
        end
    end
    return _step(tree.root)
end

function insert!(tree::KDT, data; index=nothing)
    """Insert a new node containing the specified data.

        Parameters:
            data: a k-dimensional point to insert into the tree.
        Raises:
            ArgumentError: if data does not have the same dimensions as other
            values in the tree.
            KeyError: If the data is already in the tree.
        """

    data = vec(data)
    newNode = KDTNode(data, index)

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
                # if the parents pivot is k-1 set child's pivot to 0
                if currentNode.pivot == tree.k -1
                    newNode.pivot = 0
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
                if currentNode.pivot == tree.k - 1
                    newNode.pivot = 0
                else
                    newNode.pivot = currentNode.pivot + 1
                end
            else
                _step(currentNode.right)
            end
        end
        return
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
                tree.root.pivot = 0
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

function println(tree::KDT)
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

    nodes, strs = Queue{KDTNode}(), Vector{String}(undef,0)
    enqueue!(nodes, tree.root)

    while length(nodes) > 0
        currentNode = dequeue!(nodes)
        currentString = string(currentNode.value) * "\tpivot = " * string(currentNode.pivot)
        append!(strs, currentNode)
        for childNode in [currentNode.left, currentNode.right]
            if !isequal(childNode, nothing)
                enqueue!(nodes, childNode)
            end
        end

    end

    return "KDT(k=" * string(tree.k) *")\n" * join(strs, "\n")
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
            nearest = currentNode
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

        return nearest, d
    end

    return KDSearch(tree.root, tree.root, norm(tree.root.value-z, 2))

end


mutable struct KNeighborsClassifier
    """
    A k-nearest neighbors classifier that uses our KDTRee to solve the
    nearest neighbor problem.
    """

    nNeighbors::Int
    tree::Union{KDT, Nothing}
    labels::Union{Vector{Union{String, Int}}, Nothing}

    function KNeighborsClassifier(nNeighbors;tree=nothing,labels=nothing)
        """
        Constructor for when only nNeighbors is given. i.e. KNeighborsClassifier(nNeigbors)
        """
        return new(nNeighbors,tree,labels)
    end

    function KNeighborsClassifier(nNeighbors,tree,labels)
        return new(nNeighbors,tree,labels)
    end
end

function fit!(classifier::KNeighborsClassifier, X, y)
    """
    creates the tree and labels attributes
    Paramaters:
        X: ((m,k) matrix) the training set
        y:((m,), vector) the training entries
    """

    KDTree = KDT()

    for ii=1:size(X, 1)
        currentRow = X[ii, 1:end]
        insert!(KDTree, currentRow; index=ii)
    end

    classifier.tree = KDTree
    classifier.labels = float.(vec(y))
end

function predict(classifier::KNeighborsClassifier, z)
    """
    returns the most common label
    Parameters:
    z (k Vector) elements whose label is to be predicted
    i.e. there are n data points of size k.
    """

    nodes, distances = query(classifier.tree, z)




end


end
