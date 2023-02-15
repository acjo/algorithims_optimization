# binary_trees.jl

module BinaryTrees

import Base.append!

using Plots
using Graphs
using GraphPlot

mutable struct SinglyLinkedListNode
    data::Union{Nothing, <:Number, String, Vector{<:Any}}
    next::Union{Nothing,SinglyLinkedListNode}
end

mutable struct SinglyLinkedList
    head::Union{Nothing, SinglyLinkedListNode}
    tail::Union{Nothing, SinglyLinkedListNode}
end

function append!(SLL::SinglyLinkedList, data)

    newNode = SinglyLinkedListNode(data, nothing)

    if isequal(SLL.head,nothing)
        SLL.head=newNode
        SLL.tail=newNode
    else
        SLL.tail.next = newNode
        SLL.tail = newNode
    end
end

function iterativeFind(SLL::SinglyLinkedList, data)

    currentNode = SLL.head
    while !isequal(currentNode, nothing)
        if currentNode.data == data
            return currentNode
        end
        currentNode = currentNode.next
    end
    throw(KeyError(data))
end

function recursiveFind(SLL::SinglyLinkedList, data)

    function isNode(currentNode::SinglyLinkedListNode)

        if isequal(currentNode,nothing)
            throw(KeyError(data))
        elseif currentNode.data == data
            return currentNode
        elseif isequal(currentNode.next,nothing)
            throw(KeyError(data))
        else
            return isNode(currentNode.next)
        end
    end

    return isNode(SLL.head)
end

mutable struct BSTNode
    data::Union{<:Number, String, Vector{<:Any}}
    prev::Union{Nothing, BSTNode}
    left::Union{Nothing, BSTNode}
    right::Union{Nothing, BSTNode}

    function BSTNode(data; prev=nothing,left=nothing,right=nothing)
        return new(data,prev,left,right)
    end

    function BSTNode(data,prev,left,right)
        return new(data,prev,left,right)
    end
end

mutable struct BST
    root::Union{Nothing, BSTNode}
    nodeCount::Int64

    function BST(root; nodeCount=0)
        if !isequal(root,nothing)
            nodeCount += 1
        end
        return new(root,nodeCount)
    end

    function BST(root,nodeCount)
        if !isequal(root,nothing)
            nodeCount = 1
        else
            nodeCount = 0
        end

        return new(root,nodeCount)
    end
end

function find(B::BST, data)
    """ Return the node containing the data. If there is no such node in the tree,
        including if the tree is empty, raise a KeyError.
    """

    # define recursive function to traverse the tree.
    function _step(currentNode)
        """ Recursively step through the tree until the node containing
            the data is found. If there is no such node, throw a KeyError.
        """
        if isequal(currentNode,nothing)
            throw(KeyError(data))
        elseif currentNode.data == data
            return currentNode
        elseif currentNode.data > data
            return _step(currentNode.left)
        else
            return _step(currentNode.right)
        end
    end
    return _step(B.root)
end


function insert!(B::BST,data)
    """ Insert a new node containing the specified data.

        throws:
            KeyError: if the data is already in the tree.
    """

    function _step(currentNode)
        if data < currentNode.data # if the data is less than the node value we insert to the left
            if isequal(currentNode.left,nothing) # if there is no left node we insert the data here
                newNode = BSTNode(data)
                currentNode.left = newNode
                newNode.prev = currentNode
            else #otherwise recursive call to this function
                _step(currentNode.left)
            end
        else # if the data is greater than the node value do the same thing above except on the right
             # notice no duplicate values are allowed in the BST.
            if isequal(currentNode.right,nothing)
                newNode = BSTNode(data)
                currentNode.right = newNode
                newNode.prev = currentNode
            else
                _step(currentNode.right)
            end
        end
    end

    try # try and find the data
        find(B,data)
    catch e
        if isa(e, KeyError) # if the data is not found insert our new node
            if isequal(B, nothing) # if there is no root, the BST is empty, make the newnode the root
                B.root = BSTNode(data)
            else # otherwise call the step function to insert the node
                _step(B.root)
            end
        else # if the wrong error is caught, throw the error again.
            throw(e)
        end
    else
        # if no error is caught at all that means the data is already contained in the true.
        throw(ArgumentError("Data already contained in tree, no duplicates allowed."))
    end
end

function remove!(B::BST, data)
    """ This function removes the node containing the specified data.

        throws:
            KeyError: If there is no node containing the data, including if
            the tree is empty.
    """

    if isequal(B.root,nothing) # if the BST is empty raise a value error
        throw(KeyError(data))
    end

    # this is the node to remove; find() will raise a KeyError if
    # there is no node in the BST containing the data.
    nodeToRemove = find(B,data)

    if isequal(nodeToRemove.left,nothing) && isequal(nodeToRemove.right,nothing) # if the nde to remove is a leaf node.
        if isequal(nodeToRemove.prev, nothing) # if it is the root node
            B.root = nothing
        elseif nodeToRemove.prev.left == nodeToRemove # if the node to remove is a left child of the parent node, set the left pointer to nothing
            nodeToRemove.prev.left=nothing
        else # otherwise it is a right child so set the parent's right child to nothing.
            nodeToRemove.prev.right=nothing
        end
    elseif isequal(nodeToRemove.right,nothing) # if the node to remove has only a left child
        if B.root == nodeToRemove # if the node to remove happens to be the root.
            B.root = B.root.left
            B.root.prev = nothing
        elseif nodeToRemove.prev.left == nodeToRemove # if the node to remove is a left child of its parent node.
            prevNode = nodeToRemove.prev
            prevNode.left = nodeToRemove.left
            prevNode.left.prev = prevNode
        else # otherwise it is a right childof the parent node
            prevNode = nodeToRemove.prev
            prevNode.right = nodeToRemove.left
            prevNode.right.prev = prevNode
        end
    elseif isequal(nodeToRemove.left,nothing) # if the node to remove has only a right child.
        if B.root == nodeToRemove # if this node happens to be the root
            B.root = B.root.right
            B.root.prev = nothing
        elseif nodeToRemove.prev.left == nodeToRemove # if the node to remove is a left child of its parent node.
            prevNode = nodeToRemove.prev
            prevNode.left = nodeToRemove.right
            prevNode.left.prev = prevNode
        else # otherwise it is a right child of the parent node
            prevNode = nodeToRemove.prev
            prevNode.right = nodeToRemove.right
            prevNode.right.prev = prevNode
        end
    else # if the node to remove has two children or is the root node.
        if B.root == nodeToRemove # if the node to remove is the root node (and has two children)
            leaf = B.root.left
            while !isequal(B.right,nothing) # go to the farthes right laft node after going one level left
                leaf = leaf.right
            end
            newData = leaf.data
            remove!(B,leaf.data) # remove the leaf
            self.root.data = newData # set the roots new value
        else # if the node to remove is not the root, follow the same strategy as above.
            leaf = nodeToRemove.left
            while !isequal(B.right,nothing)
                leaf = leaf.right
            end
            newData = leaf.Data
            remove!(B, leaf.Data)
            nodeToRemove.data = newData
        end
    end
end

function draw(B::BST)
    if isequal(B.root,nothing)
        return
    end

    G = SimpleGraph()




end



end
