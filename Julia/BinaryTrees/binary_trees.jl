# binary_trees.jl

module BinaryTrees

import Base.append!, Base.insert!

using Plots, Graphs, GraphRecipes

export SinglyLinkedListNode, SinglyLinkedList, append!, iterativeFind,recursiveFind, BSTNode, BST, AVL, find, insert!, remove!, draw

mutable struct SinglyLinkedListNode
    """
    A node with a value and a reference to the next node
    """
    data::Union{<:Number, String, Vector{<:Any}}
    next::Union{Nothing,SinglyLinkedListNode}

    function SinglyLinkedListNode(data=nothing, next=nothing)
        """
        constructor if no data and no next node is passed in. I.e. SinglyLinkedListNode()
        """
        return new(data, next)
    end
    function SinglyLinkedListNode(data; next=nothing)
        """
        constructor if no next node is passed in. I.e. SinglyLinkedListNode([3])
        """
        return new(data, next)
    end
    function SinglyLinkedListNode(data,next)
        """
        Constructor if all information is passed in I.e. SinglyLinkedListNode([3], nextNode)
        """
        return new(data,next)
    end
end

mutable struct SinglyLinkedList
    """
    A singly linekd list with a head and a tail
    """
    head::Union{Nothing, SinglyLinkedListNode}
    tail::Union{Nothing, SinglyLinkedListNode}
    function SinglyLinkedList(;head=nothing, tail=nothing)
        """
        Constructor if no data is passed in i.e. SinglyLinkedList()
        """
        return new(head,tail)
    end
    function SinglyLinkedList(head;tail=nothing)
        """
        Constructor if only the head is passed in.
        """
        return new(head, head)
    end
    function SinglyLinkedList(head, tail)
        """
        Constructor if all the data is passed in: i.e. SinglyLinkedList(head, tail)
        """
        if isequal(head,nothing) && isequal(tail,nothing)
            return new(nothing, nothing)
        elseif isequal(head.next, nothing)
            throw(ArgumentError("The head is not connected to the tail"))
        end
        return new(head,tail)
    end

end

function append!(SLL::SinglyLinkedList, data)
    """
    Add a node containing th data to the end of the linked list.
    """
    # create the node
    newNode = SinglyLinkedListNode(data)

    if isequal(SLL.head,nothing) # if the linked list is empty.
        SLL.head=newNode
        SLL.tail=newNode
    else # otherwise add the node to the end of the list.
        SLL.tail.next = newNode
        SLL.tail = newNode
    end

    return
end

function iterativeFind(SLL::SinglyLinkedList, data)
    """
    Search iteratively for a node containing the data.
    If there is no such node in the list, including if the list is empty,
    throw a key error.
    """

    currentNode = SLL.head # start with the head node
    while !isequal(currentNode, nothing)
        # check if the data is in the current node and return the node if it is
        if currentNode.data == data
            return currentNode
        end
        # otherwise go to the next node
        currentNode = currentNode.next
    end

    # throw a key error if it is not found
    throw(KeyError(data))

    return
end

function recursiveFind(SLL::SinglyLinkedList, data)
    """
    We search recursively for the node containing the data and return the node.
    If there is no such node found or the list is empty a key error is thrown.
    """

    function isNode(currentNode::SinglyLinkedListNode)
        """
        This is the function used to recursively find the node
        """

        # if the current node is nothing then the list is empty (because of function construction -2nd elseif-
        # this if will only be activated on the head node)
        if isequal(currentNode,nothing)
            throw(KeyError(data))
        # check if the current node contains the data and return it
        elseif currentNode.data == data
            return currentNode
        # if the next node is nothing that means the data can't be contained in the list
        elseif isequal(currentNode.next,nothing)
            throw(KeyError(data))
        else # if none of the above happen, call this function on the next node
            return isNode(currentNode.next)
        end
    end
    # call function on head node.
    return isNode(SLL.head)
end

mutable struct BSTNode
    """
    A node class for binary search and avl trees. Contains a value,
    A reference to the parent node, and references to two child nodes.
    """
    data::Union{Nothing, <:Number, String, Vector{<:Any}}
    prev::Union{Nothing, BSTNode}
    left::Union{Nothing, BSTNode}
    right::Union{Nothing, BSTNode}

    function BSTNode(;data=nothing, prev=nothing,left=nothing,right=nothing)
        """
        Constructor if no data is provided. I.e. BSTNode()
        """
        return new(data,prev,left,right)
    end
    function BSTNode(data; prev=nothing,left=nothing,right=nothing)
        """
        Constructor if only the data is provided. I.e. BSTNode(data)
        """
        return new(data,prev,left,right)
    end
    function BSTNode(data,prev,left,right)
        """
        Constructor if all the data is provided. i.e. BSTNode(data,prev,left,right)
        """
        return new(data,prev,left,right)
    end
end

#=
TODO: Consider the following structure as a
replacement for the current architecture:

abstract type Tree end

mutable struct TreeStructureTest <: Tree
    root::Union{Nothing, BSTNode}
    nodeCount::Int64
end

mutable struct BST{T::TreeStructureTest}
    structure::TreeStructureTest
end

mutable struct AVL{T::TreeStructureTest}
    structure::TreeStructureTest

mutable struct FullTree{T<:TreeStructureTest}
    structure::T
    help::String
end
=#


# abstract type to build the BST and AVL structs off of
abstract type TreeStructure end

mutable struct BST <: TreeStructure
    """
    binary search tree data structure class.
    The root attribute references the first node in the tree.
    The node count is the length of the tree.
    """
    root::Union{Nothing, BSTNode}
    nodeCount::Int64

    function BST(;root=nothing,nodeCount=nothing)
        """
        Constructor if no data is provide. i.e. BST()
        """
        return new(root, 0)
    end

    function BST(root; nodeCount=0)
        """
            Constructor if the root is provided. i.e. BST(root).
        """
        if isequal(root,nothing)
            nodeCount = 0
        end
        return new(root,nodeCount)
    end

    function BST(root,nodeCount)
        """
        Constructor if the root and the node count is provided. i.e. BST(root, 3)
        Can be helpful if root already is linked to it's children.
        """
        if isequal(root,nothing)
            nodeCount = 0
        end
        return new(root,nodeCount)
    end
end

mutable struct AVL <: TreeStructure
    root::Union{Nothing, BSTNode}
    nodeCount::Int64

    function AVL(;root=nothing,nodeCount=nothing)
        """
        Constructor if no data is provide. i.e. AVL()
        """
        return new(root, 0)
    end
    function AVL(root; nodeCount=0)
        """
            Constructor if the root is provided. i.e. AVL(root).
        """
        if isequal(root,nothing)
            nodeCount = 0
        end
        newAVL = new(root,nodeCount)
        return _rebalance!(newAVL, newAVL.root)
    end

    function AVL(root,nodeCount)
        """
        Constructor if the root and the node count is provided. i.e. AVL(root, 3)
        Can be helpful if root already is linked to it's children.
        """
        if isequal(root,nothing)
            nodeCount = 0
        end

        return new(root,nodeCount)
    end
end


function find(B::TreeStructure, data)
    """
        Return the node containing the data. If there is no such node in the tree,
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

function _rebalance!(A::AVL, currentNode::Union{BSTNode, Nothing})
    # rebalance the subtree starting at the specified node

    balance = _balanceFactor(currentNode)

    if balance == -2
        if _height(currentNode.left.left) > _height(currentNode.left.right)
            currentNode = _rotate_left_left!(A, currentNode)
        else
            currentNode = _rotate_left_right!(A, currentNode)
        end
    elseif balance == 2
        if _height(currentNode.right.right) > _height(currentNode.right.left)
            currentNode = _rotate_right_right!(A, currentNode)
        else
            currentNode = _rotate_right_left!(A, currentNode)
        end
    end
    return currentNode
end


function _height(currentNode::Union{BSTNode, Nothing})

    if isequal(currentNode, nothing)
        return -1
    end

    return 1 + maximum((_height(currentNode.right), _height(currentNode.left)))
end

function _balanceFactor(currentNode::BSTNode)
    return _height(currentNode.right) - _height(currentNode.left)
end

function _rotate_left_left!(A::AVL, currentNode::Union{BSTNode, Nothing})
    temp = currentNode.left
    currentNode.left = temp.right
    if !isequal(temp.right, nothing)
        temp.right.prev = currentNode
    end

    temp.right = currentNode
    temp.prev = currentNode.prev
    currentNode.prev = temp

    if !isequal(temp.prev,nothing)
        if temp.prev.data > temp.data
            temp.prev.left = temp
        else
            temp.prev.right = temp
        end
    end

    if isequal(currentNode, A.root)
        A.root = temp
    end
    return temp
end

function _rotate_right_right!(A::AVL, currentNode::Union{BSTNode, Nothing})
    temp = currentNode.right
    currentNode.right = temp.left

    if !isequal(temp.left, nothing)
        temp.left.prev = currentNode
    end
    temp.left = currentNode
    temp.prev = currentNode.prev
    currentNode.prev = temp

    if !isequal(temp.prev,nothing)
        if temp.prev.data > temp.data
            temp.prev.left = temp
        else
            temp.prev.right = temp
        end
    end

    if isequal(currentNode,A.root)
        A.root = temp
    end

    return temp
end

function _rotate_left_right!(A::AVL, currentNode::Union{BSTNode, Nothing})
    temp1 = currentNode.left
    temp2 = temp1.right
    temp1.right = temp2.left

    if !isequal(temp2.left,nothing)
        temp2.left.prev = temp1
    end

    temp2.prev = currentNode
    temp2.left = temp1
    temp1.prev = temp2
    currentNode.left = temp2
    return _rotate_left_left!(A, currentNode)
end

function _rotate_right_left!(A::AVL, currentNode::Union{BSTNode, Nothing})
    temp1 = currentNode.right
    temp2 = temp1.left
    temp1.left = temp2.right

    if !isequal(temp2.right,nothing)
        temp2.right.prev = temp1
    end
    temp2.prev =currentNode
    temp2.right = temp1
    temp1.prev = temp2
    currentNode.right = temp2
    return _rotate_right_right!(A,currentNode)
end


function insert!(B::TreeStructure,data)
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
            if isequal(B.root, nothing) # if there is no root, the BST is empty, make the newnode the root
                B.root = BSTNode(data)
                B.nodeCount += 1
            else # otherwise call the step function to insert the node
                _step(B.root)
                B.nodeCount += 1
            end

            if isequal(typeof(B), AVL)
                node = find(B ,data)
                while !isequal(node,nothing)
                    node = _rebalance!(B,node).prev
                end
            end

        else # if the wrong error is caught, throw the error again.
            throw(e)
        end
    else
        # if no error is caught at all that means the data is already contained in the true.
        throw(ArgumentError("Data already contained in tree, no duplicates allowed."))
    end

    return
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
    B.nodeCount -= 1

    return
end


function draw(B::TreeStructure)
    if isequal(B.root,nothing)
        return
    end

    nodes = [B.root]
    edges = []
    while length(nodes) > 0
        currentNode = pop!(nodes)
        for childNode in [currentNode.left, currentNode.right]
            if !isequal(childNode,nothing)
                append!(edges, [(currentNode.data, childNode.data)])
                # add_edge!(G,currentNode.data, childNode.data)
                append!(nodes, [childNode])
            end
        end
    end

    el = Edge.(edges)

    G = SimpleDiGraph(el)

    graphplot(G,curves=false)

end

end
