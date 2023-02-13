#linked_lists.jl

module LinkedLists


abstract type Node end

mutable struct LinedListNode <: Node
    data::Union{Nothing, Int64, Float64, String, Vector{Any}}
    next::Union{Nothing,LinkedListNode}
    prev::Union{Nothing,LinkedListNode}
end

abstract type CustomDataStructure end

mutable struct LinkedList <: CustomDataStructure

    head::Union{Nothing,LinkedListNode}
    tail::Union{Nothing,LinkedListNode}
    nodeCount::Int64

    function LinkedList(head,tail,nodeCount)
        if isnothing(head)
            nodeCount = 0
        else
            nodeCount = 1
        end
        return new(head,tail,nodeCount)
    end
end

mutable struct Deque <: CustomDataStructure

    head::Union{Nothing,LinkedListNode}
    tail::Union{Nothing,LinkedListNode}
    nodeCount::Int64

    function Deque(head,tail,nodeCount)
        if isnothing(head)
            nodeCount = 0
        else
            nodeCount = 1
        end
        return new(head,tail,nodeCount)
    end
end

function append!(LL<:CustomDataStructure, data)

    # initialize the new node using the data
    newNode = LinkedListNode(data,nothing,nothing)

    if isequal(nothing,LL.head)
        # check to see if the linked list is empty
        # if it is assing the new node to be the head and tai
        LL.head = newNode
        LL.tail = newNode
        LL.nodeCount += 1
    else
        # if the linked list is not empty then the new node goes at
        # the end of the list
        LL.tail.next = newNode # connect the new node to the end of the list
        newNode.prev = LL.tail # make it so the previous of the new node is the old tail
        LL.tail = newNode # reassign the tail to be the new node
        LL.nodeCount += 1
    end

    return
end

function find(LL<:CustomDataStructure, data::Union{Nothing, Int64, Float64, String, Vector{Any}})

    # check to see if the LL is empty
    if isequal(LL.head,nothing)
        throw(ArgumentError(LL, "The linked list is empty."))
    end

    # now we search through the list
    # set intiial node
    currentNode = LL.head
    while search
        # checkt to see if the data inside the LLNode is equivalent
        # to the given data
        if currentNode.data == data
            search = false
        else
            # get the next node in the Linked List
            currentNode = currentNode.next
            if isequal(currentNode, nothing)
                search = false
                throw(ArgumentError(LL, "The data is not found in the linked list."))
            end
        end
    end

    return currentNode
end

function get(LL<:CustomDataStructure, index::Int64)

    if index < 1:
        throw(KeyError(index, "Index out of range (too small)."))
    end
    if index > LL.nodeCount:
        throw(KeyError(index, "Index out of range (too large)."))

    end

    currentNode = LL.head
    for i=2:index
        currentNode = currentNode.next
    end

    return currentNode

end



length(LL<:CustomDataStructure) = LL.nodeCount


function repr(LL<:CustomDataStructure)

    currentNode = LL.head
    stringRep = "["

    if LL.nodeCount == 0
        stringRep += "]"
        return repr(stringRep);
    elseif LL.nodeCount == 1
        if typeof(currentNode.data) == String
            stringRep += repr(currentNode.data) + "]";
        else
            stringRep += string(currentNode.data) + "]";
        end
        return stringRep
    end

    for i=1:LL.nodeCount
        if typeof(currentNode.data) == String
            if i == 1
                stringRep += repr(currentNode.data);
            elseif i == LL.nodeCount
                stringRep += ", " repr(currentNode.data) + "]";
            else
                stringRep += ", " repr(currentNode.data);
            end
        else
            if i == 1
                string_rep += string(currentNode.data);
            elseif i == LL.nodeCount
                stringRep += ", " + string(currentNode.data) + "]"''
            else
                stringRep += ", " + string(currentNode.data);
            end
        end
        currentNode = currentNode.next
    end

    return stringRep

end


function remove!(LL<:CustomDataStructure, data::Union{Nothing, Int64, Float64, String, Vector{Any}})

    if typeof(LL) == Deque
        throw(ArgumentError(LL, "Use pop!() or popleft!() for removal."))
    end

    if LL.nodeCount == 0:
        throw(KeyError(LL.nodeCount, "Empty Linked List."))
    end

    currentNode = LL.head

    # special case where the list only has one element
    if LL.nodeCount == 1  && data == currentNode.data
        LL.head = nothing
        LL.tail = nothing
        LL.nodeCount = 0
        return

    # special case where the linked list has more than one element and the data to remove is the first node.
    elseif data == currentNode.data
        LL.head = currentNode.next
        LL.head.prev = nothing
        LL.nodeCount -= 1


    else
        for i=2:LL.nodeCount
            currentNode = currentNode.next
            # special case where data to remove is the final node
            if data == currentNode.data
                if i == LL.nodeCount
                    LL.tail = LL.tail.prev
                    LL.tail.next = nothing
                    LL.nodeCount -= 1
                    return
                else
                    currentNode.prev.next = currentNode.next
                    currentNode.next.prev = currentNode.prev
                    LL.nodeCount -= 1
                    return
                end
            end
        end
    end
    throw(KeyError(data, "Data not found."))
    return
end

function pop!(D::Deque)
    # function remove last node
    if D.nodeCount == 0 # edge case if deque is empty
        throw(ArgumentError(D, "Index out of range"))
    elseif D.nodeCount == 1 #edge case if deque only has one node
        nodeData = D.tail.data
        D.tail = nothing
        D.head= nothing
        D.nodeCount -= 1
    else: # otherwise reset the tail value
        nodeData = D.tail.data
        D.tail = D.tail.prev
        D.tail.next = nothing
        D.nodeCount -= 1
    end
    return nodeData
end

function popleft!(D::Deque)
    # function to remove first node
    if D.nodeCount == 0 # edge case if our deque is empty
        throw(ArgumentError(D, "Index out of range"))
    elseif D.nodeCount == 1 # edge case if our deque only has one elemnt
        nodeData = D.head.data
        D.tail = nothing
        D.head = nothing
        D.nodeCount -= 1
    else # otherwise we can just use remove.
        nodeData = D.head.data
        remove!(D, nodeData)
    end

    return nodeData
end


function insert!(LL<:CustomDataStructure, index::Int64, data)
    if typeof(LL) == Deque
        throw(ArgumentError(LL, "Use append!() or appendleft!() for insertion."))
    end

    if index < 1 || index > LL.nodeCount + 1
        throw(KeyError(index, "index out of range."))

    end

    # create the new node
    newNode = LinkedListNode(data, nothing,nothing)
    # special case where we are inserting and resetting the head and the LL is not empty.
    if index == 1 and LL.nodeCount > 0
        newNode.next = LL.head
        newNode.next.prev = newNode
        LL.head = newNode
        LL.nodeCount += 1

    # special case where we are inserting and resetting the head and the LL is empty.
    elseif index == 1 and LL.nodeCount == 0
        LL.head = newNode
        LL.tail = newNode
        LL.nodeCount += 1

   # if the insert is just adding on a new node then we can just append
   elseif index == LL.nodeCount + 1
        append!(LL, data)
   # the final case where the insertion point is in the middle of the Linked List
   else
        currentNode = LL.head
        for i=1:index-1
            currentNode = currentNode.next
        end
        newNode.prev = currentNode
        newNode.next = currentNode.next
        currentNode.next = newNode
        newNode.next.prev = newNode
        LL.nodeCount += 1
    end
    return
end

function appendleft!(D::Deque, data)
    insert!(D,0,data)
    return
end
