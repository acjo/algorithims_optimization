# linked_lists.jl
module LinkedLists

import Base.repr, Base.append!, Base.==, Base.insert!

export LinkedListNode, LinkedList, Deque, append!, find, get, length, repr, remove!, pop!, popleft!, insert!, appendleft!, ==

# abstract type Node end
# mutable struct LinkedListNode <: Node
mutable struct LinkedListNode
    data::Union{Nothing, <:Number, String, Vector{<:Any}}
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
        if isnothing(head) && isnothing(tail)
            nodeCount = 0
        else
            nodeCount = 1
        end
        return new(head,tail,nodeCount)
    end
end

function append!(DS::CustomDataStructure, data)

    # initialize the new node using the data
    newNode = LinkedListNode(data,nothing,nothing)

    if isequal(DS.head,nothing)
        # check to see if the linked list is empty
        # if it is assing the new node to be the head and tai
        DS.head = newNode
        DS.tail = newNode
        DS.nodeCount += 1
    else
        # if the linked list is not empty then the new node goes at
        # the end of the list
        DS.tail.next = newNode # connect the new node to the end of the list
        newNode.prev = DS.tail # make it so the previous of the new node is the old tail
        DS.tail = newNode # reassign the tail to be the new node
        DS.nodeCount += 1
    end

    return
end

function find(DS::CustomDataStructure, data::Union{Nothing, <:Number, String, Vector{<:Any}})

    # check to see if the DS is empty
    if isequal(DS.head,nothing)
        throw(KeyError(data))
    end
    # now we search through the list
    # set intiial node
    currentNode = DS.head
    search=true
    while search
        # checkt to see if the data inside the DSNode is equivalent
        # to the given data
        if currentNode.data == data
            search = false
        else
            # get the next node in the Linked List
            currentNode = currentNode.next
            if isequal(currentNode, nothing)
                search = false
                throw(KeyError(data))
            end
        end
    end
    return currentNode
end

function get(DS::CustomDataStructure, index::Int64)

    if index < 1
        throw(KeyError(index))
    end
    if index > DS.nodeCount
        throw(KeyError(index))
    end

    currentNode = DS.head
    for i=2:index
        currentNode = currentNode.next
    end

    return currentNode
end


length(DS::CustomDataStructure) = DS.nodeCount

function ==(CDS1::CustomDataStructure, CDS2::CustomDataStructure)


    if !isequal(typeof(CDS1),typeof(CDS2))
        return false
    elseif CDS1.nodeCount ≠ CDS2.nodeCount
        return false
    else
        currentNode1 = CDS1.head
        currentNode2 = CDS2.head
        for i=2:CDS1.nodeCount
            if currentNode1.data ≠ currentNode2.data
                return false
            end
                currentNode1 = currentNode1.next
                currentNode2 = currentNode2.next

        end
    end
    return true
end

function repr(DS::CustomDataStructure)

    currentNode = DS.head
    stringRep = "["

    if DS.nodeCount == 0
        stringRep *= "]"
        return Base.repr(stringRep);
    elseif DS.nodeCount == 1
        if typeof(currentNode.data) == String
            stringRep *= Base.repr(currentNode.data) * "]";
        else
            stringRep *= string(currentNode.data) * "]";
        end
        return stringRep
    end

    for i=1:DS.nodeCount
        if typeof(currentNode.data) == String
            if i == 1
                stringRep *= Base.repr(currentNode.data);
            elseif i == DS.nodeCount
                stringRep *= ", " * Base.repr(currentNode.data) * "]";
            else
                stringRep *= ", " * Base.repr(currentNode.data);
            end
        else
            if i == 1
                string_rep *= string(currentNode.data);
            elseif i == DS.nodeCount
                stringRep *= ", " * string(currentNode.data) * "]"''
            else
                stringRep *= ", " * string(currentNode.data);
            end
        end
        currentNode = currentNode.next
    end

    return stringRep
end


function remove!(DS::CustomDataStructure, data::Union{Nothing,<:Number,String, Vector{<:Any}})

    if typeof(DS) == Deque
        throw(ArgumentError("Use pop!() or popleft!() for removal."))
    end

    if DS.nodeCount == 0
        throw(KeyError(DS.nodeCount))
    end

    currentNode = DS.head

    # special case where the list only has one element
    if DS.nodeCount == 1  && data == currentNode.data
        DS.head = nothing
        DS.tail = nothing
        DS.nodeCount = 0
        return

    # special case where the linked list has more than one element and the data to remove is the first node.
    elseif data == currentNode.data
        DS.head = currentNode.next
        DS.head.prev = nothing
        DS.nodeCount -= 1
        return

    else
        for i=2:DS.nodeCount
            currentNode = currentNode.next
            # special case where data to remove is the final node
            if data == currentNode.data
                if i == DS.nodeCount
                    DS.tail = DS.tail.prev
                    DS.tail.next = nothing
                    DS.nodeCount -= 1
                    return
                else
                    currentNode.prev.next = currentNode.next
                    currentNode.next.prev = currentNode.prev
                    DS.nodeCount -= 1
                    return
                end
            end
        end
    end
    throw(KeyError(data))
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
    else # otherwise reset the tail value
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


function insert!(DS::CustomDataStructure, index::Int64, data)
    if typeof(DS) == Deque
        throw(ArgumentError(DS, "Use append!() or appendleft!() for insertion."))
    end

    if index < 1 || index > DS.nodeCount + 1
        throw(KeyError(index, "index out of range."))

    end

    # create the new node
    newNode = LinkedListNode(data, nothing,nothing)
    # special case where we are inserting and resetting the head and the DS is not empty.
    if index == 1 && DS.nodeCount > 0
        newNode.next = DS.head
        newNode.next.prev = newNode
        DS.head = newNode
        DS.nodeCount += 1

    # special case where we are inserting and resetting the head and the DS is empty.
    elseif index == 1 && DS.nodeCount == 0
        DS.head = newNode
        DS.tail = newNode
        DS.nodeCount += 1

   # if the insert is just adding on a new node then we can just append
   elseif index == DS.nodeCount + 1
        append!(DS, data)
   # the final case where the insertion point is in the middle of the Linked List
   else
        currentNode = DS.head
        for i=1:index-1
            currentNode = currentNode.next
        end
        newNode.prev = currentNode
        newNode.next = currentNode.next
        currentNode.next = newNode
        newNode.next.prev = newNode
        DS.nodeCount += 1
    end
    return
end

function appendleft!(D::Deque, data)
    insert!(D,0,data)
    return
end
end
