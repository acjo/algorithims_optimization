# test.jl

include("binary_trees.jl")
using .BinaryTrees


root = BSTNode(5);

tree = BST(root);

insert!(tree, 2);
insert!(tree, 7);
insert!(tree, 1);
insert!(tree, 6);
insert!(tree, 8);


draw(tree);
